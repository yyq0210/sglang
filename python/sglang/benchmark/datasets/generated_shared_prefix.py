import math
import pickle
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
)


def _reorder_lengths_by_popularity(
    lengths: List[int], target_corr: float, seed: int
) -> List[int]:
    """Permute per-group prefix lengths so that group popularity rank and
    prefix length attain an approximate Pearson correlation of ``target_corr``.

    Group index ``g`` is the zipf rank (group 0 is the hottest), so a group's
    popularity value is ``(N - 1 - g)`` (higher = hotter). We assign the fixed
    multiset of ``lengths`` to group indices via a Gaussian copula: each group
    gets a latent score

        score_g = rho * z_pop_g + sqrt(1 - rho**2) * z_noise_g

    where ``z_pop_g`` is the normal quantile of the group's popularity rank and
    ``z_noise_g`` is the normal quantile of an isolated random permutation.
    Sorting groups by score and handing the longest lengths to the highest
    scores yields ``corr(popularity, length) ~= rho`` (exact in expectation;
    the offline test asserts the achieved correlation tracks the target).

    Uses an isolated ``numpy.random.default_rng(seed)`` so the module-level
    random / numpy.random state that gen_prompt / compute_random_lens rely on
    is never perturbed -- the system-prompt and question pools stay
    byte-identical to the len_pop_corr=None path for the same seed.
    """
    n = len(lengths)
    if n <= 1:
        return list(lengths)
    rho = float(target_corr)

    rng = np.random.default_rng(seed)
    # popularity rank: group 0 hottest -> highest popularity value (n-1).
    pop_value = np.arange(n - 1, -1, -1, dtype=np.float64)  # [n-1, ..., 0]
    # pop_value is already a permutation of {0, ..., n-1} (each integer used
    # exactly once), so its own value IS its ascending rank -- no argsort
    # needed. Normal quantile of that rank's plotting position (ties-free).
    z_pop = _normal_quantile((pop_value + 0.5) / n)

    noise_order = rng.permutation(n)
    z_noise = _normal_quantile((noise_order + 0.5) / n)

    score = rho * z_pop + math.sqrt(max(0.0, 1.0 - rho * rho)) * z_noise

    sorted_lengths = np.sort(np.asarray(lengths))  # ascending
    # highest score gets the longest length
    groups_by_score = np.argsort(score)  # ascending score -> shortest length
    out = np.empty(n, dtype=sorted_lengths.dtype)
    out[groups_by_score] = sorted_lengths
    return out.tolist()


def _normal_quantile(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (probit) via Acklam's rational approximation.

    Avoids a scipy dependency. Max abs error ~1.15e-9 on (0,1), which is far
    below the correlation tolerances used here.
    """
    p = np.asarray(p, dtype=np.float64)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    out = np.empty_like(p)

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    q = np.sqrt(-2 * np.log(p[lo])) if lo.any() else np.array([])
    if lo.any():
        out[lo] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out


def _zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    """Rank-based Zipf probability vector with rank starting at 1.

    weight(rank)      = 1 / rank ** alpha       (rank in 1..num_groups)
    probability(rank) = weight(rank) / sum_over_all_ranks(weight)

    The returned array has length num_groups; element i corresponds to
    group index i (rank i + 1), so group 0 is the hottest.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / (ranks**alpha)
    return weights / weights.sum()


@dataclass
class GeneratedSharedPrefixDataset(BaseDataset):
    num_groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    range_ratio: float
    seed: int
    fast_prepare: bool
    send_routing_key: bool
    num_turns: int
    ordered: bool
    group_distribution: str = "uniform"
    zipf_alpha: Optional[float] = None
    len_pop_corr: Optional[float] = None

    @classmethod
    def from_args(cls, args: Namespace) -> "GeneratedSharedPrefixDataset":
        assert not getattr(args, "tokenize_prompt", False)
        group_distribution = getattr(args, "gsp_group_distribution", "uniform")
        zipf_alpha = getattr(args, "gsp_zipf_alpha", None)
        len_pop_corr = getattr(args, "gsp_len_pop_corr", None)

        # Defensive validation for in-process callers that construct a
        # Namespace by hand and bypass the argparse boundary in
        # serving.py. The CLI hook enforces the same rules first.
        if group_distribution not in ("uniform", "zipf"):
            raise ValueError(
                f"--gsp-group-distribution must be 'uniform' or 'zipf', "
                f"got {group_distribution!r}"
            )
        if group_distribution == "zipf":
            if zipf_alpha is None:
                raise ValueError(
                    "--gsp-group-distribution=zipf requires --gsp-zipf-alpha "
                    "(a finite float > 0)"
                )
            if not math.isfinite(zipf_alpha) or zipf_alpha <= 0:
                raise ValueError(
                    f"--gsp-zipf-alpha must be a finite float > 0, got {zipf_alpha!r}"
                )
        elif zipf_alpha is not None:
            raise ValueError(
                "--gsp-zipf-alpha is only meaningful with "
                "--gsp-group-distribution=zipf; remove --gsp-zipf-alpha "
                "or set --gsp-group-distribution=zipf"
            )

        if len_pop_corr is not None:
            if not math.isfinite(len_pop_corr) or not (-1.0 <= len_pop_corr <= 1.0):
                raise ValueError(
                    f"--gsp-len-pop-corr must be a finite float in [-1, 1], "
                    f"got {len_pop_corr!r}"
                )
            if group_distribution != "zipf":
                raise ValueError(
                    "--gsp-len-pop-corr is only meaningful with "
                    "--gsp-group-distribution=zipf (popularity is defined by "
                    "the zipf rank); set --gsp-group-distribution=zipf or "
                    "remove --gsp-len-pop-corr"
                )

        return cls(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            seed=args.seed,
            fast_prepare=getattr(args, "gsp_fast_prepare", False),
            send_routing_key=getattr(args, "gsp_send_routing_key", False),
            num_turns=getattr(args, "gsp_num_turns", 1),
            ordered=getattr(args, "gsp_ordered", False),
            group_distribution=group_distribution,
            zipf_alpha=zipf_alpha,
            len_pop_corr=len_pop_corr,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_generated_shared_prefix_requests(
            num_groups=self.num_groups,
            prompts_per_group=self.prompts_per_group,
            system_prompt_len=self.system_prompt_len,
            question_len=self.question_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            seed=self.seed,
            send_routing_key=self.send_routing_key,
            num_turns=self.num_turns,
            fast_prepare=self.fast_prepare,
            ordered=self.ordered,
            group_distribution=self.group_distribution,
            zipf_alpha=self.zipf_alpha,
            len_pop_corr=self.len_pop_corr,
        )


def get_gen_prefix_cache_path(
    seed: int,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
    len_pop_corr: Optional[float] = None,
):
    """Create cache directory under ~/.cache/sglang/benchmark.

    The uniform-mode filename is preserved exactly as before so existing
    on-disk caches remain valid. Non-default sampling modes get an extra
    suffix encoding the parameters that affect the cached payload.
    """
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    suffix = ""
    if group_distribution != "uniform":
        suffix = f"_{group_distribution}_{zipf_alpha}"
    if len_pop_corr is not None:
        suffix += f"_lpc{len_pop_corr}"

    cache_key = (
        f"gen_shared_prefix_{seed}_{num_groups}_{prompts_per_group}_"
        f"{system_prompt_len}_{question_len}_{output_len}{suffix}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    send_routing_key: bool = False,
    num_turns: int = 1,
    fast_prepare: bool = False,
    ordered: bool = False,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
    len_pop_corr: Optional[float] = None,
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts using random tokens and caching.

    When group_distribution is "uniform" (default), each group receives exactly
    prompts_per_group requests; behavior matches the legacy generator.

    When group_distribution is "zipf", each request's group is sampled by rank
    with probability 1/rank**zipf_alpha / sum_k(1/k**zipf_alpha); rank starts at
    1 and group index 0 is the hottest. Sampling uses an isolated
    numpy.random.default_rng(seed) so the shared question/system-prompt pool
    stays byte-identical to uniform mode for the same seed and other args.
    Zipf mode is cached on disk under a distinct key per (group_distribution,
    zipf_alpha) value.
    """
    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
        group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
        len_pop_corr=len_pop_corr,
    )
    # range_ratio != 1 / num_turns > 1 perturb the payload but are not in the
    # cache key; send_routing_key embeds a per-run uuid + timestamp that is
    # meaningless to cache. Bypass for these pre-existing reasons only.
    should_cache = range_ratio == 1 and not send_routing_key and num_turns == 1

    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not should_cache:
        print(f"\nCache bypassed ({range_ratio=}, {send_routing_key=}, {num_turns=})")

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, {num_turns=}, {group_distribution=}, {zipf_alpha=})"
    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    system_prompt_lens = compute_random_lens(
        full_len=system_prompt_len,
        range_ratio=range_ratio,
        num=num_groups,
    )
    if len_pop_corr is not None:
        # Reassign the fixed multiset of shared-prefix lengths across group
        # ranks so that popularity (zipf rank) and prefix length attain the
        # requested correlation. Requires range_ratio < 1 for length variance;
        # with range_ratio == 1 all lengths are equal and the reorder is a
        # no-op (correlation is undefined / 0).
        system_prompt_lens = _reorder_lengths_by_popularity(
            system_prompt_lens, len_pop_corr, seed
        )
    question_lens = np.array(
        compute_random_lens(
            full_len=question_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group * num_turns,
        )
    ).reshape(num_groups, prompts_per_group, num_turns)
    output_lens = np.array(
        compute_random_lens(
            full_len=output_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group,
        )
    ).reshape(num_groups, prompts_per_group)
    del system_prompt_len, question_len, output_len

    system_prompts = [
        gen_prompt(tokenizer, system_prompt_lens[i]) for i in range(num_groups)
    ]

    # shape: (num_groups, prompts_per_group, num_turns)
    questions = [
        [
            [
                gen_prompt(tokenizer, int(question_lens[g, p, t]))
                for t in range(num_turns)
            ]
            for p in range(prompts_per_group)
        ]
        for g in range(num_groups)
    ]

    # Per-slot group assignment. Uniform mode is the identity assignment
    # [0,0,...,1,1,...,N-1,N-1]; zipf mode samples from the rank distribution
    # using an isolated RNG so the module-level random / numpy.random state
    # that compute_random_lens / gen_prompt rely on is never perturbed -- this
    # keeps the system-prompt and question pool byte-identical to uniform mode
    # for the same seed and other args.
    total_slots = num_groups * prompts_per_group
    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    else:  # "zipf"
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        assignment = rng.choice(num_groups, size=total_slots, replace=True, p=probs)

    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    for slot_idx, sampled_g in enumerate(
        tqdm(assignment, desc="Generating shared-prefix prompts")
    ):
        # src_(g,p) walks the question pool in uniform-enumeration order, so
        # per-slot question text is reproducibly identical across modes.
        src_g, src_p = divmod(slot_idx, prompts_per_group)
        sampled_g = int(sampled_g)

        system_prompt = system_prompts[sampled_g]
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{sampled_g}"
            if send_routing_key
            else None
        )
        turn_questions = questions[src_g][src_p]
        turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
        full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
        prompt_len = 1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
        output_len_val = int(output_lens[src_g, src_p])

        input_requests.append(
            DatasetRow(
                prompt=full_prompt,
                prompt_len=prompt_len,
                output_len=output_len_val,
                routing_key=routing_key,
            )
        )
        total_input_tokens += prompt_len
        total_output_tokens += output_len_val

    if not ordered:
        random.shuffle(input_requests)

    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Number of turns: {num_turns}")
    print(f"Group distribution: {group_distribution}")
    if group_distribution == "zipf":
        print(f"Zipf alpha: {zipf_alpha}")
    if len_pop_corr is not None:
        achieved = float(
            np.corrcoef(
                np.arange(num_groups - 1, -1, -1, dtype=np.float64),
                np.asarray(system_prompt_lens, dtype=np.float64),
            )[0, 1]
        )
        print(f"Length-popularity corr: target={len_pop_corr} achieved={achieved:.3f}")
    print(f"Total prompts: {len(input_requests)}")
    if not fast_prepare:
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(
            f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
        )
        all_questions = [q for group in questions for conv in group for q in conv]
        print(
            f"Average question length: {sum(len(tokenizer.encode(q)) for q in all_questions) / len(all_questions):.1f} tokens\n"
        )

    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

    return input_requests
