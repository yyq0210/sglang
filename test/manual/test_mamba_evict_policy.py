"""GATE-G1 (offline, no GPU): value-aware mamba checkpoint eviction policy.

Exercises the REAL TreeNode + LRUList + MambaRadixCache.evict_mamba /
_evict_mamba_value victim-selection logic. Only the physical leaf-free mechanics
(`_evict_leaf_node`, `_free_mamba_value`, `_tombstone_internal_node`) are stubbed,
since those touch GPU pools; the selection order under test is fully real.

Asserts:
  (1) "lru"   policy evicts by pure recency (oldest first)   -- byte-identical to prior.
  (2) "value" policy evicts lowest-value first, value=(hit_count, last_access_time).
  (3) Under recency/frequency ANTI-correlation (a popular prefix cached early), the
      value policy RETAINS the high-frequency checkpoints that LRU wrongly evicts.
  (4) Locked checkpoints (mamba_lock_ref>0) are never evicted by the value policy.
  (5) The value path handles the internal-node (tombstone) branch.
  (6) "gdsf" cost-aware policy evicts lowest (hit_count+1)*prefix_len first.
  (7) Under freq/length anti-alignment, gdsf RETAINS more reconstruction-cost mass
      than value (keeps expensive-to-rebuild hot prefixes value would evict).
  (8) Under UNIFORM prefix length, gdsf == value (value is the special case).
  (9) gdsf never evicts locked checkpoints.
 (10) gdsf aging clock is monotonic non-decreasing across evictions.
 (11) gdsf handles the internal-node (tombstone) branch.

Run: python test/manual/test_mamba_evict_policy.py
"""

import types

from sglang.srt.mem_cache.mamba_radix_cache import (
    LRUList,
    MambaRadixCache,
    TreeNode,
)


def _make_cache(policy: str) -> MambaRadixCache:
    """A minimal MambaRadixCache with only the fields evict_mamba touches, plus
    stubbed physical-free mechanics (no GPU pools)."""
    c = MambaRadixCache.__new__(MambaRadixCache)
    c.disable = False
    c.mamba_evict_policy = policy
    c.root_node = TreeNode()
    c.full_lru_list = LRUList(mamba=False)
    c.mamba_lru_list = LRUList(mamba=True)
    # fields the cost-aware "gdsf" path reads (harmless for lru/value):
    c.page_size = 1
    c.gdsf_clock = 0.0

    def _evict_leaf_node(self, x, is_evict_mamba):
        n = len(x.mamba_value)
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = None
        self.mamba_lru_list.remove_node(x)
        if x.id in self.full_lru_list.cache:
            self.full_lru_list.remove_node(x)
        return 0, n, x, x_next

    def _free_mamba_value(self, value):
        return None

    def _tombstone_internal_node(self, x):
        # mirror the real invariant: a tombstoned node has no mamba value
        x.mamba_value = None

    c._evict_leaf_node = types.MethodType(_evict_leaf_node, c)
    c._free_mamba_value = types.MethodType(_free_mamba_value, c)
    c._tombstone_internal_node = types.MethodType(_tombstone_internal_node, c)
    return c


def _add_leaf(cache: MambaRadixCache, hit_count: int, lock: int = 0) -> TreeNode:
    """Insert a leaf checkpoint (1 mamba token) as most-recently-used."""
    n = TreeNode()
    n.parent = cache.root_node
    n.mamba_value = [n.id]  # length-1, uniquely identifiable
    n.hit_count = hit_count
    n.mamba_lock_ref = lock
    cache.mamba_lru_list.insert_mru(n)
    return n


def _add_leaf_cost(
    cache: MambaRadixCache, hit_count: int, prefix_len: int, lock: int = 0
) -> TreeNode:
    """Leaf with a controllable reconstruction cost. `_recon_cost` walks node->root
    summing len(hash_value)*page_size; a leaf directly under root has cost =
    len(hash_value) (page_size=1). So set hash_value to `prefix_len` page hashes."""
    n = _add_leaf(cache, hit_count=hit_count, lock=lock)
    n.hash_value = ["h"] * prefix_len
    return n


def _resident_ids(cache: MambaRadixCache):
    return set(cache.mamba_lru_list.cache.keys())


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_lru_evicts_oldest_regardless_of_frequency():
    c = _make_cache("lru")
    # insert oldest..newest; give the OLD ones high hit_count (adversarial for LRU)
    nodes = [_add_leaf(c, hit_count=100 - i) for i in range(6)]  # node0 oldest, hottest
    evicted = c.evict_mamba(3)
    _assert(evicted == 3, f"lru evicted count {evicted} != 3")
    resident = _resident_ids(c)
    # LRU removes the 3 oldest = nodes 0,1,2 (the hottest) -- pure recency
    _assert(nodes[0].id not in resident, "lru should evict oldest node0")
    _assert(nodes[1].id not in resident, "lru should evict node1")
    _assert(nodes[2].id not in resident, "lru should evict node2")
    _assert({nodes[3].id, nodes[4].id, nodes[5].id} <= resident, "lru kept newest 3")
    print("  [ok] (1) lru evicts pure recency (oldest first)")


def test_value_evicts_lowest_frequency():
    c = _make_cache("value")
    nodes = [_add_leaf(c, hit_count=100 - i) for i in range(6)]  # node5 coldest
    evicted = c.evict_mamba(3)
    _assert(evicted == 3, f"value evicted count {evicted} != 3")
    resident = _resident_ids(c)
    # value removes the 3 lowest hit_count = nodes 5,4,3
    _assert(nodes[5].id not in resident, "value should evict coldest node5")
    _assert(nodes[4].id not in resident, "value should evict node4")
    _assert(nodes[3].id not in resident, "value should evict node3")
    _assert({nodes[0].id, nodes[1].id, nodes[2].id} <= resident, "value kept hottest 3")
    print("  [ok] (2) value evicts lowest frequency first")


def test_value_beats_lru_under_anticorrelation():
    # Popular prefixes cached early (oldest) -> LRU evicts exactly the wrong ones.
    def build(policy):
        c = _make_cache(policy)
        # node i: inserted i-th (older = smaller i), hit_count = 20 - i (older = hotter)
        ns = [_add_leaf(c, hit_count=20 - i) for i in range(20)]
        c.evict_mamba(10)  # evict half the pool
        retained = _resident_ids(c)
        hits = sum(n.hit_count for n in ns if n.id in retained)
        return hits

    lru_hits = build("lru")
    value_hits = build("value")
    # LRU evicts oldest10 = hottest10 -> keeps sum(10..1)=55; value keeps sum(20..11)=155
    _assert(lru_hits == 55, f"lru retained-hits {lru_hits} != 55")
    _assert(value_hits == 155, f"value retained-hits {value_hits} != 155")
    _assert(
        value_hits > lru_hits,
        f"value should retain more reuse mass ({value_hits} vs {lru_hits})",
    )
    print(
        f"  [ok] (3) value retains {value_hits} reuse-mass vs lru {lru_hits} "
        f"(anti-correlated recency/frequency)"
    )


def test_value_never_evicts_locked():
    c = _make_cache("value")
    nodes = [_add_leaf(c, hit_count=100 - i) for i in range(6)]
    # lock the coldest node (would be value victim #1) -> must be protected
    coldest = nodes[5]
    coldest.mamba_lock_ref = 1
    evicted = c.evict_mamba(3)
    _assert(evicted == 3, f"value evicted count {evicted} != 3")
    resident = _resident_ids(c)
    _assert(coldest.id in resident, "locked coldest node must NOT be evicted")
    # instead the next-3 coldest unlocked (nodes 4,3,2) are evicted
    _assert(nodes[4].id not in resident, "node4 should be evicted")
    _assert(nodes[3].id not in resident, "node3 should be evicted")
    _assert(nodes[2].id not in resident, "node2 should be evicted")
    print("  [ok] (4) value never evicts locked checkpoints")


def test_value_internal_node_tombstone_branch():
    c = _make_cache("value")
    # a low-value INTERNAL node (has a child) -> value path must tombstone, not leaf-evict
    internal = _add_leaf(c, hit_count=0)
    child = TreeNode()
    child.parent = internal
    internal.children[child.id] = child  # give it a child -> internal
    _add_leaf(c, hit_count=50)  # a hotter leaf that must survive
    evicted = c.evict_mamba(1)
    _assert(evicted == 1, f"internal-branch evicted {evicted} != 1")
    _assert(internal.mamba_value is None, "internal node should be tombstoned")
    _assert(internal.id not in _resident_ids(c), "tombstoned node leaves mamba lru list")
    print("  [ok] (5) value path tombstones low-value internal node")


# ---------------------------------------------------------------------------
# GATE-GDSF: cost-aware (Greedy-Dual-Size-Frequency) mamba eviction. The mamba
# checkpoint slot is fixed O(1) size, but a miss re-prefills the prefix so cost ~
# prefix length. gdsf priority H = clock + (hit_count+1)*recon_cost; evict min H.
# value/LFU is the uniform-length special case (recon_cost const => cancels).
# ---------------------------------------------------------------------------

def test_gdsf_orders_by_freq_times_cost():
    c = _make_cache("gdsf")
    # products (hit_count+1)*prefix_len, ascending id order for readability:
    #   A: (0+1)*1 = 1     (cheapest -> first victim)
    #   B: (1+1)*1 = 2
    #   C: (0+1)*4 = 4
    #   D: (3+1)*2 = 8     (most valuable -> retained)
    A = _add_leaf_cost(c, hit_count=0, prefix_len=1)
    B = _add_leaf_cost(c, hit_count=1, prefix_len=1)
    C = _add_leaf_cost(c, hit_count=0, prefix_len=4)
    D = _add_leaf_cost(c, hit_count=3, prefix_len=2)
    evicted = c.evict_mamba(2)
    _assert(evicted == 2, f"gdsf evicted {evicted} != 2")
    resident = _resident_ids(c)
    # lowest two products = A(1), B(2) evicted; C(4), D(8) retained
    _assert(A.id not in resident and B.id not in resident, "gdsf evicts lowest freq*cost")
    _assert(C.id in resident and D.id in resident, "gdsf keeps highest freq*cost")
    print("  [ok] (6) gdsf evicts lowest (hit_count+1)*prefix_len first")


def test_gdsf_beats_value_under_freq_cost_tradeoff():
    # freq and length ANTI-aligned: short prefixes are hotter, long prefixes warmer.
    # value(freq-only) evicts the LONG ones (lower freq); gdsf keeps them (freq*cost
    # higher). Retained RECONSTRUCTION-COST mass (sum hit_count*prefix_len) is the
    # thing that converts to re-prefill savings -> gdsf should retain far more.
    def build(policy):
        c = _make_cache(policy)
        short = [_add_leaf_cost(c, hit_count=10, prefix_len=1) for _ in range(4)]
        long = [_add_leaf_cost(c, hit_count=5, prefix_len=8) for _ in range(4)]
        c.evict_mamba(4)  # evict half
        retained = _resident_ids(c)
        alln = short + long
        recon_mass = sum(
            n.hit_count * len(n.hash_value) for n in alln if n.id in retained
        )
        return recon_mass, retained, short, long

    v_mass, v_ret, v_short, v_long = build("value")
    g_mass, g_ret, g_short, g_long = build("gdsf")
    # value keeps the 4 short (freq 10): mass = 4*(10*1) = 40, evicts all long
    _assert(v_mass == 40, f"value retained recon-mass {v_mass} != 40")
    _assert(all(n.id not in v_ret for n in v_long), "value evicts the long (lower-freq) prefixes")
    # gdsf keeps the 4 long (freq*cost 5*8=40 > short 10*1=10): mass = 4*(5*8) = 160
    _assert(g_mass == 160, f"gdsf retained recon-mass {g_mass} != 160")
    _assert(all(n.id in g_ret for n in g_long), "gdsf keeps the long expensive-to-rebuild prefixes")
    _assert(g_mass > v_mass, f"gdsf should retain more recon-cost mass ({g_mass} vs {v_mass})")
    print(
        f"  [ok] (7) gdsf retains {g_mass} recon-cost mass vs value {v_mass} "
        f"(freq/length anti-aligned)"
    )


def test_gdsf_reduces_to_value_under_uniform_length():
    # UNIFORM prefix length => recon_cost constant => gdsf order == value order.
    # Compare by hit_count (a stable key): node ids differ between the two fresh
    # builds since TreeNode.counter is global, so id-sets are not comparable.
    def evicted_hits(policy):
        c = _make_cache(policy)
        ns = [_add_leaf_cost(c, hit_count=20 - i, prefix_len=3) for i in range(12)]
        c.evict_mamba(6)
        retained = _resident_ids(c)
        return sorted(n.hit_count for n in ns if n.id not in retained)

    _assert(
        evicted_hits("gdsf") == evicted_hits("value"),
        "under uniform length gdsf must match value (value is the special case)",
    )
    print("  [ok] (8) gdsf == value under uniform length (value is the special case)")


def test_gdsf_never_evicts_locked():
    c = _make_cache("gdsf")
    # cheapest node (would be gdsf victim #1) but LOCKED -> must be protected
    locked = _add_leaf_cost(c, hit_count=0, prefix_len=1, lock=1)
    keep_hot = _add_leaf_cost(c, hit_count=9, prefix_len=9)  # highest product, retained
    mid = [_add_leaf_cost(c, hit_count=1, prefix_len=1) for _ in range(3)]  # product 2
    evicted = c.evict_mamba(3)
    _assert(evicted == 3, f"gdsf evicted {evicted} != 3")
    resident = _resident_ids(c)
    _assert(locked.id in resident, "locked cheapest node must NOT be evicted")
    _assert(keep_hot.id in resident, "highest freq*cost node retained")
    _assert(all(n.id not in resident for n in mid), "the 3 unlocked mid nodes evicted")
    print("  [ok] (9) gdsf never evicts locked checkpoints")


def test_gdsf_clock_monotonic():
    c = _make_cache("gdsf")
    _ = [_add_leaf_cost(c, hit_count=i, prefix_len=i + 1) for i in range(8)]
    before = c.gdsf_clock
    c.evict_mamba(4)
    _assert(c.gdsf_clock >= before, "gdsf clock must be non-decreasing")
    _assert(c.gdsf_clock > 0.0, "gdsf clock should advance past 0 after real evictions")
    print(f"  [ok] (10) gdsf aging clock monotonic (0 -> {c.gdsf_clock:g})")


def test_gdsf_internal_node_tombstone_branch():
    c = _make_cache("gdsf")
    internal = _add_leaf_cost(c, hit_count=0, prefix_len=1)  # lowest product, internal
    child = TreeNode()
    child.parent = internal
    internal.children[child.id] = child
    _add_leaf_cost(c, hit_count=50, prefix_len=9)  # valuable leaf must survive
    evicted = c.evict_mamba(1)
    _assert(evicted == 1, f"gdsf internal-branch evicted {evicted} != 1")
    _assert(internal.mamba_value is None, "gdsf must tombstone low-value internal node")
    _assert(internal.id not in _resident_ids(c), "tombstoned node leaves mamba lru list")
    print("  [ok] (11) gdsf path tombstones low-value internal node")


def main():
    print("GATE-G1: mamba value-aware eviction policy (offline, no GPU)")
    test_lru_evicts_oldest_regardless_of_frequency()
    test_value_evicts_lowest_frequency()
    test_value_beats_lru_under_anticorrelation()
    test_value_never_evicts_locked()
    test_value_internal_node_tombstone_branch()
    print("GATE-GDSF: cost-aware (freq x recon-cost) eviction (offline, no GPU)")
    test_gdsf_orders_by_freq_times_cost()
    test_gdsf_beats_value_under_freq_cost_tradeoff()
    test_gdsf_reduces_to_value_under_uniform_length()
    test_gdsf_never_evicts_locked()
    test_gdsf_clock_monotonic()
    test_gdsf_internal_node_tombstone_branch()
    print("GATE-G1 + GATE-GDSF PASS")


if __name__ == "__main__":
    main()
