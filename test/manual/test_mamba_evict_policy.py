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


def main():
    print("GATE-G1: mamba value-aware eviction policy (offline, no GPU)")
    test_lru_evicts_oldest_regardless_of_frequency()
    test_value_evicts_lowest_frequency()
    test_value_beats_lru_under_anticorrelation()
    test_value_never_evicts_locked()
    test_value_internal_node_tombstone_branch()
    print("GATE-G1 PASS")


if __name__ == "__main__":
    main()
