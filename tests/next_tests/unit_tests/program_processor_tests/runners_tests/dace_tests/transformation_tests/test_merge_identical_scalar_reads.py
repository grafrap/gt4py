# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for gt_merge_identical_scalar_reads.

Tests verify that duplicate outgoing edges from a GPU MapEntry with the same
(data name, connector, subset) are merged into one shared scalar transient.
"""

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


# ── Helper: build an SDFG with a GPU ThreadBlock map ─────────────────────────

def _make_gpu_map_sdfg(
    global_array_name: str = "u_vert",
    n_consumers: int = 2,
    n_identical: int = 2,
    dtype=dace.float64,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry]:
    """Build a minimal SDFG with a GPU ThreadBlock map.

    The map has one global input array (non-transient). It creates `n_consumers`
    tasklets: the first `n_identical` all read the global array at the SAME subset
    (via the same MapEntry OUT connector and same memlet subset); the remaining
    `n_consumers - n_identical` each use a DIFFERENT subset.

    Each tasklet writes to a separate scalar transient (no MapExit / output needed;
    we only care about the INPUT side for testing the transformation).

    Returns (sdfg, state, map_entry).
    """
    sdfg = dace.SDFG(f"test_misr_{global_array_name}")
    state = sdfg.add_state(is_start_block=True)

    # Global input array (shape = [10])
    sdfg.add_array(global_array_name, shape=(10,), dtype=dtype, transient=False)
    # Sink output (1-element array written by map, size 1 to simplify subset)
    sdfg.add_scalar("out", dtype=dtype, transient=False)

    src = state.add_access(global_array_name)
    dst = state.add_access("out")

    # GPU ThreadBlock map over index i
    me, mx = state.add_map(
        "gpu_map",
        ndrange={"__i": "0:10"},
        schedule=dace.ScheduleType.GPU_ThreadBlock,
    )
    me.add_scope_connectors(global_array_name)
    mx.add_scope_connectors("out")

    # Connect global src → MapEntry
    state.add_edge(src, None, me, f"IN_{global_array_name}",
                   dace.Memlet(f"{global_array_name}[0:10]"))

    # Create n_consumers tasklets inside the map
    for i in range(n_consumers):
        # Shared subset for the first n_identical; unique for the rest
        if i < n_identical:
            read_subset = "__i"
        else:
            # Modulo so we stay in [0, 9]
            read_subset = f"(__i + {i}) % 10"
        t = state.add_tasklet(f"tlet_{i}", {"__in"}, {"__out"}, "__out = __in + 0")
        state.add_edge(me, f"OUT_{global_array_name}", t, "__in",
                       dace.Memlet(f"{global_array_name}[{read_subset}]"))
        # Each tasklet writes to a separate internal transient (accumulate via "+")
        sdfg.add_scalar(f"tmp_{i}", dtype=dtype, transient=True)
        tmp = state.add_access(f"tmp_{i}")
        state.add_edge(t, "__out", tmp, None, dace.Memlet(f"tmp_{i}[0]"))

    # Final reduction tasklet: sum all tmp_* and write to out (keeps SDFG valid)
    params = {f"__in{i}": None for i in range(n_consumers)}
    body = " + ".join(f"__in{i}" for i in range(n_consumers))
    reduce_tlet = state.add_tasklet("reduce", set(params.keys()), {"__out"}, f"__out = {body}")
    for i in range(n_consumers):
        tmp_node = [n for n in state.nodes()
                    if isinstance(n, dace_nodes.AccessNode) and n.data == f"tmp_{i}"][0]
        state.add_edge(tmp_node, None, reduce_tlet, f"__in{i}", dace.Memlet(f"tmp_{i}[0]"))
    state.add_edge(reduce_tlet, "__out", mx, "IN_out", dace.Memlet("out[0]"))
    state.add_edge(mx, "OUT_out", dst, None, dace.Memlet("out[0]"))

    sdfg.validate()
    return sdfg, state, me


# ── Tests ────────────────────────────────────────────────────────────────────

class TestMergeIdenticalScalarReads:

    def test_two_identical_reads_merged(self):
        """Two edges with the same subset → one shared scalar, 1 read eliminated."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=2, n_identical=2)
        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 1, f"Expected 1 read eliminated, got {n}"

        # The MapEntry should now have exactly 1 edge to the shared scalar (for the
        # merged group) plus edges for the output
        map_out_edges = list(state.out_edges(me))
        u_edges = [e for e in map_out_edges if e.data.data == "u_vert"]
        assert len(u_edges) == 1, f"Expected 1 u_vert edge from MapEntry, got {len(u_edges)}"

        # That one edge leads to an AccessNode (the shared transient scalar)
        shared_dst = u_edges[0].dst
        assert isinstance(shared_dst, dace_nodes.AccessNode)
        assert sdfg.arrays[shared_dst.data].transient

        # Both original tasklets are now consumers of the shared scalar
        consumers = list(state.out_edges(shared_dst))
        assert len(consumers) == 2, f"Expected 2 consumers of shared scalar, got {len(consumers)}"
        for c in consumers:
            assert isinstance(c.dst, dace_nodes.Tasklet)

    def test_n7_identical_reads_merged(self):
        """7 edges with same subset → 1 shared scalar, 6 reads eliminated."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=7, n_identical=7)
        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 6

        u_edges = [e for e in state.out_edges(me) if e.data.data == "u_vert"]
        assert len(u_edges) == 1

        shared_dst = u_edges[0].dst
        assert isinstance(shared_dst, dace_nodes.AccessNode)
        consumers = list(state.out_edges(shared_dst))
        assert len(consumers) == 7

    def test_different_subsets_not_merged(self):
        """Edges with different subsets are not merged."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=3, n_identical=1)
        # n_identical=1 means each consumer has a UNIQUE subset → no duplicates
        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 0, f"Expected 0 reads eliminated, got {n}"

        # MapEntry still has 3 outgoing u_vert edges
        u_edges = [e for e in state.out_edges(me) if e.data.data == "u_vert"]
        assert len(u_edges) == 3

    def test_mixed_identical_and_unique(self):
        """First 3 consumers share a subset; last 2 have unique subsets.
        Only the 3-group is merged → 2 reads eliminated."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=5, n_identical=3)
        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 2

        u_edges = [e for e in state.out_edges(me) if e.data.data == "u_vert"]
        # 1 to the shared scalar + 2 unique direct edges = 3 total
        assert len(u_edges) == 3
        # Exactly one of them goes to an AccessNode (the shared scalar)
        access_dsts = [e for e in u_edges if isinstance(e.dst, dace_nodes.AccessNode)]
        assert len(access_dsts) == 1
        shared = access_dsts[0].dst
        assert len(list(state.out_edges(shared))) == 3

    def test_no_gpu_map_not_merged(self):
        """Duplicate edges outside a GPU map scope are NOT merged."""
        sdfg = dace.SDFG("test_no_gpu")
        state = sdfg.add_state(is_start_block=True)
        sdfg.add_array("u_vert", shape=(10,), dtype=dace.float64, transient=False)
        sdfg.add_array("out", shape=(2,), dtype=dace.float64, transient=False)

        src = state.add_access("u_vert")
        dst = state.add_access("out")

        # CPU (Sequential) map — NOT a GPU map
        me, mx = state.add_map("cpu_map", ndrange={"__i": "0:10"},
                                schedule=dace.ScheduleType.Sequential)
        me.add_scope_connectors("u_vert")
        mx.add_scope_connectors("out")

        state.add_edge(src, None, me, "IN_u_vert", dace.Memlet("u_vert[0:10]"))

        for i in range(2):
            t = state.add_tasklet(f"tlet_{i}", {"__in"}, {"__out"}, "__out = __in")
            state.add_edge(me, "OUT_u_vert", t, "__in", dace.Memlet("u_vert[__i]"))
            sdfg.add_scalar(f"tmp_{i}", dtype=dace.float64, transient=True)
            tmp = state.add_access(f"tmp_{i}")
            state.add_edge(t, "__out", tmp, None, dace.Memlet(f"tmp_{i}[0]"))
            state.add_edge(tmp, None, mx, "IN_out", dace.Memlet(f"out[{i}]"))

        mx.add_scope_connectors("out")
        state.add_edge(mx, "OUT_out", dst, None, dace.Memlet("out[0:2]"))
        sdfg.validate()

        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 0, "Should not merge in non-GPU maps"

    def test_multiple_arrays_merged_independently(self):
        """Two global arrays (u_vert, v_vert) each with duplicate reads — both merged."""
        sdfg = dace.SDFG("test_multi_array")
        state = sdfg.add_state(is_start_block=True)
        sdfg.add_array("u_vert", shape=(10,), dtype=dace.float64, transient=False)
        sdfg.add_array("v_vert", shape=(10,), dtype=dace.float64, transient=False)
        sdfg.add_array("out", shape=(4,), dtype=dace.float64, transient=False)

        u_src = state.add_access("u_vert")
        v_src = state.add_access("v_vert")
        dst = state.add_access("out")

        me, mx = state.add_map("gpu_map", ndrange={"__i": "0:10"},
                                schedule=dace.ScheduleType.GPU_ThreadBlock)
        for arr in ("u_vert", "v_vert"):
            me.add_scope_connectors(arr)
        mx.add_scope_connectors("out")

        state.add_edge(u_src, None, me, "IN_u_vert", dace.Memlet("u_vert[0:10]"))
        state.add_edge(v_src, None, me, "IN_v_vert", dace.Memlet("v_vert[0:10]"))

        for i, arr in enumerate(["u_vert", "u_vert", "v_vert", "v_vert"]):
            t = state.add_tasklet(f"tlet_{i}", {"__in"}, {"__out"}, "__out = __in")
            state.add_edge(me, f"OUT_{arr}", t, "__in", dace.Memlet(f"{arr}[__i]"))
            sdfg.add_scalar(f"tmp_{i}", dtype=dace.float64, transient=True)
            tmp = state.add_access(f"tmp_{i}")
            state.add_edge(t, "__out", tmp, None, dace.Memlet(f"tmp_{i}[0]"))
            state.add_edge(tmp, None, mx, "IN_out", dace.Memlet(f"out[{i}]"))

        mx.add_scope_connectors("out")
        state.add_edge(mx, "OUT_out", dst, None, dace.Memlet("out[0:4]"))
        sdfg.validate()

        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 2, f"Expected 2 (1 from u_vert + 1 from v_vert), got {n}"

    def test_sdfg_validates_after_merge(self):
        """The SDFG must validate after applying the transformation."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=6, n_identical=6)
        gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        sdfg.validate()

    def test_zero_eliminations_on_no_duplicates(self):
        """Already-unique reads return 0."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=4, n_identical=1)
        n = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n == 0

    def test_idempotent(self):
        """Applying the transformation twice gives the same result as once."""
        sdfg, state, me = _make_gpu_map_sdfg(n_consumers=5, n_identical=5)
        n1 = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        n2 = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
        assert n1 == 4
        assert n2 == 0, "Second application should find nothing to merge"
