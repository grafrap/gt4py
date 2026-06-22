# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for gt_hoist_k_invariant_reads (LICM for Sequential K-map).

Tests verify that K-invariant reads are moved from the Sequential K-map body
into the outer GPU scope, while K-dependent reads are left in place.
"""

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


# ── Helper: build an SDFG with GPU map → Sequential K-map ────────────────────

def _make_seqk_sdfg(
    *,
    k_invariant_arrays: list[str] | None = None,
    k_dependent_arrays: list[str] | None = None,
    k_dim_size: int = 50,
    horizontal_size: int = 10,
    dtype=dace.float64,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry]:
    """Build a minimal GPU map → Sequential K-map SDFG using add_mapped_tasklet.

    Uses DaCe's helper to get the map structure right. The GPU (outer) map covers
    i=[0:horizontal_size]. Inside it, a Sequential map iterates K=[0:k_dim_size].
    K-invariant arrays have IN connectors with subsets NOT referencing K.
    K-dependent arrays have subsets that DO reference K.

    Returns (sdfg, state, seq_k_entry).
    """
    if k_invariant_arrays is None:
        k_invariant_arrays = ["primal_normal"]
    if k_dependent_arrays is None:
        k_dependent_arrays = ["z_nabla2_e"]
    all_arrays = k_invariant_arrays + k_dependent_arrays

    sdfg = dace.SDFG("test_licm")
    state = sdfg.add_state(is_start_block=True)

    # Declare arrays
    sdfg.add_array("out", shape=(horizontal_size,), dtype=dtype, transient=False)
    for name in all_arrays:
        sdfg.add_array(name, shape=(horizontal_size, k_dim_size), dtype=dtype,
                       transient=False)
    sdfg.add_scalar("tmp_k", dtype=dtype, transient=True)

    # AccessNodes
    out_node = state.add_access("out")
    array_nodes = {n: state.add_access(n) for n in all_arrays}

    # Build GPU outer map manually using add_map
    gpu_me, gpu_mx = state.add_map(
        "gpu_map",
        ndrange={"__i": f"0:{horizontal_size}"},
        schedule=dace.ScheduleType.GPU_ThreadBlock,
    )

    # Connect global arrays into GPU map entry
    for name in all_arrays:
        gpu_me.add_in_connector(f"IN_{name}")
        gpu_me.add_out_connector(f"OUT_{name}")
        # External edge: global → GPU map entry
        state.add_edge(array_nodes[name], None, gpu_me, f"IN_{name}",
                       dace.Memlet(f"{name}[0:{horizontal_size}, 0:{k_dim_size}]"))

    # Sequential K-map inside GPU map
    seq_me, seq_mx = state.add_map(
        "seq_k_map",
        ndrange={"__k": f"0:{k_dim_size}"},
        schedule=dace.ScheduleType.Sequential,
    )

    # Route GPU OUT → Sequential IN with appropriate subsets
    for name in k_invariant_arrays:
        seq_me.add_in_connector(f"IN_{name}")
        seq_me.add_out_connector(f"OUT_{name}")
        # K-invariant: fixed slice at k=0, no __k
        state.add_edge(gpu_me, f"OUT_{name}", seq_me, f"IN_{name}",
                       dace.Memlet(f"{name}[__i, 0]"))
    for name in k_dependent_arrays:
        seq_me.add_in_connector(f"IN_{name}")
        seq_me.add_out_connector(f"OUT_{name}")
        # K-dependent: uses __k
        state.add_edge(gpu_me, f"OUT_{name}", seq_me, f"IN_{name}",
                       dace.Memlet(f"{name}[__i, __k]"))

    # Tasklet inside Sequential body: sum all inputs
    in_conns = {f"__in_{n}" for n in all_arrays}
    code = "__out = " + " + ".join(f"__in_{n}" for n in all_arrays)
    tlet = state.add_tasklet("compute", in_conns, {"__out"}, code)

    for name in k_invariant_arrays:
        state.add_edge(seq_me, f"OUT_{name}", tlet, f"__in_{name}",
                       dace.Memlet(f"{name}[__i, 0]"))
    for name in k_dependent_arrays:
        state.add_edge(seq_me, f"OUT_{name}", tlet, f"__in_{name}",
                       dace.Memlet(f"{name}[__i, __k]"))

    # Tasklet → tmp_k → Sequential exit → GPU exit → out
    tmp_node = state.add_access("tmp_k")
    state.add_edge(tlet, "__out", tmp_node, None, dace.Memlet("tmp_k"))
    seq_mx.add_in_connector("IN_out")
    seq_mx.add_out_connector("OUT_out")
    state.add_edge(tmp_node, None, seq_mx, "IN_out", dace.Memlet("out[__i]"))

    gpu_mx.add_in_connector("IN_out")
    gpu_mx.add_out_connector("OUT_out")
    state.add_edge(seq_mx, "OUT_out", gpu_mx, "IN_out", dace.Memlet("out[__i]"))
    state.add_edge(gpu_mx, "OUT_out", out_node, None,
                   dace.Memlet(f"out[0:{horizontal_size}]"))

    sdfg.validate()
    return sdfg, state, seq_me


# ── Tests ────────────────────────────────────────────────────────────────────

class TestHoistKInvariantReads:

    def test_k_invariant_read_hoisted(self):
        """K-invariant connector is removed from Sequential map; cache scalar is added."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=["primal_normal"],
            k_dependent_arrays=[],
        )
        n = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n == 1, f"Expected 1 hoist, got {n}"

        # The Sequential map should no longer have IN_primal_normal
        assert "IN_primal_normal" not in seq_me.in_connectors, \
            "K-invariant connector should have been removed from Sequential map"
        assert "OUT_primal_normal" not in seq_me.out_connectors, \
            "K-invariant OUT connector should have been removed from Sequential map"

        # A new Register-storage scalar transient must exist
        reg_scalars = [
            name for name, desc in sdfg.arrays.items()
            if desc.transient and desc.storage == dace.StorageType.Register
        ]
        assert len(reg_scalars) >= 1, "Expected at least one Register-storage scalar"

    def test_k_dependent_read_not_hoisted(self):
        """K-dependent connector (uses __k in subset) must stay in Sequential map."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=[],
            k_dependent_arrays=["z_nabla2_e"],
        )
        n = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n == 0, f"Expected 0 hoists, got {n}"

        # K-dependent connector stays
        assert "IN_z_nabla2_e" in seq_me.in_connectors

    def test_mixed_only_invariant_hoisted(self):
        """With mixed arrays, only K-invariant ones are hoisted."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=["primal_normal"],
            k_dependent_arrays=["z_nabla2_e"],
        )
        n = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n == 1, f"Expected 1 hoist, got {n}"

        # K-invariant gone, K-dependent stays
        assert "IN_primal_normal" not in seq_me.in_connectors
        assert "IN_z_nabla2_e" in seq_me.in_connectors

    def test_multiple_invariant_reads_hoisted(self):
        """All K-invariant arrays are hoisted independently."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=["primal_normal_v1", "primal_normal_v2", "ptr_coeff"],
            k_dependent_arrays=["z_nabla2_e"],
        )
        n = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n == 3, f"Expected 3 hoists, got {n}"

        for name in ["primal_normal_v1", "primal_normal_v2", "ptr_coeff"]:
            assert f"IN_{name}" not in seq_me.in_connectors
        assert "IN_z_nabla2_e" in seq_me.in_connectors

    def test_sdfg_validates_after_hoist(self):
        """The SDFG must validate after applying the transformation."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=["primal_normal"],
            k_dependent_arrays=["z_nabla2_e"],
        )
        gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        sdfg.validate()  # Must not raise

    def test_idempotent(self):
        """Applying the transformation twice gives 0 on the second pass."""
        sdfg, state, seq_me = _make_seqk_sdfg(
            k_invariant_arrays=["primal_normal"],
            k_dependent_arrays=[],
        )
        n1 = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        n2 = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n1 == 1
        assert n2 == 0, "Second application should find nothing to hoist"

    def test_no_sequential_map_no_hoist(self):
        """SDFG without any Sequential map: 0 hoists."""
        sdfg = dace.SDFG("no_seq")
        state = sdfg.add_state(is_start_block=True)
        sdfg.add_array("x", shape=(10,), dtype=dace.float64, transient=False)
        n = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
        assert n == 0
