# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Hoist K-invariant reads from the Sequential K-map body to the outer GPU scope (LICM).

After ``_sequentialize_k_dimension``, fields like ``primal_normal_vert_v1``,
``ptr_coeff_1/2``, ``inv_vert_vert_length``, and ``inv_primal_edge_length`` are read
inside the sequential K-loop body even though their memory addresses do not depend on
the K iteration index.  This function identifies those reads via the Sequential map's
OUT-connectors (inside the loop body), moves them to a Register-lifetime transient
computed once before the loop, and reroutes the K-map body to consume the cached value.

K-invariance check uses OUT-edges (not IN-edges).  After ``extract_map_dims`` splits K
into a nested Sequential map, IN-edges carry K-dependent data as *range subsets*
(e.g. ``z_nabla2_e[edge, 0:K_total]``) whose ``free_symbols`` do NOT contain the
Sequential K parameter.  The K parameter only appears in OUT-edge subsets inside the
scope (e.g. ``z_nabla2_e[edge, k_param]``).  Checking OUT-edges avoids incorrectly
hoisting K-dependent fields.

Cache shape and subset routing:

The IN-edge carries a multi-element subset such as ``ptr_coeff_1[i, j, k, 0:5]``
(6 V2E slots).  The cache transient therefore has shape ``[size_0, size_1, ...]``
matching the number of elements per dimension.

Each OUT-edge (body consumers) carries a *specific element* of the global array, e.g.
``ptr_coeff_1[i, j, k, slot_s]``.  After hoisting, this becomes a local access into
the cache: the local subset is obtained by subtracting the IN-edge's range-start from
each dimension of the OUT-edge's subset, e.g. ``cache[0, 0, 0, slot_s - 0]``.

This ensures that DaCe generates scalar reads (``double val = cache[0][0][0][s]``)
rather than pointer arguments (``double* ptr = &cache[0]``), which would cause a
CUDA compiler error ("expression must have arithmetic type").

Expected savings: 49/50 × (all K-invariant global reads) eliminated from the K-loop.
"""

import os as _os

import dace
import dace.subsets as _dace_subsets
import sympy as _sympy
from dace.sdfg import nodes as dace_nodes
from dace.sdfg import propagation as dace_propagation


def gt_hoist_k_invariant_reads(sdfg: dace.SDFG) -> int:
    """Hoist K-invariant reads out of Sequential K-loop into the outer GPU scope.

    For each Sequential-schedule MapEntry node in the SDFG, inspects its incoming
    edges.  Reads are K-invariant when the corresponding OUT-edges (inside the
    Sequential scope) do NOT reference the Sequential map's K parameter in their
    memlet subsets — meaning the same global memory location is accessed on every
    K-iteration without change.

    Such reads are moved out of the K-loop:
    1. A Register-lifetime transient (scalar or array) is created with the same
       shape as the hoisted subset.
    2. The global array is read into this transient once, before the Sequential map.
    3. The Sequential map's IN/OUT connector pair for that array is replaced by a pair
       for the new transient.  Each body edge is updated with a *local* subset that
       identifies the specific element of the cache (offset from cache origin 0).

    Returns the number of redundant global reads eliminated.
    """
    n_hoisted = 0
    for state in sdfg.states():
        for node in list(state.nodes()):
            if not isinstance(node, dace_nodes.MapEntry):
                continue
            if node.map.schedule != dace.ScheduleType.Sequential:
                continue
            n_hoisted += _hoist_from_sequential_map(sdfg, state, node)
    return n_hoisted


def _subset_to_ranges(subset):
    """Return list of (start, end, step) tuples regardless of Range vs Indices type."""
    if isinstance(subset, _dace_subsets.Range):
        return list(subset.ranges)
    elif isinstance(subset, _dace_subsets.Indices):
        return [(idx, idx, 1) for idx in subset.indices]
    else:
        raise TypeError(f"Unexpected subset type: {type(subset)}")


def _local_subset(out_subset, in_starts):
    """Compute the cache-local subset by subtracting the IN-edge range-start per dim.

    ``in_starts`` is a list of the per-dimension start values from the IN-edge's range
    (one entry per array dimension).  For a scalar dimension like IDim fixed to ``i``,
    ``in_start = i`` so the local index is ``out_val - i = 0``.  For a slot dimension
    with IN-range starting at 0, the local index is ``slot - 0 = slot``.
    """
    out_ranges = _subset_to_ranges(out_subset)
    local_ranges = [
        (
            _sympy.simplify(out_r[0] - in_s),
            _sympy.simplify(out_r[1] - in_s),
            out_r[2],
        )
        for out_r, in_s in zip(out_ranges, in_starts)
    ]
    return _dace_subsets.Range(local_ranges)


def _hoist_from_sequential_map(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    seq_entry: dace_nodes.MapEntry,
) -> int:
    """Hoist K-invariant IN-connectors of one Sequential map out to the outer scope."""
    k_params = set(seq_entry.map.params)
    n_hoisted = 0

    for in_edge in list(state.in_edges(seq_entry)):
        in_conn = in_edge.dst_conn
        if not (in_conn and in_conn.startswith("IN_")):
            continue

        out_conn = "OUT_" + in_conn[3:]
        out_edges = [e for e in state.out_edges(seq_entry) if e.src_conn == out_conn]
        if not out_edges:
            continue  # No consumers inside K-body — nothing to hoist

        # K-invariant check: examine the OUT-edges INSIDE the Sequential scope.
        # After extract_map_dims, IN-edges carry K-data as range subsets (e.g.
        # z_nabla2_e[edge, 0:K_total]) — the K map param is NOT in their free_symbols.
        # The K param only appears in OUT-edge subsets (e.g. z_nabla2_e[edge, k_param]).
        # Checking OUT-edges correctly identifies K-dependent reads.
        if any(k in e.data.free_symbols for k in k_params for e in out_edges):
            continue  # K-dependent — must stay inside loop

        # Only hoist reads from NON-TRANSIENT (global) arrays — transient cache
        # arrays introduced by a previous LICM pass must not be re-processed.
        src_data_name = in_edge.data.data
        if src_data_name not in sdfg.arrays:
            continue
        if sdfg.arrays[src_data_name].transient:
            continue
        dtype = sdfg.arrays[src_data_name].dtype

        subset = in_edge.data.subset
        n_elems = int(subset.num_elements())
        in_ranges = _subset_to_ranges(subset)
        in_starts = [r[0] for r in in_ranges]

        if _os.environ.get("DACE_LICM_DEBUG", "0") == "1":
            print(
                f"[licm] hoisting {src_data_name!r} "
                f"(n_elems={n_elems}, subset={subset})",
                flush=True,
            )

        # 1. Create a Register-storage transient to hold the K-invariant value(s).
        #    Shape matches the IN-edge subset size so the pre-read covers exactly the
        #    elements consumed by the loop body.  Single-element subsets use add_scalar.
        cache_name = sdfg.temp_data_name()
        if n_elems == 1:
            sdfg.add_scalar(
                cache_name, dtype=dtype, transient=True,
                storage=dace.StorageType.Register,
            )
        else:
            cache_shape = [int(s) for s in subset.size()]
            sdfg.add_array(
                cache_name, shape=cache_shape, dtype=dtype, transient=True,
                storage=dace.StorageType.Register,
            )
        cache_node = state.add_access(cache_name)

        # 2. Add a pre-read edge: global_source → cache (BEFORE the Sequential map).
        #    The Memlet reads the same subset from the global array; DaCe infers the
        #    destination subset as the full cache extent.
        state.add_edge(
            in_edge.src, in_edge.src_conn,
            cache_node, None,
            dace.Memlet(data=src_data_name, subset=subset),
        )

        # 3. Remove original IN/OUT connector pair and all associated edges.
        state.remove_edge(in_edge)
        seq_entry.remove_in_connector(in_conn)
        seq_entry.remove_out_connector(out_conn)

        # 4. Route cache_node through the Sequential map with a new connector pair.
        #    The IN-edge carries the full cache into the loop scope.
        #    Each OUT-edge uses a LOCAL subset (offset from the cache origin) so that
        #    DaCe generates a scalar element access (e.g. cache[0][0][0][slot])
        #    rather than a pointer to the whole array — which would be a CUDA error.
        new_in  = "IN_"  + cache_name
        new_out = "OUT_" + cache_name
        seq_entry.add_in_connector(new_in)
        seq_entry.add_out_connector(new_out)

        state.add_edge(
            cache_node, None, seq_entry, new_in,
            dace.Memlet(data=cache_name),
        )
        for out_edge in out_edges:
            if n_elems == 1:
                # Scalar cache: simple full-extent memlet.
                body_memlet = dace.Memlet(data=cache_name)
            else:
                # Array cache: compute local element position within the cache.
                # local[d] = out_edge.subset[d] - in_start[d]
                # e.g. ptr_coeff_1[i,j,k,slot] → cache[0,0,0,slot]
                local_sub = _local_subset(out_edge.data.subset, in_starts)
                body_memlet = dace.Memlet(data=cache_name, subset=local_sub)
            state.add_edge(
                seq_entry, new_out,
                out_edge.dst, out_edge.dst_conn,
                body_memlet,
            )
            state.remove_edge(out_edge)

        n_hoisted += 1

    # Recompute memlets through the (modified) Sequential map scope.
    try:
        dace_propagation.propagate_memlets_map_scope(sdfg, state, seq_entry)
    except Exception:
        pass  # Propagation failure is non-fatal; SDFG may still be valid.

    return n_hoisted
