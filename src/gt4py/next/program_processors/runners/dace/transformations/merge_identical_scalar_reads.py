# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Merge multiple identical reads from the same MapEntry output connector and subset.

In CSI-fused DaCe GPU kernels, the same global field position is read N times through
the same MapEntry OUT connector (once per use in the fused lambda body). This function
introduces one shared scalar transient per unique (data, connector, subset) group,
eliminating the redundant global-memory reads.

Actual SDFG structure observed (e.g. for u_vert via connector OUT__cs_23):

  MapEntry --[OUT__cs_23, subset=center]--> Tasklet_mul_1 --> gtir_tmp_89
  MapEntry --[OUT__cs_23, same subset]---> Tasklet_mul_2 --> gtir_tmp_XX
  ...×6 total

After transformation:

  MapEntry --[OUT__cs_23, subset=center]--> AccessNode(__misr_0)
  AccessNode(__misr_0) --[]--> Tasklet_mul_1 --> gtir_tmp_89
  AccessNode(__misr_0) --[]--> Tasklet_mul_2 --> gtir_tmp_XX
  ...

One global read, six local reads from the shared transient scalar.
"""

import collections

import dace
from dace.sdfg import nodes as dace_nodes


def gt_merge_identical_scalar_reads(sdfg: dace.SDFG) -> int:
    """Merge duplicate outgoing edges from GPU MapEntry nodes that share the same
    (data name, out-connector, memlet subset) into a single shared scalar transient.

    Returns the total number of redundant reads eliminated.
    """
    n_eliminated = 0
    for state in sdfg.states():
        for node in list(state.nodes()):
            if not isinstance(node, dace_nodes.MapEntry):
                continue
            if node.schedule not in (
                dace.ScheduleType.GPU_Device,
                dace.ScheduleType.GPU_ThreadBlock,
            ):
                continue
            n_eliminated += _merge_mapentry_duplicates(sdfg, state, node)
    return n_eliminated


def _merge_mapentry_duplicates(
    sdfg: dace.SDFG, state: dace.SDFGState, map_entry: dace_nodes.MapEntry
) -> int:
    """Group the MapEntry's outgoing edges by (data, connector, subset); merge groups > 1."""
    groups: dict[tuple, list] = collections.defaultdict(list)
    for edge in state.out_edges(map_entry):
        data_name = edge.data.data
        # Skip edges that don't reference a known (non-transient) array
        if data_name not in sdfg.arrays:
            continue
        subset_key = str(edge.data.subset) if edge.data.subset is not None else ""
        key = (data_name, edge.src_conn, subset_key)
        groups[key].append(edge)

    n_eliminated = 0
    for (_data_name, _conn, _sub), edges in groups.items():
        if len(edges) > 1:
            n_eliminated += _introduce_shared_scalar(sdfg, state, map_entry, edges)
    return n_eliminated


def _introduce_shared_scalar(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    edges: list,
) -> int:
    """Replace N identical MapEntry→dst edges with one shared scalar AccessNode.

    Keeps ONE MapEntry→scalar edge and adds N scalar→dst edges.
    Returns N-1 (number of redundant reads eliminated).
    """
    ref = edges[0]
    dtype = sdfg.arrays[ref.data.data].dtype

    # Create the shared scalar transient
    new_name = sdfg.temp_data_name()
    sdfg.add_scalar(new_name, dtype=dtype, transient=True)
    shared_node = state.add_access(new_name)

    # One MapEntry --[original data+subset]--> shared_scalar AccessNode.
    # The destination subset on the scalar is the single element [0].
    state.add_edge(
        map_entry, ref.src_conn, shared_node, None,
        dace.Memlet(
            data=ref.data.data,
            subset=ref.data.subset,
            other_subset=dace.subsets.Range([(0, 0, 1)]),
        ),
    )

    # All N downstream consumers now read from the shared scalar at element [0].
    for edge in edges:
        dst_node = edge.dst
        dst_conn = edge.dst_conn
        state.remove_edge(edge)
        state.add_edge(
            shared_node, None, dst_node, dst_conn,
            dace.Memlet(data=new_name, subset=dace.subsets.Range([(0, 0, 1)])),
        )

    return len(edges) - 1
