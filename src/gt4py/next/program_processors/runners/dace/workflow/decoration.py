# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config, utils as gtx_utils
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace import sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    compilation as gtx_wfdcompilation,
)


def _read_copy_kernel_time_s(collect_copy_time_arg: np.ndarray) -> float:
    """Return the accumulated copy-kernel wall time (seconds) written by the SDFG.

    ``_instrument_copy_kernels_for_compute_only`` wraps each ``copy_*`` state with
    pre/post chrono timing states that accumulate the elapsed GPU wall time into the
    ``SDFG_ARG_METRIC_COPY_TIME`` output array (``collect_copy_time_arg``). This is
    readable directly after ``fast_call()`` — no report flush or file I/O required.
    """
    return float(collect_copy_time_arg[0])


def convert_args(
    fun: gtx_wfdcompilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> stages.ExecutableProgram:
    # Retrieve metrics level from GT4Py environment variable.
    collect_time = metrics.is_level_enabled(metrics.PERFORMANCE)
    collect_time_arg = np.array([1], dtype=np.float64)
    # Detect whether this SDFG has copy-kernel timing states (SDFG_ARG_METRIC_COPY_TIME array
    # added by _instrument_copy_kernels_for_compute_only). Only present for GPU SDFGs with
    # copy_* kernels when DACE_TIME_COMPUTE_ONLY != 0.
    has_copy_time = gtx_wfdcommon.SDFG_ARG_METRIC_COPY_TIME in fun.sdfg_program.sdfg.arrays
    collect_copy_time_arg = np.array([0.0], dtype=np.float64) if has_copy_time else None
    # One-shot flag: announce the first nonzero copy-time subtraction per program, so a working
    # compute-only timer is distinguishable from a silent no-op (e.g. stale binary without the
    # copy-timing states). Greppable, same style as COPY_KERNEL_DETECTED / DIM_ORDER.
    copy_time_announced = [False]
    # We use the callback function provided by the compiled program to update the SDFG arglist.
    update_sdfg_call_args = functools.partial(
        fun.update_sdfg_ctype_arglist, device, fun.sdfg_argtypes
    )

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        # Reset copy-time accumulator before each call so C++ starts from 0 for this invocation.
        if has_copy_time and collect_copy_time_arg is not None:
            collect_copy_time_arg[0] = 0.0

        try:
            # Not the first call.
            #  We will only update the argument vector  for the normal call.
            # NOTE: If this is the first time then we will generate an exception because
            #   `fun.csdfg_args` is `None`
            # TODO(phimuell, edopao): Think about refactor the code such that the update
            #   of the argument vector is a Method of the `CompiledDaceProgram`.
            update_sdfg_call_args(args, fun.csdfg_argv, offset_provider)  # type: ignore[arg-type]  # Will error out in first call.

        except TypeError:
            # First call. Construct the initial argument vector of the `CompiledDaceProgram`.
            assert fun.csdfg_argv is None and fun.csdfg_init_argv is None
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(args)
            this_call_args = sdfg_callable.get_sdfg_args(
                fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            this_call_args |= {
                gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL: config.COLLECT_METRICS_LEVEL,
                gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME: collect_time_arg,
            }
            if has_copy_time and collect_copy_time_arg is not None:
                this_call_args[gtx_wfdcommon.SDFG_ARG_METRIC_COPY_TIME] = collect_copy_time_arg
            fun.construct_arguments(**this_call_args)

        # Perform the call to the SDFG.
        fun.fast_call()

        if collect_time:
            sample = collect_time_arg[0].item()
            # Compute-only override (opt-in via DACE_TIME_COMPUTE_ONLY=1): subtract the wall-time
            # of the copy_* kernels so the reported metric reflects only the compute kernels.
            # Valid because gt4py forces max_concurrent_streams=-1 (sequential ⇒ total = compute +
            # copy). No-op (copy_t == 0) when the stencil has no copy kernels.
            if has_copy_time and collect_copy_time_arg is not None:
                copy_t = _read_copy_kernel_time_s(collect_copy_time_arg)
                if copy_t > 0.0:
                    sample = max(sample - copy_t, 0.0)
                    if not copy_time_announced[0]:
                        copy_time_announced[0] = True
                        print(
                            f"COMPUTE_ONLY_TIMER: program={fun.sdfg_program.sdfg.name} "
                            f"copy_ms={copy_t * 1e3:.4f} (subtracted from reported metric)",
                            flush=True,
                        )
            metrics.add_sample_to_current_source(metrics.COMPUTE_METRIC, sample)

    return decorated_program
