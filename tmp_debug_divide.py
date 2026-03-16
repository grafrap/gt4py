"""Debug script to understand why compute_divide_volume fails type inference after CartUnroll."""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
os.environ['GT4PY_TRANSLATOR_MESH'] = os.path.expanduser('grid-generator/parallelogram_grid.nc')

from gt4py.next.iterator.transforms import cart_unroll
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.pass_manager import _print_ir_block

orig_apply = cart_unroll.CartUnroll.apply

@classmethod
def patched_apply(cls, ir_node, **kwargs):
    result = orig_apply.__func__(cls, ir_node, **kwargs)
    
    # Walk the result tree and find all SymRef nodes, track by Python id
    seen = {}  # id -> (node, path)
    duplicates = []

    def walk(node, path=''):
        if isinstance(node, ir.SymRef):
            nid = id(node)
            if nid in seen:
                duplicates.append((node, path, seen[nid][1]))
            else:
                seen[nid] = (node, path)
        
        if isinstance(node, (ir.FunCall,)):
            walk(node.fun, path + '.fun')
            for i, a in enumerate(node.args):
                walk(a, path + f'.arg{i}')
        elif isinstance(node, ir.Lambda):
            for i, p in enumerate(node.params):
                walk(p, path + f'.param{i}')
            walk(node.expr, path + '.body')
        elif isinstance(node, ir.Program):
            for i, s in enumerate(node.body):
                walk(s, f'prog.body[{i}]')
        elif isinstance(node, ir.SetAt):
            walk(node.expr, path + '.set_at_expr')
            walk(node.target, path + '.target')
            walk(node.domain, path + '.domain')

    walk(result)

    if duplicates:
        print("=== SHARED SYMREF NODES AFTER CART_UNROLL ===")
        for node, p1, p2 in duplicates:
            print(f"  id={hex(id(node))} '{node.id}' type={node.type}")
            print(f"    first at: {p2}")
            print(f"    also at:  {p1}")
    else:
        print("=== NO SHARED SYMREF NODES AFTER CART_UNROLL ===")
    
    return result

cart_unroll.CartUnroll.apply = patched_apply

# Now intercept type inference to see what context __arg0 is being typed in
from gt4py.next.iterator.type_system import inference
from gt4py.next.iterator.type_system import type_specifications as it_ts

orig_visit = inference.ITIRTypeInference.visit

call_count = [0]

def patched_visit(self, node, **kwargs):
    call_count[0] += 1
    if isinstance(node, ir.SymRef) and str(node.id) == '__arg0':
        ctx = kwargs.get('ctx', {})
        from gt4py.next.eve.concepts import SymbolName
        key = SymbolName('__arg0')
        ctx_type = ctx.get(key)
        print(f"  [TI] visit __arg0: node.type={node.type}, ctx_type={ctx_type}, node_id={hex(id(node))}")
    return orig_visit(self, node, **kwargs)

inference.ITIRTypeInference.visit = patched_visit

# Actually run the test
import pytest
import numpy as np

# We need to construct the exec_alloc_descriptor
from gt4py.next.program_processors.runners.gtfn import run_gtfn

# Use the module-level fixture machinery
import gt4py.next as gtx
from gt4py.next import allocators as gtx_allocators

class FakeAlloc:
    executor = run_gtfn
    allocator = gtx_allocators.StandardCPUFieldBufferAllocator()

from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_ffront_fvm_nabla_decomposition import (
    _prepare_parallelogram_structured_case,
    setup_program,
    IDim, JDim, Kolor,
)
from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_ffront_fvm_nabla import (
    compute_zavgS,
    compute_neighbor_sum_weighted,
    compute_divide_volume,
)

try:
    case = _prepare_parallelogram_structured_case(FakeAlloc())
    
    pnabla_m_struct = gtx.zeros(
        {IDim: int(case["remap_sizes"].max_i),
         JDim: int(case["remap_sizes"].max_j),
         Kolor: 1},
        allocator=FakeAlloc().allocator,
    )
    out_struct = gtx.zeros(
        {IDim: int(case["remap_sizes"].max_i),
         JDim: int(case["remap_sizes"].max_j),
         Kolor: 1},
        allocator=FakeAlloc().allocator,
    )
    
    divide_program = setup_program(
        compute_divide_volume,
        backend=case["backend"],
    )
    print("divide_program compiled OK, calling...")
    divide_program(pnabla_M=pnabla_m_struct, vol=case["vol_struct"], out=out_struct)
    print("divide_program ran OK")

except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
