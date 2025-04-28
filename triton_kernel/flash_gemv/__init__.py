import importlib
import os.path as osp
import torch
from .flash_gemv import (
    fuse_gemv_cmp, 
    fuse_gemv_flag, 
    fuse_gemv_flag_gemv, 
    fuse_gemv_flag_gemv_gemv, 
    fuse_gemv_gemv_gemv, 
    fuse_gemv_flag_batch, 
    fuse_gemv_flag_local, 
    fuse_flag_gemv_local, 
    atomic_gemv, 
    flag_gemv_gemv_atomic, 
    flag_gemv_gemv, 
    flag_gemv_gemv_inner, 
    flag_gemv_gemv_inner_fp32, 
    flag_gemv_gemv_inner_bf16, 
)
from .kernels import (
    gather_gemv_elemul_flag_3d,
    gather_transposed_gemv_flag_3d,
    gather_transposed_gemv_flag_3d_activation,
    mistral_mlp_partial_sparse,
    mistral_mlp_sparse_direct_index_2d
)

__version__ = '0.0.1'

library = '_C'
spec = importlib.machinery.PathFinder().find_spec(
    library, [osp.dirname(__file__)])
if spec is not None:
    torch.ops.load_library(spec.origin)
else:
    raise ImportError(f"Could not find module '{library}' in "
                      f'{osp.dirname(__file__)}')

def flag_gemv_gemv_triton(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    flags = torch.abs(x_1) > threshold
    x = gather_gemv_elemul_flag_3d(x, x_1, Wup, flags)
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)

# cats
def gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = gather_gemv_elemul_flag_3d(x, x_1, Wup, flags)
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)

def gemv_gemv_triton_rsparse(x, Wupt, Wgatet, Wdownt, threshold_in, threshold_hidden):
    flags = torch.abs(x) > threshold_in
    act_fn = torch.nn.SiLU()

    x_gate = act_fn(gather_transposed_gemv_flag_3d(x, Wgatet, flags))
    x_up = gather_transposed_gemv_flag_3d(x, Wupt, flags)

    x = x_gate * x_up
    flags = torch.abs(x) > threshold_hidden
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)

# def gemv_gemv_triton_rsparse(x, Wupt, Wgatet, Wdownt, threshold_in, threshold_hidden, up_A, up_B, down_A, down_B, gate_A, gate_B):
#     flags = torch.abs(x) > threshold_in
#     float_flags = flags.to(x.dtype)
#     act_fn = torch.nn.SiLU()

#     x_gate_sparse = gather_transposed_gemv_flag_3d(x, Wgatet, flags)
#     # x_gate_low_rank = (x * (1-float_flags)) @ gate_A @ gate_B
#     x_gate_low_rank = 0
#     x_gate = act_fn(x_gate_sparse + x_gate_low_rank)

#     x_up_sparse = gather_transposed_gemv_flag_3d(x, Wupt, flags)
#     # x_up_low_rank = (x * (1-float_flags)) @ up_A @ up_B
#     x_up_low_rank = 0
#     x_up = x_up_sparse + x_up_low_rank

#     x = x_gate * x_up
#     flags = torch.abs(x) > threshold_hidden
#     float_flags = flags.to(x.dtype)

#     x_down_sparse = gather_transposed_gemv_flag_3d(x, Wdownt, flags)
#     # x_down_low_rank = (x * (1-float_flags)) @ down_A @ down_B
#     x_down_low_rank = 0

#     return x_down_sparse + x_down_low_rank


def gemv_gemv_triton_qkv_rsparse(x, Wqt, Wkt, Wvt, threshold_in):
    flags = torch.abs(x) > threshold_in
    q = gather_transposed_gemv_flag_3d(x, Wqt, flags)
    k = gather_transposed_gemv_flag_3d(x, Wkt, flags)
    v = gather_transposed_gemv_flag_3d(x, Wvt, flags)
    return q, k, v

def gemv_gemv_triton_o_rsparse(x, Wot, threshold_in):
    flags = torch.abs(x) > threshold_in
    return gather_transposed_gemv_flag_3d(x, Wot, flags)

# baseline
def flag_gemv_gemv_dejavu(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_partial_sparse(x, x_1[0][idx], Wup, Wdownt, idx)

# fuse deja vu
def flag_gemv_gemv_fuse_dejavu(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_sparse_direct_index_2d(x, x_1[0][idx], Wup, Wdownt, idx)

__all__ = [fuse_gemv_cmp, 
           fuse_gemv_flag, 
           fuse_gemv_flag_gemv, 
           fuse_gemv_flag_gemv_gemv, 
           fuse_gemv_gemv_gemv, 
           fuse_gemv_flag_batch, 
           fuse_gemv_flag_local, 
           fuse_flag_gemv_local, 
           atomic_gemv, 
           flag_gemv_gemv_atomic, 
           flag_gemv_gemv, 
           flag_gemv_gemv_inner, 
           flag_gemv_gemv_inner_fp32, 
           flag_gemv_gemv_inner_bf16, 
           flag_gemv_gemv_triton, 
           gemv_gemv_triton,
           gemv_gemv_triton_o_rsparse,
           gemv_gemv_triton_rsparse,
           gemv_gemv_triton_qkv_rsparse,
           flag_gemv_gemv_fuse_dejavu,
           flag_gemv_gemv_dejavu,
        ]
