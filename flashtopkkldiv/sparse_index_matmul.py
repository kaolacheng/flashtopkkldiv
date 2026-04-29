import torch
import triton
import triton.language as tl
from torch.library import custom_op
from typing import Tuple

@triton.jit()
def _sparse_index_matmul_kernel(
    X_p, E_p, Idx_p, Y_p,
    N, K, D,
    stride_xn, stride_xd,
    stride_ev, stride_ed,
    stride_idxn, stride_idxk,
    stride_yn, stride_yk,
    BLOCK_SIZE_D: tl.constexpr
): #我说triton就是魔法有没有懂得
    pid = tl.program_id(0)
    if pid >= N:
        return 
    
    offs_xn = pid * stride_xn
    offs_idxn = pid * stride_idxn
    offs_yn = pid * stride_yn

    for k in range(K):
        idx_ptr = Idx_p + offs_idxn + k * stride_idxk
        vocab_idx = tl.load(idx_ptr)
        
        acc = 0.0#tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
            d_mask = d_offs < D

            x_ps = X_p + offs_xn + d_offs * stride_xd
            x = tl.load(x_ps, mask=d_mask, other=0.0)
            
            e_ps = E_p + vocab_idx * stride_ev + d_offs * stride_ed
            e = tl.load(e_ps, mask=d_mask, other=0.0)

            acc += tl.sum(x * e)

        y_ptr = Y_p + offs_yn + k * stride_yk
        tl.store(y_ptr, acc)


@triton.jit()
def _sparse_index_matmul_backward_dx_kernel(
    GY_p, E_p, Idx_p, GX_p,
    N, K, D,
    stride_gyn, stride_gyk,
    stride_ev, stride_ed,
    stride_idxn, stride_idxk,
    stride_gxn, stride_gxd,
    BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return 

    offs_gxn = pid * stride_gxn
    offs_idxn = pid * stride_idxn

    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offs < D
        
        gx_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        for k in range(K):
            idx_ptr = Idx_p + offs_idxn + k * stride_idxk
            vocab_idx = tl.load(idx_ptr)
            
            gy_ptr = GY_p + pid * stride_gyn + k * stride_gyk
            gy = tl.load(gy_ptr)
            
            e_ps = E_p + vocab_idx * stride_ev + d_offs * stride_ed
            e = tl.load(e_ps, mask=d_mask, other=0.0)
            
            gx_acc += gy * e
            
        gx_ptr = GX_p + offs_gxn + d_offs * stride_gxd
        tl.store(gx_ptr, gx_acc, mask=d_mask)

@triton.jit()
def _sparse_index_matmul_backward_de_kernel(
    GY_p, X_p, Idx_p, GE_p,
    N, K, D,
    stride_gyn, stride_gyk,
    stride_xn, stride_xd,
    stride_idxn, stride_idxk,
    stride_gev, stride_ged,
    BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(0)
    NK = N * K
    if pid >= NK:
        return 
        
    n = pid // K
    k = pid % K
    
    idx_ptr = Idx_p + n * stride_idxn + k * stride_idxk
    vocab_idx = tl.load(idx_ptr)
    
    gy_ptr = GY_p + n * stride_gyn + k * stride_gyk
    gy = tl.load(gy_ptr)

    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offs < D
        
        x_ps = X_p + n * stride_xn + d_offs * stride_xd
        x = tl.load(x_ps, mask=d_mask, other=0.0)
        
        ge_val = gy * x
        
        ge_ptr = GE_p + vocab_idx * stride_gev + d_offs * stride_ged
        tl.atomic_add(ge_ptr, ge_val, mask=d_mask) #这不支持turing，但是谁会求e的梯度呢？

def _launch_sparse_matmul(x, e, idx):
    N, D = x.shape
    K = idx.shape[1]
    out = torch.empty((N, K), device=x.device, dtype=x.dtype)
    BLOCK_SIZE_D = 256 if D >= 256 else triton.next_power_of_2(D)
    grid = (N,)
    _sparse_index_matmul_kernel[grid](
        x, e, idx, out,
        N, K, D,
        x.stride(0), x.stride(1),
        e.stride(0), e.stride(1),
        idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return out

def _launch_backward(grad_output, x, e, idx,need_x_grad=True,need_e_grad=False):
    N, D = x.shape
    K = idx.shape[1]
    BLOCK_SIZE_D = 256 if D >= 256 else triton.next_power_of_2(D)    
    grad_e = torch.zeros_like(e) 
    grad_x = torch.zeros_like(x)
    if need_x_grad:
        grid_dx = (N,)
        _sparse_index_matmul_backward_dx_kernel[grid_dx](
            grad_output, e, idx, grad_x,
            N, K, D,
            grad_output.stride(0), grad_output.stride(1),
            e.stride(0), e.stride(1),
            idx.stride(0), idx.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
    if need_e_grad:
        grid_de = (N * K,)
        _sparse_index_matmul_backward_de_kernel[grid_de](
            grad_output, x, idx, grad_e,
            N, K, D,
            grad_output.stride(0), grad_output.stride(1),
            x.stride(0), x.stride(1),
            idx.stride(0), idx.stride(1),
            grad_e.stride(0), grad_e.stride(1),
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
    
    return grad_x, grad_e


@custom_op("sparse_matmul::sparse_index_matmul_backward", mutates_args=())
def sparse_index_matmul_backward_op(grad_output: torch.Tensor, x: torch.Tensor, e: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _launch_backward(grad_output, x, e, idx)

@sparse_index_matmul_backward_op.register_fake
def sparse_index_matmul_backward_fake(grad_output, x, e, idx):

    return torch.empty_like(x), torch.empty_like(e)

@custom_op("sparse_matmul::sparse_index_matmul", mutates_args=())
def sparse_index_matmul(x: torch.Tensor, e: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return _launch_sparse_matmul(x, e, idx)

@sparse_index_matmul.register_fake
def sparse_index_matmul_fake(x, e, idx):
    return torch.empty(x.shape[0], idx.shape[1], device=x.device, dtype=x.dtype)

def sparse_index_matmul_setup_context(ctx, inputs, output):
    x, e, idx = inputs
    ctx.save_for_backward(x, e, idx)

def sparse_index_matmul_backward(ctx, grad_output):
    x, e, idx = ctx.saved_tensors
    grad_x, grad_e = torch.ops.sparse_matmul.sparse_index_matmul_backward(grad_output, x, e, idx)
    
    if not ctx.needs_input_grad[1]:
        grad_e = None
    if not ctx.needs_input_grad[0]:
        grad_x = None
        
    return grad_x, grad_e, None

sparse_index_matmul.register_autograd(
    sparse_index_matmul_backward, 
    setup_context=sparse_index_matmul_setup_context
)

def sparse_index_matmul_lib(x, e, idx):
    return torch.ops.sparse_matmul.sparse_index_matmul(x, e, idx)
