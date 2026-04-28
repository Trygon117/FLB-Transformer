import torch
torch.set_float32_matmul_precision('high')
import triton
import triton.language as tl
import math

# =====================================================================
# 1. THE RAW TRITON GPU KERNEL (Optimized Math)
# =====================================================================
@triton.jit
def fused_additive_attention_kernel(
    q_ptr, k_ptr, v_ptr, v_a_ptr, out_ptr,
    
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_dim,
    stride_o_batch, stride_o_head, stride_o_seq, stride_o_dim,
    
    stride_va_batch, stride_va_head, stride_va_dim, 
    
    num_batches, num_heads, seq_len, head_dim: tl.constexpr,
    scale,
    SEQ_BLOCK_SIZE: tl.constexpr,  
    HEAD_BLOCK_SIZE: tl.constexpr 
):
    batch_head_idx = tl.program_id(0)
    current_batch = batch_head_idx // num_heads
    current_head = batch_head_idx % num_heads

    q_start = q_ptr + (current_batch * stride_q_batch) + (current_head * stride_q_head)
    k_start = k_ptr + (current_batch * stride_k_batch) + (current_head * stride_k_head)
    v_start = v_ptr + (current_batch * stride_v_batch) + (current_head * stride_v_head)
    out_start = out_ptr + (current_batch * stride_o_batch) + (current_head * stride_o_head)
    va_start = v_a_ptr + (current_batch * stride_va_batch) + (current_head * stride_va_head)

    seq_offsets = tl.arange(0, SEQ_BLOCK_SIZE)
    dim_offsets = tl.arange(0, HEAD_BLOCK_SIZE)

    seq_mask = seq_offsets < seq_len
    dim_mask = dim_offsets < head_dim
    grid_mask = seq_mask[:, None] & dim_mask[None, :] 

    q_addresses = q_start + (seq_offsets[:, None] * stride_q_seq) + (dim_offsets[None, :] * stride_q_dim)
    k_addresses = k_start + (seq_offsets[:, None] * stride_k_seq) + (dim_offsets[None, :] * stride_k_dim)
    v_addresses = v_start + (seq_offsets[:, None] * stride_v_seq) + (dim_offsets[None, :] * stride_v_dim)
    va_addresses = va_start + (dim_offsets * stride_va_dim)

    q_block = tl.load(q_addresses, mask=grid_mask, other=0.0)
    k_block = tl.load(k_addresses, mask=grid_mask, other=0.0)
    v_block = tl.load(v_addresses, mask=grid_mask, other=0.0)
    va_block = tl.load(va_addresses, mask=dim_mask, other=0.0)

    q_expanded = tl.expand_dims(q_block, 1) 
    k_expanded = tl.expand_dims(k_block, 0) 
    
    interaction = q_expanded + k_expanded
    
    interaction_fp32 = interaction.to(tl.float32)
    # OPTIMIZATION: Use native Triton math instead of CUDA libdevice
    activated_fp32 = tl.extra.cuda.libdevice.tanh(interaction_fp32)
    
    va_expanded = tl.expand_dims(tl.expand_dims(va_block, 0), 0).to(tl.float32)
    scaled_fp32 = activated_fp32 * va_expanded
    
    attention_scores = tl.sum(scaled_fp32, axis=2)
    attention_scores = attention_scores / scale

    attention_mask = seq_mask[:, None] & seq_mask[None, :]
    attention_scores = tl.where(attention_mask, attention_scores, float("-inf"))
    
    safe_max_scores = tl.where(seq_mask[:, None], attention_scores, 0.0)
    max_scores = tl.max(safe_max_scores, axis=1)
    
    # OPTIMIZATION: Use native Triton math
    exp_scores = tl.exp(attention_scores - max_scores[:, None])
    
    exp_scores = tl.where(attention_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=1) + 1e-6  
    
    attention_weights_fp32 = exp_scores / sum_exp[:, None]
    
    attention_weights = attention_weights_fp32.to(v_block.dtype)
    final_output = tl.dot(attention_weights, v_block)
    
    out_addresses = out_start + (seq_offsets[:, None] * stride_o_seq) + (dim_offsets[None, :] * stride_o_dim)
    tl.store(out_addresses, final_output, mask=grid_mask)

# =====================================================================
# 2. PURE COMPILED BACKWARD MATH (Bypasses the Dynamo Context Bug)
# =====================================================================
@torch.compile(backend="aot_eager")
def compiled_backward_math(q_b, k_b, v_b, v_a_b, grad_out, scale):
    """
    By isolating this math outside the autograd function, PyTorch will aggressively 
    fuse these operations into a single fast kernel without crashing.
    """
    interaction = torch.tanh(q_b.unsqueeze(-2) + k_b.unsqueeze(-3))
    
    vmap_dims = v_a_b.shape[:-3]
    H = v_a_b.shape[-3]
    D = v_a_b.shape[-1]
    num_batch_dims = interaction.ndim - len(vmap_dims) - 4
    va_shape = vmap_dims + (1,) * num_batch_dims + (H, 1, 1, D)
    va_view = v_a_b.view(va_shape)
        
    scores = (interaction * va_view).sum(dim=-1) / scale
    attn = torch.softmax(scores, dim=-1)
    
    attn_T = attn.transpose(-1, -2).contiguous()
    v_T = v_b.transpose(-1, -2).contiguous()

    grad_v = torch.matmul(attn_T, grad_out)
    grad_attn = torch.matmul(grad_out, v_T)
    
    grad_scores = attn * (grad_attn - (attn * grad_attn).sum(dim=-1, keepdim=True))
    grad_scores = grad_scores / scale 
    
    grad_interaction_pre_tanh = grad_scores.unsqueeze(-1) * va_view
    grad_va_raw = grad_scores.unsqueeze(-1) * interaction
    
    grad_qk_sum = grad_interaction_pre_tanh * (1.0 - interaction ** 2)
    
    grad_q = grad_qk_sum.sum(dim=-2)
    grad_k = grad_qk_sum.sum(dim=-3)
    
    keep_dims = list(range(len(vmap_dims))) + [interaction.ndim - 4, interaction.ndim - 1]
    sum_dims = [i for i in range(interaction.ndim) if i not in keep_dims]
    grad_va = grad_va_raw.sum(dim=sum_dims).view(v_a_b.shape)
        
    return grad_q, grad_k, grad_v, grad_va

# =====================================================================
# 3. THE PYTORCH WRAPPER
# =====================================================================
class FusedAdditiveAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, v_a):
        shape = q.shape
        D = shape[-1]
        N = shape[-2]
        H = shape[-3]
        
        batch_shape = shape[:-3]
        B_combined = math.prod(batch_shape)
        scale = math.sqrt(D)
        
        # OPTIMIZATION: Use reshape to avoid contiguous VRAM copies
        q_flat = q.reshape(B_combined, H, N, D)
        k_flat = k.reshape(B_combined, H, N, D)
        v_flat = v.reshape(B_combined, H, N, D)
        out_flat = torch.empty_like(q_flat)
        
        aligned_v_a = v_a
        while aligned_v_a.ndim < q.ndim:
            aligned_v_a = aligned_v_a.unsqueeze(-4)
            
        va_expected_shape = batch_shape + (H, 1, D)
        # Reshape avoids deep copy allocations if the broadcast logic permits
        v_a_flat = torch.broadcast_to(aligned_v_a, va_expected_shape).reshape(B_combined, H, 1, D)
        
        BLOCK_N = max(16, triton.next_power_of_2(N))
        BLOCK_D = max(16, triton.next_power_of_2(D))
        
        grid = (B_combined * H, )
        
        fused_additive_attention_kernel[grid](
            q_flat, k_flat, v_flat, v_a_flat, out_flat,
            
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2), q_flat.stride(3),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2), k_flat.stride(3),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2), v_flat.stride(3),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2), out_flat.stride(3),
            
            v_a_flat.stride(0), v_a_flat.stride(1), v_a_flat.stride(3),
            
            B_combined, H, N, D, 
            scale,
            BLOCK_N, BLOCK_D 
        )
        
        return out_flat.view(shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, v_a = inputs
        ctx.save_for_backward(q, k, v, v_a)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_out):
        q, k, v, v_a = ctx.saved_tensors
        scale = math.sqrt(q.shape[-1])
        
        # 1. Align the incoming gradient memory
        grad_out = grad_out.contiguous()

        # 2. OPTIMIZATION: Call the isolated, pre-compiled pure math function!
        return compiled_backward_math(q, k, v, v_a, grad_out, scale)

    @staticmethod
    def vmap(info, in_dims, q, k, v, v_a):
        output = FusedAdditiveAttentionFunc.apply(q, k, v, v_a)
        return output, 0

def fused_additive_attention(q, k, v, v_a):
    return FusedAdditiveAttentionFunc.apply(q, k, v, v_a)