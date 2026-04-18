import torch
import triton
import triton.language as tl
import math

# =====================================================================
# 1. THE RAW TRITON GPU KERNEL (Compiles to Machine Code)
# =====================================================================
@triton.jit
def fused_additive_attention_kernel(
    # 1. Pointers to the start of our memory blocks
    q_ptr, k_ptr, v_ptr, v_a_ptr, out_ptr,
    
    # 2. Strides (How many steps to skip to get to the next dimension)
    # We only need strides for Q, K, V, and Out. 
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_dim,
    stride_o_batch, stride_o_head, stride_o_seq, stride_o_dim,
    
    # v_a is simpler: shape is [Heads, 1, Head_Dim], so we just need the head and dim strides
    stride_va_head, stride_va_dim,
    
    # 3. Shape Constants (Passed in from Python)
    num_batches, num_heads, seq_len, head_dim: tl.constexpr,
    
    # 4. Block Sizes (For breaking data into chunks that fit in SRAM cache)
    SEQ_BLOCK_SIZE: tl.constexpr,  # We will use 16
    HEAD_BLOCK_SIZE: tl.constexpr  # E.g., 32 or 64
):
    # 1. Which thread am I? (program_id maps to the grid we define in Python)
    batch_head_idx = tl.program_id(0)
    
    # Calculate exact batch and head index for this thread
    current_batch = batch_head_idx // num_heads
    current_head = batch_head_idx % num_heads

    # 2. Fast-forward the pointers to the exact starting location for this Batch & Head
    q_start = q_ptr + (current_batch * stride_q_batch) + (current_head * stride_q_head)
    k_start = k_ptr + (current_batch * stride_k_batch) + (current_head * stride_k_head)
    v_start = v_ptr + (current_batch * stride_v_batch) + (current_head * stride_v_head)
    out_start = out_ptr + (current_batch * stride_o_batch) + (current_head * stride_o_head)
    
    # v_a doesn't care about the batch, it only cares about the head!
    va_start = v_a_ptr + (current_head * stride_va_head)

    # 3. Create ranges (0 to 15, and 0 to 63) to grab a block of data
    seq_offsets = tl.arange(0, SEQ_BLOCK_SIZE)
    dim_offsets = tl.arange(0, HEAD_BLOCK_SIZE)

    # 4. Create safe masks (Because our sequence is only 3 tokens long, but the block is 16)
    # This prevents the GPU from reading memory that doesn't belong to it
    seq_mask = seq_offsets < seq_len
    dim_mask = dim_offsets < head_dim
    grid_mask = seq_mask[:, None] & dim_mask[None, :] # 2D mask

    # Calculate the exact memory addresses for every single element in our block
    q_addresses = q_start + (seq_offsets[:, None] * stride_q_seq) + (dim_offsets[None, :] * stride_q_dim)
    k_addresses = k_start + (seq_offsets[:, None] * stride_k_seq) + (dim_offsets[None, :] * stride_k_dim)
    v_addresses = v_start + (seq_offsets[:, None] * stride_v_seq) + (dim_offsets[None, :] * stride_v_dim)
    va_addresses = va_start + (dim_offsets * stride_va_dim) # 1D vector

    # Load the tensors into the cache! Replace out-of-bounds data with 0.0
    q_block = tl.load(q_addresses, mask=grid_mask, other=0.0)
    k_block = tl.load(k_addresses, mask=grid_mask, other=0.0)
    v_block = tl.load(v_addresses, mask=grid_mask, other=0.0)
    va_block = tl.load(va_addresses, mask=dim_mask, other=0.0)

    ### MATH ###

    # 1. Expand Q and K to create a 3D grid: [Seq_len, Seq_len, Head_dim]
    q_expanded = tl.expand_dims(q_block, 1) # Shape: (16, 1, 64)
    k_expanded = tl.expand_dims(k_block, 0) # Shape: (1, 16, 64)
    
    # 2. The Additive Interaction
    interaction = q_expanded + k_expanded
    
    # UPCAST to float32 for high-precision math
    interaction_fp32 = interaction.to(tl.float32)
    activated_fp32 = tl.extra.cuda.libdevice.tanh(interaction_fp32)
    
    # 3. Multiply by your learned v_a vector
    va_expanded = tl.expand_dims(tl.expand_dims(va_block, 0), 0).to(tl.float32)
    scaled_fp32 = activated_fp32 * va_expanded
    
    # 4. Collapse the feature dimension
    attention_scores = tl.sum(scaled_fp32, axis=2)

    ### SOFTMAX ###
    attention_mask = seq_mask[:, None] & seq_mask[None, :]
    attention_scores = tl.where(attention_mask, attention_scores, float("-inf"))
    
    # Stable Softmax (Still in fp32)
    max_scores = tl.max(attention_scores, axis=1)
    exp_scores = tl.extra.cuda.libdevice.exp(attention_scores - max_scores[:, None])
    sum_exp = tl.sum(exp_scores, axis=1)
    attention_weights_fp32 = exp_scores / sum_exp[:, None] 
    
    # DOWNCAST back to the input dtype (BF16) before the Matmul and Store
    attention_weights = attention_weights_fp32.to(v_block.dtype)
    final_output = tl.dot(attention_weights, v_block)
    
    # 4. Write the final computed block back to the slow VRAM
    out_addresses = out_start + (seq_offsets[:, None] * stride_o_seq) + (dim_offsets[None, :] * stride_o_dim)
    tl.store(out_addresses, final_output, mask=grid_mask)


# =====================================================================
# 2. THE PYTORCH WRAPPER (The Bridge to your FLB_Model.py)
# =====================================================================
class FusedAdditiveAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, v_a):
        shape = q.shape
        D = shape[-1]
        N = shape[-2]
        H = shape[-3]
        B_combined = math.prod(shape[:-3])
        
        out = torch.empty_like(q)
        
        BLOCK_N = max(16, triton.next_power_of_2(N))
        BLOCK_D = max(16, triton.next_power_of_2(D))
        
        grid = (B_combined * H, )
        fused_additive_attention_kernel[grid](
            q, k, v, v_a, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            v_a.stride(0), v_a.stride(2),
            B_combined, H, N, D, 
            BLOCK_N, BLOCK_D 
        )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, v_a = inputs
        ctx.save_for_backward(q, k, v, v_a)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, v_a = ctx.saved_tensors
        
        with torch.enable_grad():
            q_b, k_b, v_b, v_a_b = q.detach().requires_grad_(True), k.detach().requires_grad_(True), v.detach().requires_grad_(True), v_a.detach().requires_grad_(True)
            
            # 1. Standard interaction: [..., Seq, Seq, Head_Dim]
            interaction = torch.tanh(q_b.unsqueeze(-2) + k_b.unsqueeze(-3))
            
            # 2. THE BULLETPROOF FIX: 
            # We treat v_a as a flat pool of parameters and align it ONLY with 
            # the very last dimension (Head_Dim). We use .view on the interaction 
            # to make it 2D, multiply, then view it back.
            
            D = interaction.shape[-1]
            orig_shape = interaction.shape
            
            # Flatten everything except the last dimension
            interaction_flat = interaction.reshape(-1, D)
            
            # Reshape v_a to be 1D, but only take the last D elements 
            # (This handles the Head-slicing automatically)
            va_flat = v_a_b.reshape(-1)[-D:]
            
            # Perform the scaling in flattened space
            scaled_flat = interaction_flat * va_flat
            
            # Restore original shape
            scores = scaled_flat.reshape(orig_shape).sum(dim=-1)
            
            # 3. Softmax and Matmul
            attn = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(attn, v_b)
            
            out_b.backward(grad_out)
            
        return q_b.grad, k_b.grad, v_b.grad, v_a_b.grad

    @staticmethod
    def vmap(info, in_dims, q, k, v, v_a):
        # 1. Call apply as usual
        output = FusedAdditiveAttentionFunc.apply(q, k, v, v_a)
        
        # 2. FIX: Return '0' instead of '(0,)'
        # This tells PyTorch that the result is a single tensor 
        # with the batch dimension at index 0.
        return output, 0

# This is the single function you will import into FLB_Model.py!
def fused_additive_attention(q, k, v, v_a): # <--- ADDED v_a
    return FusedAdditiveAttentionFunc.apply(q, k, v, v_a)