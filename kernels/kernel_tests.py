import torch
import math
from .additive_attention import fused_additive_attention

# =====================================================================
# 1. THE GROUND TRUTH (Pure PyTorch Implementation)
# =====================================================================
def pure_torch_additive_attention(q, k, v, v_a):
    """
    A mathematically perfect, pure PyTorch implementation of our kernel.
    It uses standard Autograd and handles N-dimensional batching natively.
    """
    D = q.shape[-1]
    scale = math.sqrt(D)
    
    # 1. Interaction grid
    interaction = torch.tanh(q.unsqueeze(-2) + k.unsqueeze(-3))
    
    # 2. Align v_a for broadcasting [..., 1, D] -> [..., 1, 1, D]
    va_view = v_a.unsqueeze(-2)
    
    # 3. Scale and Sum
    scores = (interaction * va_view).sum(dim=-1) / scale
    
    # 4. Softmax and Matmul
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    
    return out

# =====================================================================
# 2. TEST UTILITIES
# =====================================================================
def assert_tensors_close(name, triton_tensor, torch_tensor, atol=1e-3):
    """Helper to cleanly print out exactly where tensors diverge."""
    match = torch.allclose(triton_tensor, torch_tensor, atol=atol)
    if match:
        print(f"✅ {name} matches perfectly!")
    else:
        diff = torch.abs(triton_tensor - torch_tensor)
        max_diff = diff.max().item()
        print(f"❌ {name} FAILED! Maximum difference: {max_diff:.6f}")
        
        # Identify NaN presence
        if torch.isnan(triton_tensor).any():
            print(f"   [!] Triton {name} contains NaNs!")
        if torch.isnan(torch_tensor).any():
            print(f"   [!] PyTorch {name} contains NaNs!")

# =====================================================================
# 3. THE TEST SUITE
# =====================================================================
def run_kernel_tests():
    print("=== Starting Custom Kernel Test Suite ===\n")
    torch.manual_seed(42)
    
    # Simulate a single batch of your architecture
    B, H, Seq, D = 2, 8, 3, 32 
    
    # Initialize inputs (using float32 for testing to isolate math errors from precision errors)
    q = torch.randn(B, H, Seq, D, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, Seq, D, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, Seq, D, dtype=torch.float32, device='cuda')
    v_a = torch.randn(H, 1, D, dtype=torch.float32, device='cuda')
    
    # Create distinct copies for the Triton and PyTorch paths so gradients don't mix
    q_tri, k_tri, v_tri, va_tri = [t.clone().requires_grad_(True) for t in (q, k, v, v_a)]
    q_tor, k_tor, v_tor, va_tor = [t.clone().requires_grad_(True) for t in (q, k, v, v_a)]
    
    # --- TEST 1: FORWARD PASS ---
    print("--- Test 1: Forward Pass Equivalence ---")
    out_tri = fused_additive_attention(q_tri, k_tri, v_tri, va_tri)
    out_tor = pure_torch_additive_attention(q_tor, k_tor, v_tor, va_tor)
    assert_tensors_close("Forward Output", out_tri, out_tor, atol=5e-3)

    # --- TEST 2: BACKWARD PASS (GRADIENTS) ---
    print("\n--- Test 2: Backward Pass Equivalence ---")
    # Create a dummy loss by summing the outputs, then backpropagate
    grad_output = torch.randn_like(out_tri) # Random upstream gradient
    
    out_tri.backward(grad_output, retain_graph=True)
    out_tor.backward(grad_output, retain_graph=True)
    
    assert_tensors_close("d_Q (Query Grad)", q_tri.grad, q_tor.grad)
    assert_tensors_close("d_K (Key Grad)", k_tri.grad, k_tor.grad)
    assert_tensors_close("d_V (Value Grad)", v_tri.grad, v_tor.grad)
    assert_tensors_close("d_Va (Attention Weight Grad)", va_tri.grad, va_tor.grad)

    # --- TEST 3: VMAP (WAVEFRONT SIMULATION) ---
    print("\n--- Test 3: VMAP / Batched Wavefront Simulation ---")
    Layers = 4
    
    # Create inputs with an extra 'Layers' dimension at the front
    q_vmap = torch.randn(Layers, B, H, Seq, D, dtype=torch.float32, device='cuda')
    k_vmap = torch.randn(Layers, B, H, Seq, D, dtype=torch.float32, device='cuda')
    v_vmap = torch.randn(Layers, B, H, Seq, D, dtype=torch.float32, device='cuda')
    va_vmap = torch.randn(Layers, H, 1, D, dtype=torch.float32, device='cuda')
    
    # Simulate how Wavefront runs it
    vmap_func = torch.vmap(fused_additive_attention)
    out_vmap = vmap_func(q_vmap, k_vmap, v_vmap, va_vmap)
    
    # Standard PyTorch handles N-dims automatically
    out_tor_vmap = torch.vmap(pure_torch_additive_attention)(q_vmap, k_vmap, v_vmap, va_vmap)
    
    assert_tensors_close("VMAP Forward Output", out_vmap, out_tor_vmap, atol=5e-3)
    print("\n===========================================")

if __name__ == "__main__":
    run_kernel_tests()