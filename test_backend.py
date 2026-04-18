import torch
import wavefront_backend

print("--- Testing C++ Extension ---")

# Create a dummy tensor
dummy_buffer = torch.zeros((5, 3))
print("Before C++ Loop:")
print(dummy_buffer)

# Run our custom C++ function!
wavefront_backend.test_loop(5, dummy_buffer)

print("\nAfter C++ Loop:")
print(dummy_buffer)