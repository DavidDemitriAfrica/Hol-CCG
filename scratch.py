import torch


torch.cuda.set_per_process_memory_fraction(0.8, device = 1) 
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2} MB")
print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**2} MB")