import sys
import torch
from safetensors.torch import load_file, save_file

a = load_file("/workspace/m3_exec1_amax_300.safetensors")
b = load_file("/workspace/m3_tail_amax.safetensors")
ka, kb = set(a), set(b)
if ka != kb:
    print(f"KEY MISMATCH: only-exec1={len(ka-kb)} only-exec2={len(kb-ka)}")
    for k in list(ka ^ kb)[:10]:
        print("  ", k)
    sys.exit(1)
merged = {k: torch.maximum(a[k], b[k]) for k in a}
n_a = sum((merged[k] == a[k]).all().item() for k in merged)
save_file(merged, "/workspace/m3_merged_amax.safetensors")
print(f"MERGED {len(merged)} tensors -> m3_merged_amax.safetensors")
print(f"sanity: {n_a}/{len(merged)} tensors already covered by exec1@300 (rest grew from tail)")
