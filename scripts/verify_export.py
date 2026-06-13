#!/usr/bin/env python3
"""Verify the exported NVFP4 checkpoint before upload."""
import json, os, sys
D = "/workspace/MiniMax-M3-NVFP4"
files = os.listdir(D)
shards = [f for f in files if f.endswith(".safetensors") and "amax" not in f]
total = sum(os.path.getsize(f"{D}/{f}") for f in shards) / 1e9
print(f"shards: {len(shards)}  total: {total:.1f} GB")
need = ["config.json"]
for n in need:
    print(f"{n}: {'OK' if os.path.exists(f'{D}/{n}') else 'MISSING'}")
hq = [f for f in files if "quant" in f.lower() and f.endswith(".json")]
print("quant config files:", hq)
cfg = json.load(open(f"{D}/config.json"))
print("quantization_config present:", "quantization_config" in cfg or bool(hq))
assert 180 < total < 320, f"size {total} outside NVFP4 expectations"
print("EXPORT VERIFICATION PASSED")
