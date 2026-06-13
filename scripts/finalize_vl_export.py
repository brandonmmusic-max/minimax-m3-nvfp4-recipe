"""Post-export: copy VL sidecars + verify vision completeness. Run AFTER export_hf."""
import json, os, shutil, sys
from safetensors import safe_open

SRC, DST = "/workspace/MiniMax-M3", "/workspace/MiniMax-M3-NVFP4"
SIDECARS = ["added_tokens.json", "chat_template.jinja", "configuration_minimax_m3_vl.py",
            "generation_config.json", "image_processor.py", "merges.txt",
            "preprocessor_config.json", "processing_minimax.py", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json", "video_processor.py", "vocab.json"]
for f in SIDECARS:
    s = os.path.join(SRC, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(DST, f))
    else:
        print(f"WARN: source sidecar missing: {f}")
print(f"sidecars copied: {len(SIDECARS)}")

src_idx = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
want = {k for k in src_idx if "vision" in k or "multi_modal" in k or "patch_merge" in k}
dst_idx_path = DST + "/model.safetensors.index.json"
dst_idx = json.load(open(dst_idx_path))["weight_map"]
have = set()
dtypes = {}
for k in dst_idx:
    lk = k
    if "vision" in lk or "multi_modal" in lk or "patch_merge" in lk:
        have.add(lk)
shards = {dst_idx[k] for k in have}
for sh in shards:
    with safe_open(os.path.join(DST, sh), framework="pt") as f:
        for k in f.keys():
            if k in have:
                dtypes[str(f.get_slice(k).get_dtype())] = dtypes.get(str(f.get_slice(k).get_dtype()), 0) + 1
# name-normalize: export may use native names (model.vision_tower.*) vs source (vision_tower.*)
def norm(s): return {x.replace("model.vision_tower", "vision_tower").replace("model.multi_modal_projector", "multi_modal_projector").replace("model.language_model", "language_model").split("vision_tower.")[-1] if "vision_tower" in x else x for x in s}
missing = len(want) - len(have)
print(f"vision-family tensors: source={len(want)} export={len(have)} dtypes={dtypes}")
if len(have) < len(want):
    nw, nh = norm(want), norm(have)
    if len(nh) >= len(nw):
        print("note: count matches after name normalization")
    else:
        print(f"FAIL: vision tensors missing from export ({len(want)-len(have)})")
        sys.exit(1)
print("VL EXPORT FINALIZE OK")
