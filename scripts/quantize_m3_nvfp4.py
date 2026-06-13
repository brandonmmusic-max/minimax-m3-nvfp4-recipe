#!/usr/bin/env python3
"""MiniMax-M3 -> NVFP4 PTQ via NVIDIA ModelOpt (8xB200).

Recipe: load bf16 across 8 GPUs, calibrate the LANGUAGE tower on text
samples, quantize Linear weights to NVFP4 (block-scaled), exclude the
vision tower / router gates / embeddings / lm_head, export a HF-layout
ModelOpt NVFP4 checkpoint.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoProcessor

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

MODEL_DIR = "/workspace/MiniMax-M3"
OUT_DIR = "/workspace/MiniMax-M3-NVFP4"
CALIB_N = int(os.environ.get("CALIB_N", "256"))
CALIB_LEN = int(os.environ.get("CALIB_LEN", "2048"))

def main():
    t0 = time.time()
    print(f"[load] tokenizer/config", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    print(f"[load] model bf16 device_map=auto across {torch.cuda.device_count()} GPUs", flush=True)
    # transformers main has NATIVE minimax_m3_vl support — use it
    # (trust_remote_code=False so the repo auto_map cannot shadow native).
    try:
        from transformers import AutoModelForImageTextToText as AM
        _maxmem = {i: "165GiB" for i in range(torch.cuda.device_count())}
        model = AM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                   device_map="auto", max_memory=_maxmem,
                                   trust_remote_code=False)
    except Exception as e:
        print(f"[load] Auto path failed ({e}); using direct class", flush=True)
        from transformers.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
        model = MiniMaxM3SparseForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[load] done in {time.time()-t0:.0f}s; structure:", flush=True)
    for name, _ in list(model.named_children()):
        print(f"   .{name}", flush=True)

    # language tower (VL models nest it; fall back to the model itself)
    lm = getattr(model, "language_model", None) or getattr(model, "model", model)

    # calibration data: simple text corpus through the tokenizer
    print(f"[calib] building {CALIB_N} samples x {CALIB_LEN} tokens", flush=True)
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"train[:{CALIB_N*3}]")
    texts = [r["article"] for r in ds if len(r["article"]) > 1000][:CALIB_N]
    embed_device = None
    for n, p in model.named_parameters():
        if "embed" in n:
            embed_device = p.device; break
    batches = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=CALIB_LEN).input_ids
        batches.append(ids.to(embed_device or "cuda:0"))

    def calib_loop(m):
        with torch.no_grad():
            for i, ids in enumerate(batches):
                try:
                    m(input_ids=ids)
                except TypeError:
                    model(input_ids=ids)
                if (i+1) % 32 == 0:
                    print(f"[calib] {i+1}/{len(batches)}", flush=True)

    import copy
    # ModelOpt 0.44 rule-list format. Base = NVIDIA's curated MoE NVFP4
    # recipe (already excludes lm_head/router/gates/shared_expert_gate/
    # output layers); append VL-tower + embedding + MTP exclusions.
    qcfg = copy.deepcopy(mtq.MAMBA_MOE_NVFP4_CONSERVATIVE_CFG)
    for _pat in ("*embed*", "*vision*", "*visual*", "*mtp*"):
        qcfg["quant_cfg"].append({"quantizer_name": _pat, "enable": False})

    # 2026-06-12: register M3's fused experts with ModelOpt's GptOss-style
    # quant wrapper — identical layout (gate_up_proj/down_proj 3D Parameters,
    # positional forward). Without this, routed expert banks (the bulk of
    # 428B) silently skip quantization.
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
    from modelopt.torch.quantization.plugins.huggingface import _QuantGptOssExperts
    from modelopt.torch.quantization.nn import QuantModuleRegistry
    if MiniMaxM3VLExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({MiniMaxM3VLExperts: "hf.MiniMaxM3VLExperts"})(_QuantGptOssExperts)
        print("[quantize] registered MiniMaxM3VLExperts -> _QuantGptOssExperts", flush=True)

    print("[quantize] NVFP4 PTQ starting", flush=True)
    t1 = time.time()
    mtq.quantize(lm, qcfg, calib_loop)
    print(f"[quantize] done in {time.time()-t1:.0f}s", flush=True)
    mtq.print_quant_summary(lm)

    print("[export] HF checkpoint ->", OUT_DIR, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    export_hf_checkpoint(model, export_dir=OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    try:
        AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True).save_pretrained(OUT_DIR)
    except Exception as e:
        print(f"[export] processor save skipped: {e}", flush=True)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
