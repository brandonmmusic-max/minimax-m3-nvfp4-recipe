# MiniMax-M3 → NVFP4: conversion recipe & gotchas

How [brandonmusic/MiniMax-M3-NVFP4](https://huggingface.co/brandonmusic/MiniMax-M3-NVFP4)
(241 GB, multimodal-complete) was produced from MiniMaxAI/MiniMax-M3 (854 GB bf16)
on 8x B200, using TensorRT Model Optimizer 0.44 through a streaming
per-expert calibration pipeline (lukealonso/local-inference-lab quant-toolkit).
Weights are published to HF only; this doc is the reproduction guide.

## Why per-expert calibration
M3 ships fused 3D expert tensors ([E,2I,H]/[E,H,I]). Calibrating the fused
modules gives ONE amax per fused tensor; unfusing into per-expert Linears
(toolkit `_QuantFusedExperts` lineage) gives per-expert amax fidelity —
128 experts x 57 layers calibrated independently, re-fused at export.

## Pipeline shape
- Streaming loader: GPU0 executes; layers stored across remaining GPUs
  (+CPU overflow via `--cpu-capacity`); pre-forward hooks materialize each
  layer on GPU0 per batch.
- Max calibration: 445 batches x 300K tokens (~136M tokens) across three
  corpora (deep reasoning @8K, diverse @4K, agentic coding @4K).
- Cost model: time/batch = F (fixed ~17 s: 854 GB layer traffic) +
  V (~57 s per 300K tokens: ModelOpt per-expert observer machinery —
  ~44K QuantLinear calls/batch). V*total_tokens is invariant to batch
  size; bigger batches only amortize F.
- Max-calibration is partition-independent: we split batches across two
  executors (GPU-stored + CPU-pinned-stored) and merged amaxes with an
  elementwise max — bit-identical to a single run. Resume = rolling
  amax checkpoint every 50 batches; final export restores the (merged)
  amax file with `--resume-batch <total>` and runs ZERO batches.

## Gotchas (each cost a debugging round)
1. **M3 has no standalone modeling file** — repo auto_map covers config
   only. Use transformers main (>=5.10 dev) native `minimax_m3_vl`;
   checkpoint->native key renames live in transformers
   `conversion_mapping.py` (vision patch_embedding->proj,
   language_model prefix moves, block_sparse_moe->mlp, w1/w3/w2->
   gate/up/down, index_*->indexer.*). Translate keys BEFORE streaming
   load and verify zero-diff against the index.
2. **ModelOpt 0.44 quant_cfg is a rule LIST, not a dict** — dict-style
   `qcfg["quant_cfg"][pattern] = override` raises TypeError; append
   `{"quantizer_name": pattern, "enable"/"cfg": ...}` entries.
3. **Wildcard exclusions overmatch.** `*gate*` (meant for the router
   `mlp.gate`) silently disabled every expert `gate_proj` AND the
   shared/dense `gate_up_proj` -> a 456 GB "NVFP4" export (275 GB of
   bf16 gates). Detect: byte-sweep the export by tensor suffix x dtype.
   Recover WITHOUT recalibration: gate input amax := up_proj's (same
   input tensor, identical by construction); gate weight amax :=
   |w1|.max() recomputed from the source shards (weights are static).
   Re-export = ~25 min.
4. **Registry coverage is silent.** MiniMaxM3VLExperts must be
   registered in ModelOpt's QuantModuleRegistry or routed experts skip
   quantization with no warning. Verify enabled-quantizer counts before
   burning GPU-hours.
5. **swigluoai must survive quantization-aware forwards**: clamps
   (gate max 7.0, up ±7.0), `(up+1)`, alpha 1.702 inside the sigmoid —
   "same as GPT-OSS but NOT interleaved" (gate-then-up halves).
6. **Vision stays bf16**: exclude `*multi_modal_projector*` and
   `*patch_merge*` explicitly (and verify post-export: count
   vision-family tensors by dtype; ours: 523/523 BF16). Ship ALL VL
   sidecars (image/video processors, chat template, tokenizer) or the
   repo is not loadable as a VL model.
7. **Huge pinned-host registration stalls look like hangs**: a second
   executor pinning ~800 GB sat in D-state on an nvidia rwlock for
   ~16 min before its first batch. It resolves; don't kill it.
8. **Zombie executors hold GPU memory after tmux kill** — kill by PID
   from `nvidia-smi --query-compute-apps`, wait for full drain before
   relaunching (allocations take minutes to reap).
9. **CUDA dev headers** (cusparse/cusolver) are absent on some rental
   images — needed for kernel JIT alongside the quant.

## Serving note (SM120)
M3's MSA attention + this NVFP4 format run on SM120 (RTX PRO 6000) via a
b12x-based stack: golden-gated kernel port (SM100 oracle captures included
in the HF repo under `msa_golden/`), swigluoai-patched b12x fused MoE, and
a fail-closed runtime probe that rejects any MoE backend that silently
degrades the activation. Writeup of that port is separate.  The local model is being downloaded as we speak to check teh accuracy of the model, and to get a full docker image and recipe.   This read me will be updated.

## Acknowledgements

- **Luke Alonso** ([HuggingFace](https://huggingface.co/lukealonso) · [GitHub](https://github.com/lukealonso)) — author of the **b12x** SM120 kernel stack and the per-expert [quant-toolkit](https://github.com/local-inference-lab/quant-toolkit) calibration pipeline this recipe is built on.
- **[local-inference-lab/quant-toolkit](https://github.com/local-inference-lab/quant-toolkit)** — the streaming per-expert calibration toolkit and the calibration corpora used here (published in the model repo under `calibration/data/`).
- **[MiniMax](https://huggingface.co/MiniMaxAI)** — the base model, MiniMax-M3.

## Artifacts

- Weights + calibration amaxes + corpora: <https://huggingface.co/brandonmusic/MiniMax-M3-NVFP4>
  - `calibration/m3_merged_amax_gatefix.safetensors` regenerates the export in ~25 min with **no recalibration**.
  - `calibration/data/*.jsonl` are the quant-toolkit corpora, to recalibrate from scratch.
