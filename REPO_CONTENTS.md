# Repository contents

- `README.md` — the conversion recipe & gotchas (reproduction guide).
- `toolkit/` — the MiniMax-M3 integration for the per-expert NVFP4 quant
  pipeline (built on lukealonso/local-inference-lab quant-toolkit):
  - `minimax_m3.py` — ModelQuantConfig + the full checkpoint→native key
    translation table + per-expert MoE registration.
  - `moe_registry.py` — `_QuantM3FusedExperts` (vectorized per-expert
    swigluoai forward) + registration.
  - `streaming_loader.py` — key-translate + dense/shared gate_up
    pair-fusion in the streaming materializer.
  - `quantize.py` — ModelOpt 0.44 rule-list quant-config handling +
    rolling amax checkpoint.
  - `calib_minimax_m3.toml` — the 3-corpus calibration config.
- `scripts/` — `merge_amax.py` (split-executor elementwise-max merge),
  `finalize_vl_export.py` (vision-tensor verification + VL sidecar copy),
  `verify_export.py`, `upload_hf.py`.
- `capture/` — SM100 golden-oracle capture scripts (B200) used to validate
  the SM120 MSA kernel port.

Weights: https://huggingface.co/brandonmusic/MiniMax-M3-NVFP4
(model card links back here). Master amax tensors that reproduce the
export without recalibration are in that repo under `calibration/`.
