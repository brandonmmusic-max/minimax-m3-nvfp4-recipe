# SPDX-License-Identifier: MIT
"""SM100 golden capture v2 — fmha_sm100 on B200, for the SM120 port gate.

Run: CUDA_VISIBLE_DEVICES=7 python3 capture_golden_v2.py
Saves one safetensors + meta per capture under /workspace/msa_golden/, and
this script itself alongside (v1's script was lost; never again).

Captures:
  A  nvfp4_kv : CSR-surface sparse attention, NVFP4 128x4 KV vs bf16 KV on
                identical selections (M3-like indexer flow, hkv=1, hq=8)
  B  varlen   : plan-surface, 3 segments, SHUFFLED kv_indices page table
  C  smscale  : plan-surface dense+maxscore at sm_scale=0.05 AND default —
                disambiguates raw-logit vs scaled max_score convention
  D  gqa      : plan-surface heads_kv=2, per-kv-head block grouping
"""
from __future__ import annotations

import json
import os
import shutil

import torch
from safetensors.torch import save_file

import fmha_sm100 as F
from fmha_sm100.sparse import Nvfp4QuantizedTensor, dequantize_nvfp4_128x4_to_bf16
from fmha_sm100.cute.quantize import swizzle_nvfp4_scale_to_128x4

_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def torch_quantize_nvfp4_128x4(x: torch.Tensor) -> "Nvfp4QuantizedTensor":
    """TE-equivalent NVFP4 quantizer in pure torch.

    transformer_engine_torch fails to build on this box, so this mirrors the
    TE NVFP4 recipe exactly as the package dequantizer defines it:
    global = amax/(448*6); per-16 block fp8e4m3 decode scales = block_amax/
    (6*global) (saturating RN cast); E2M1 nearest-level codes, sign bit 3;
    even element -> low nibble.  Validated by round-tripping through the
    package's own dequantize_nvfp4_128x4_to_bf16 and by the SM100 kernel.
    Tie-rounding at exact E2M1 midpoints may differ from TE (rounds up here,
    RN-even in hardware) — irrelevant for the kernel gate since the packed
    tensors themselves are saved.
    """
    orig_shape = tuple(int(s) for s in x.shape)
    d = orig_shape[-1]
    rows = x.numel() // d
    xf = x.reshape(rows, d).float()
    amax = xf.abs().amax()
    gs = (amax / (448.0 * 6.0)).clamp_min(1e-30).reshape(1)
    blocks = xf.reshape(rows, d // 16, 16)
    ba = blocks.abs().amax(-1)
    sc_fp8 = (ba / (6.0 * gs)).to(torch.float8_e4m3fn)
    div = sc_fp8.to(torch.float32) * gs
    y = blocks / div.clamp_min(1e-30).unsqueeze(-1)
    lv = torch.tensor(_FP4_LEVELS, device=x.device)
    mids = (lv[1:] + lv[:-1]) / 2
    idx = torch.bucketize(y.abs().clamp(max=6.0).contiguous(), mids)
    code = (idx + torch.where(y < 0, 8, 0)).to(torch.uint8)
    code = torch.where(div.unsqueeze(-1) == 0, torch.zeros_like(code), code)
    code = code.reshape(rows, d)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    scale_sw = swizzle_nvfp4_scale_to_128x4(sc_fp8, rows=rows, cols=d // 16)
    return Nvfp4QuantizedTensor(
        data=packed.reshape(*orig_shape[:-1], d // 2),
        scale_128x4=scale_sw.view(torch.uint8),
        global_scale=gs,
        logical_scale_shape=(rows, d // 16),
        original_shape=orig_shape,
    )

OUT = "/workspace/msa_golden"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda")
torch.manual_seed(42)


def lens(*xs):
    return torch.tensor(xs, dtype=torch.int32, device=DEV)


def save(name: str, tensors: dict, meta: dict) -> None:
    tensors = {k: v.contiguous().cpu() for k, v in tensors.items() if v is not None}
    save_file(tensors, f"{OUT}/{name}.safetensors")
    meta["tensors"] = {k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()}
    with open(f"{OUT}/{name}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"saved {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()))


def tdict(prefix: str, obj) -> dict:
    """Flatten every tensor attribute of an Nvfp4QuantizedTensor-ish object."""
    out = {}
    for k, v in vars(obj).items():
        if isinstance(v, torch.Tensor):
            out[f"{prefix}_{k}"] = v
    return out


# ---------------- A: NVFP4-KV via CSR surface ----------------
def capture_a():
    torch.manual_seed(42)
    total_q, hq, hkv, hd, page, topk = 512, 8, 1, 128, 128, 16
    kv_len = 64 * page
    cu_q, cu_k = lens(0, total_q), lens(0, kv_len)

    # indexer/proxy pass at kv-head granularity (MQA-compressed cache)
    proxy_q = torch.randn(total_q, hkv, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    proxy_k = torch.randn(kv_len, 1, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    proxy_v = torch.randn_like(proxy_k)
    plan = F.fmha_sm100_plan(
        cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1], hkv, 1,
        page_size=page, output_maxscore=True, causal=True,
    )
    _, ms = F.fmha_sm100(proxy_q, proxy_k, proxy_v, plan, output_maxscore=True, output_o=False)
    block_ids = F.sparse_topk_select(ms.contiguous(), topk, num_valid_pages=kv_len // page)

    real_q = torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    real_k = torch.randn(kv_len, hkv, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    real_v = torch.randn_like(real_k)
    q2k = block_ids.permute(1, 0, 2).contiguous()
    row_ptr, q_idx = F.build_k2q_csr(q2k, cu_q, cu_k, page, total_k=kv_len, max_seqlen_q=total_q, max_seqlen_k=kv_len)
    common = dict(
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q,
        max_seqlen_k=kv_len, blk_kv=page, causal=True,
    )
    out_bf16 = F.sparse_atten_func(real_q, real_k, real_v, row_ptr, q_idx, topk, **common)

    qk = torch_quantize_nvfp4_128x4(real_k)
    qv = torch_quantize_nvfp4_128x4(real_v)
    deq_k = dequantize_nvfp4_128x4_to_bf16(qk)
    deq_v = dequantize_nvfp4_128x4_to_bf16(qv)
    rt = torch.nn.functional.cosine_similarity(deq_k.float().flatten(), real_k.float().flatten(), dim=0)
    print(f"  A roundtrip dequant(quant(k)) vs k: cos={rt:.6f} (expect >0.99)")
    out_fp4 = F.sparse_atten_nvfp4_kv_func(
        real_q, qk.data.view(torch.uint8), qv.data.view(torch.uint8),
        qk.scale_128x4, qv.scale_128x4,
        qk.global_scale, qv.global_scale, row_ptr, q_idx, topk, **common,
    )
    # bf16 kernel on the DEQUANTIZED kv: the clean reference for any
    # dequant-based SM12x fallback path
    out_bf16_deq = F.sparse_atten_func(real_q, deq_k, deq_v, row_ptr, q_idx, topk, **common)
    cos = torch.nn.functional.cosine_similarity(out_fp4.float().flatten(), out_bf16.float().flatten(), dim=0)
    cos2 = torch.nn.functional.cosine_similarity(out_fp4.float().flatten(), out_bf16_deq.float().flatten(), dim=0)
    print(f"  A sanity: fp4-vs-bf16 cos={cos:.6f} (expect >0.98); fp4-vs-deq-bf16 cos={cos2:.6f}; nan={out_fp4.isnan().any().item()}")
    save(
        "msa_sm100_golden_v2a_nvfp4kv",
        dict(
            proxy_q=proxy_q, proxy_k=proxy_k, proxy_v=proxy_v, max_score=ms,
            block_ids=block_ids, row_ptr=row_ptr, q_idx=q_idx,
            q=real_q, k=real_k, v=real_v, out_bf16=out_bf16, out_nvfp4=out_fp4,
            out_bf16_dequant=out_bf16_deq, dequant_k=deq_k, dequant_v=deq_v,
            kq_data=qk.data.view(torch.uint8), kq_scale_128x4=qk.scale_128x4,
            kq_global_scale=qk.global_scale,
            vq_data=qv.data.view(torch.uint8), vq_scale_128x4=qv.scale_128x4,
            vq_global_scale=qv.global_scale,
        ),
        dict(
            capture="A nvfp4_kv CSR surface", seed=42, total_q=total_q, hq=hq, hkv=hkv,
            head_dim=hd, page_size=page, topk=topk, kv_len=kv_len, causal=True,
            quantizer="pure-torch TE-equivalent (transformer_engine_torch unbuildable on box); scales stored as uint8 views of float8_e4m3fn",
            note="proxy indexer flow; out_bf16/out_nvfp4/out_bf16_dequant share identical CSR; softmax_scale default",
        ),
    )


# ---------------- B: multi-segment varlen, shuffled page table ----------------
def capture_b():
    torch.manual_seed(43)
    hq, hkv, hd, page, topk = 8, 1, 128, 128, 16
    qo = [128, 64, 256]
    kvl = [2048, 2048, 4096]  # 16+16+32 = 64 pages, every segment >= topk pages
    n_pages = sum(x // page for x in kvl)
    q = torch.randn(sum(qo), hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp = torch.randn(n_pages, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp = torch.randn_like(kp)
    kv_indices = torch.randperm(n_pages, device=DEV).to(torch.int32)  # SHUFFLED

    plan = F.fmha_sm100_plan(
        lens(*qo), lens(*kvl), hq, hkv,
        page_size=page, output_maxscore=True, causal=True,
    )
    dense_out, ms = F.fmha_sm100(q, kp, vp, plan, kv_indices=kv_indices, output_maxscore=True)
    kbi = F.sparse_topk_select(ms.contiguous(), topk, num_valid_pages=None)
    sparse_plan = F.fmha_sm100_plan(
        lens(*qo), lens(*kvl), hq, hkv,
        page_size=page, kv_block_num=topk, causal=True,
    )
    sparse_out, _ = F.fmha_sm100(
        q, kp, vp, sparse_plan, kv_indices=kv_indices,
        kv_block_indexes=kbi[:, :hkv, :].contiguous(),
    )
    print(f"  B sanity: dense nan={dense_out.isnan().any().item()} sparse nan={sparse_out.isnan().any().item()}")
    save(
        "msa_sm100_golden_v2b_varlen",
        dict(q=q, k_pages=kp, v_pages=vp, kv_indices=kv_indices, dense_out=dense_out,
             max_score=ms, kv_block_indexes=kbi, sparse_out=sparse_out),
        dict(capture="B varlen plan surface", seed=43, qo_segment_lens=qo, kv_segment_lens=kvl,
             hq=hq, hkv=hkv, head_dim=hd, page_size=page, topk=topk, causal=True,
             note="kv_indices is a random permutation; sparse pass fed kbi[:, :1, :] (kv-head shared)"),
    )


# ---------------- C: custom sm_scale ----------------
def capture_c():
    torch.manual_seed(44)
    total_q, hq, hkv, hd, page = 256, 8, 1, 128, 128
    kv_len = 32 * page
    q = torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp = torch.randn(32, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp = torch.randn_like(kp)
    kvi = torch.arange(32, dtype=torch.int32, device=DEV)
    plan = F.fmha_sm100_plan(
        lens(total_q), lens(kv_len), hq, hkv,
        page_size=page, output_maxscore=True, causal=True,
    )
    out_custom, ms_custom = F.fmha_sm100(
        q, kp, vp, plan, kv_indices=kvi, output_maxscore=True, sm_scale=0.05,
    )
    out_default, ms_default = F.fmha_sm100(q, kp, vp, plan, kv_indices=kvi, output_maxscore=True)
    ignored = bool(torch.equal(out_custom, out_default))
    print(f"  C sanity: sm_scale kwarg ignored by SM100? {ignored}")
    save(
        "msa_sm100_golden_v2c_smscale",
        dict(q=q, k_pages=kp, v_pages=vp, kv_indices=kvi, out_sm005=out_custom,
             ms_sm005=ms_custom, out_default=out_default, ms_default=ms_default),
        dict(capture="C custom sm_scale", seed=44, total_q=total_q, hq=hq, hkv=hkv,
             head_dim=hd, page_size=page, kv_len=kv_len, causal=True, sm_scale_custom=0.05,
             sm_scale_kwarg_ignored=ignored,
             note="if ms_sm005 == ms_default, max_score is scale-independent = raw q.k proven"),
    )


# ---------------- D: GQA heads_kv=2 ----------------
def capture_d():
    torch.manual_seed(45)
    total_q, hq, hkv, hd, page, topk = 256, 8, 2, 128, 128, 16
    n_pages = 32
    kv_len = n_pages * page
    q = torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp = torch.randn(n_pages, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp = torch.randn_like(kp)
    kvi = torch.arange(n_pages, dtype=torch.int32, device=DEV)
    plan = F.fmha_sm100_plan(
        lens(total_q), lens(kv_len), hq, hkv,
        page_size=page, output_maxscore=True, causal=True,
    )
    dense_out, ms = F.fmha_sm100(q, kp, vp, plan, kv_indices=kvi, output_maxscore=True)
    print(f"  D max_score heads axis = {ms.shape[0]} (hq={hq}, hkv={hkv})")
    kbi_full = F.sparse_topk_select(ms.contiguous(), topk, num_valid_pages=n_pages)
    # per-kv-head selection: one row per kv head group
    if kbi_full.shape[1] == hq:
        kbi_kv = kbi_full[:, :: hq // hkv, :].contiguous()  # heads 0 and 4 lead groups
    else:
        kbi_kv = kbi_full.contiguous()
    sparse_plan = F.fmha_sm100_plan(
        lens(total_q), lens(kv_len), hq, hkv,
        page_size=page, kv_block_num=topk, causal=True,
    )
    sparse_out, _ = F.fmha_sm100(q, kp, vp, sparse_plan, kv_indices=kvi, kv_block_indexes=kbi_kv)
    print(f"  D sanity: dense nan={dense_out.isnan().any().item()} sparse nan={sparse_out.isnan().any().item()}")
    save(
        "msa_sm100_golden_v2d_gqa",
        dict(q=q, k_pages=kp, v_pages=vp, kv_indices=kvi, dense_out=dense_out,
             max_score=ms, kv_block_indexes_full=kbi_full, kv_block_indexes_kv=kbi_kv,
             sparse_out=sparse_out),
        dict(capture="D gqa hkv=2", seed=45, total_q=total_q, hq=hq, hkv=hkv, head_dim=hd,
             page_size=page, topk=topk, kv_len=kv_len, causal=True,
             note="sparse pass fed kv_block_indexes_kv [q, hkv, topk]; heads 0-3 -> kv0, 4-7 -> kv1"),
    )


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()}")
    for fn in (capture_a, capture_b, capture_c, capture_d):
        print(f"== {fn.__name__} ==")
        fn()
    shutil.copy(__file__, f"{OUT}/capture_golden_v2.py")
    print("ALL CAPTURES DONE")
