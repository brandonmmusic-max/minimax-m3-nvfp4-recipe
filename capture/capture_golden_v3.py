# SPDX-License-Identifier: MIT
"""SM100 golden capture v3 — edge/dtype/indexer hardening batch.

Sections (each independent; failures print loudly and are recorded):
  F edges    : pages<topk (-1 pad consumption), partial last page,
               duplicate block ids, force_begin/end_blocks placement
  G lse      : prefill CSR LSE + temperature-LSE + custom softmax_scale
  H indexer  : fp4_indexer_block_scores (the production selector matmul)
  I decode2  : split-KV long-context decode + determinism
  J fp8pre   : fp8-storage prefill (QK fp8; PV fp8 vs bf16-staged lever)
  K bigshape : multi-CTA prefill (total_q=4096, 256 pages) + determinism
"""
from __future__ import annotations

import json
import shutil
import subprocess

import torch
from safetensors.torch import save_file

import fmha_sm100 as F
from fmha_sm100.sparse import fp4_indexer_block_scores

OUT = "/workspace/msa_golden"
DEV = torch.device("cuda")
RESULTS: dict[str, str] = {}

_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def lens(*xs):
    return torch.tensor(xs, dtype=torch.int32, device=DEV)


def save(name: str, tensors: dict, meta: dict) -> None:
    tensors = {k: v.contiguous().cpu() for k, v in tensors.items() if v is not None}
    save_file(tensors, f"{OUT}/{name}.safetensors")
    meta["tensors"] = {k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()}
    with open(f"{OUT}/{name}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  saved {name}")


def quant_public_fp4(x: torch.Tensor):
    """NVFP4 with PUBLIC (logical) per-16-block scales, no global scale.
    x [..., d] bf16 -> (packed uint8 [..., d/2], scales fp8e4m3 [..., d/16])."""
    orig = tuple(int(s) for s in x.shape)
    d = orig[-1]
    rows = x.numel() // d
    xf = x.reshape(rows, d).float()
    blocks = xf.reshape(rows, d // 16, 16)
    sc_fp8 = (blocks.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    div = sc_fp8.to(torch.float32).clamp_min(1e-30).unsqueeze(-1)
    y = blocks / div
    lv = torch.tensor(_FP4_LEVELS, device=x.device)
    mids = (lv[1:] + lv[:-1]) / 2
    idx = torch.bucketize(y.abs().clamp(max=6.0).contiguous(), mids)
    code = (idx + torch.where(y < 0, 8, 0)).to(torch.uint8)
    code = torch.where(sc_fp8.unsqueeze(-1).to(torch.float32) == 0, torch.zeros_like(code), code)
    code = code.reshape(rows, d)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed.reshape(*orig[:-1], d // 2),
            sc_fp8.reshape(*orig[:-1], d // 16).contiguous())


# ---------------- F: edge semantics ----------------
def sec_f():
    torch.manual_seed(50)
    hq, hkv, hd, page, topk = 8, 1, 128, 128, 16
    # F1: 8 pages < topk 16 -> -1 padded selection consumed by sparse kernel
    kv1 = 8 * page
    q1 = torch.randn(64, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp1 = torch.randn(8, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp1 = torch.randn_like(kp1)
    kvi1 = torch.arange(8, dtype=torch.int32, device=DEV)
    plan1 = F.fmha_sm100_plan(lens(64), lens(kv1), hq, hkv, page_size=page,
                              output_maxscore=True, causal=True)
    dense1, ms1 = F.fmha_sm100(q1, kp1, vp1, plan1, kv_indices=kvi1, output_maxscore=True)
    kbi1 = F.sparse_topk_select(ms1.contiguous(), topk, num_valid_pages=8)
    splan1 = F.fmha_sm100_plan(lens(64), lens(kv1), hq, hkv, page_size=page,
                               kv_block_num=topk, causal=True)
    sp1, _ = F.fmha_sm100(q1, kp1, vp1, splan1, kv_indices=kvi1,
                          kv_block_indexes=kbi1[:, :hkv, :].contiguous())
    print(f"  F1 kbi row0: {kbi1[0,0].tolist()}")
    # F2: partial last page kv_len=2000 (15 full + 80)
    kv2 = 2000
    q2 = torch.randn(128, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp2 = torch.randn(16, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp2 = torch.randn_like(kp2)
    kvi2 = torch.arange(16, dtype=torch.int32, device=DEV)
    plan2 = F.fmha_sm100_plan(lens(128), lens(kv2), hq, hkv, page_size=page,
                              output_maxscore=True, causal=True)
    dense2, ms2 = F.fmha_sm100(q2, kp2, vp2, plan2, kv_indices=kvi2, output_maxscore=True)
    kbi2 = F.sparse_topk_select(ms2.contiguous(), topk, num_valid_pages=16)
    splan2 = F.fmha_sm100_plan(lens(128), lens(kv2), hq, hkv, page_size=page,
                               kv_block_num=topk, causal=True)
    sp2, _ = F.fmha_sm100(q2, kp2, vp2, splan2, kv_indices=kvi2,
                          kv_block_indexes=kbi2[:, :hkv, :].contiguous())
    # F3: duplicate block ids (col 15 := col 0 dup)
    kbi3 = kbi2[:, :hkv, :].contiguous().clone()
    kbi3[:, :, topk - 1] = kbi3[:, :, 0]
    sp3, _ = F.fmha_sm100(q2, kp2, vp2, splan2, kv_indices=kvi2, kv_block_indexes=kbi3)
    # F4: force flags that MATTER (suppress pages 0,1,63 then force them)
    ms4 = ms2.clone()  # reuse 16-page case? need 64 pages for force-end; build new
    kv4 = 64 * page
    q4 = torch.randn(64, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp4 = torch.randn(64, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp4 = torch.randn_like(kp4)
    kvi4 = torch.arange(64, dtype=torch.int32, device=DEV)
    plan4 = F.fmha_sm100_plan(lens(64), lens(kv4), hq, hkv, page_size=page,
                              output_maxscore=True, causal=True)
    _, ms4 = F.fmha_sm100(q4, kp4, vp4, plan4, kv_indices=kvi4, output_maxscore=True)
    ms4 = ms4.contiguous()
    ms4[:, 0:2, :] = -1000.0  # suppress: never naturally selected
    ms4[:, 63, :] = -1000.0
    kbi4_plain = F.sparse_topk_select(ms4, topk, num_valid_pages=64)
    kbi4_forced = F.sparse_topk_select(ms4, topk, num_valid_pages=64,
                                       force_begin_blocks=2, force_end_blocks=1)
    print(f"  F4 plain row0:  {sorted(kbi4_plain[0,0].tolist())}")
    print(f"  F4 forced row0: {kbi4_forced[0,0].tolist()}")
    save("msa_sm100_golden_v3f_edges",
         dict(q1=q1, k1=kp1, v1=vp1, ms1=ms1, kbi1=kbi1, dense1=dense1, sparse1=sp1,
              q2=q2, k2=kp2, v2=vp2, ms2=ms2, kbi2=kbi2, dense2=dense2, sparse2=sp2,
              kbi3_dup=kbi3, sparse3_dup=sp3,
              q4=q4, k4=kp4, v4=vp4, ms4_suppressed=ms4,
              kbi4_plain=kbi4_plain, kbi4_forced=kbi4_forced),
         dict(capture="F edges", seed=50,
              f1="8 pages < topk16, kbi -1-padded, sparse consumes -1",
              f2="kv_len=2000 partial last page (15x128+80)",
              f3="kbi col15 duplicated from col0 -> sparse3_dup pins double-count-vs-dedup",
              f4="ms pages 0,1,63 suppressed to -1000 then force_begin=2/force_end=1",
              hq=8, hkv=1, page_size=128, topk=16))
    RESULTS["F"] = "ok"


# ---------------- G: prefill LSE + temperature LSE ----------------
def sec_g():
    torch.manual_seed(51)
    total_q, hq, hkv, hd, page, topk = 256, 8, 1, 128, 128, 16
    kv_len = 32 * page
    cu_q, cu_k = lens(0, total_q), lens(0, kv_len)
    q = torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    k = torch.randn(kv_len, hkv, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k)
    proxy_plan = F.fmha_sm100_plan(lens(total_q), lens(kv_len), hkv, hkv,
                                   page_size=page, output_maxscore=True, causal=True)
    _, ms = F.fmha_sm100(q[:, :hkv].contiguous(), k, v, proxy_plan,
                         output_maxscore=True, output_o=False)
    kbi = F.sparse_topk_select(ms.contiguous(), topk, num_valid_pages=kv_len // page)
    q2k = kbi.permute(1, 0, 2).contiguous()
    rp, qi = F.build_k2q_csr(q2k, cu_q, cu_k, page, total_k=kv_len,
                             max_seqlen_q=total_q, max_seqlen_k=kv_len)
    common = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q,
                  max_seqlen_k=kv_len, blk_kv=page, causal=True)
    out, lse, tlse = F.sparse_atten_func(
        q, k, v, rp, qi, topk, return_softmax_lse=True,
        lse_temperature_scale=4.0, return_temperature_lse=True, **common)
    out_s, lse_s = F.sparse_atten_func(
        q, k, v, rp, qi, topk, return_softmax_lse=True, softmax_scale=0.05, **common)
    print(f"  G lse {tuple(lse.shape)} tlse {tuple(tlse.shape)}")
    save("msa_sm100_golden_v3g_lse",
         dict(q=q, k=k, v=v, block_ids=kbi, row_ptr=rp, q_idx=qi,
              out=out, lse=lse, temperature_lse=tlse,
              out_sm005=out_s, lse_sm005=lse_s),
         dict(capture="G prefill LSE", seed=51, total_q=total_q, hq=hq, hkv=hkv,
              kv_len=kv_len, page_size=page, topk=topk, causal=True,
              lse_temperature_scale=4.0, sm_scale_variant=0.05))
    RESULTS["G"] = "ok"


# ---------------- H: fp4 indexer block scores ----------------
def sec_h():
    torch.manual_seed(52)
    total_q, heads, hd, page, n_pages = 64, 4, 128, 128, 8
    kv_len = n_pages * page
    qb = torch.randn(total_q, heads, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kb = torch.randn(n_pages, heads, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    q4, qs = quant_public_fp4(qb)
    k4, ks = quant_public_fp4(kb)
    cu_q, cu_k = lens(0, total_q), lens(0, kv_len)
    page_offsets = lens(0, n_pages)
    kvi = torch.arange(n_pages, dtype=torch.int32, device=DEV)
    outs = {}
    for tag, kw in (
        ("noncausal", dict(causal=False)),
        ("causal_tail", dict(causal=True,
                             qo_offset=torch.full((1,), kv_len - total_q,
                                                  dtype=torch.int32, device=DEV))),
    ):
        for layout in ("public", "preordered_mma"):
            try:
                s = fp4_indexer_block_scores(
                    q4, k4, qs, ks, cu_q, cu_k, page_offsets,
                    max_seqlen_q=total_q, max_seqlen_k=kv_len, kv_indices=kvi,
                    fp4_format="nvfp4", scale_layout=layout, **kw)
                outs[f"scores_{tag}_{layout}"] = s
                print(f"  H {tag}/{layout}: {tuple(s.shape)} ok")
            except Exception as e:  # noqa: BLE001
                print(f"  H {tag}/{layout} FAIL: {type(e).__name__}: {str(e)[:100]}")
    save("msa_sm100_golden_v3h_indexer",
         dict(q_bf16=qb, k_bf16=kb, q_fp4=q4, k_fp4=k4,
              q_scale=qs.view(torch.uint8), k_scale=ks.view(torch.uint8), **outs),
         dict(capture="H fp4 indexer", seed=52, total_q=total_q, heads=heads,
              head_dim=hd, page_size=page, n_pages=n_pages, fp4_format="nvfp4",
              note="public-layout per-16-block fp8e4m3 scales (block_amax/6, no global); scales saved as uint8 views"))
    RESULTS["H"] = "ok"


# ---------------- I: decode split-KV + determinism ----------------
def sec_i():
    torch.manual_seed(53)
    HQ, HKV, page, S = 64, 4, 128, 8
    KV = [131072, 2048, 3072, 8192]
    pp = [n // page for n in KV]
    pool = sum(pp) + 8
    kp = (torch.randn(pool, HKV, page, 128, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    vp = (torch.randn(pool, HKV, page, 128, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    perm = torch.randperm(pool, device=DEV).to(torch.int32)
    pt = torch.zeros(len(KV), max(pp), dtype=torch.int32, device=DEV)
    u = 0
    for b, n in enumerate(pp):
        pt[b, :n] = perm[u:u + n]
        u += n
    su = lens(*KV)
    q8 = (torch.randn(len(KV) * S, HQ, 128, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    common = dict(page_table=pt, seqused_k=su, seqlen_q=S, max_seqlen_k=max(KV),
                  blk_kv=page, causal=True, return_softmax_lse=True)
    o1, l1 = F.sparse_decode_atten_func(q8, kp, vp, None, **common)
    o2, l2 = F.sparse_decode_atten_func(q8, kp, vp, None, **common)
    det = bool(torch.equal(o1, o2)) and bool(torch.equal(l1, l2))
    print(f"  I longctx decode: nan={o1.isnan().any().item()} deterministic={det}")
    save("msa_sm100_golden_v3i_decode_longctx",
         dict(q_fp8=q8.view(torch.uint8), k_pages_fp8=kp.view(torch.uint8),
              v_pages_fp8=vp.view(torch.uint8), page_table=pt, seqused_k=su,
              dense_out=o1, dense_lse=l1),
         dict(capture="I decode split-kv 128K", seed=53, kv_lens=KV, hq=HQ, hkv=HKV,
              seqlen_q=S, page_size=page, deterministic_rerun=det,
              note="131072-token seq forces split-KV combine path; fp8 uint8 views"))
    RESULTS["I"] = f"ok det={det}"


# ---------------- J: fp8-storage prefill ----------------
def sec_j():
    torch.manual_seed(54)
    total_q, hq, hkv, hd, page, topk = 512, 8, 1, 128, 128, 16
    kv_len = 64 * page
    cu_q, cu_k = lens(0, total_q), lens(0, kv_len)
    q8 = (torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    k8 = (torch.randn(kv_len, hkv, hd, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    v8 = (torch.randn(kv_len, hkv, hd, device=DEV, dtype=torch.bfloat16) * 0.3).to(torch.float8_e4m3fn)
    # selection computed in bf16 from the SAME fp8 values (deterministic input)
    proxy_plan = F.fmha_sm100_plan(lens(total_q), lens(kv_len), hkv, hkv,
                                   page_size=page, output_maxscore=True, causal=True)
    _, ms = F.fmha_sm100(q8.to(torch.bfloat16)[:, :hkv].contiguous(),
                         k8.to(torch.bfloat16), v8.to(torch.bfloat16),
                         proxy_plan, output_maxscore=True, output_o=False)
    kbi = F.sparse_topk_select(ms.contiguous(), topk, num_valid_pages=kv_len // page)
    q2k = kbi.permute(1, 0, 2).contiguous()
    rp, qi = F.build_k2q_csr(q2k, cu_q, cu_k, page, total_k=kv_len,
                             max_seqlen_q=total_q, max_seqlen_k=kv_len)
    common = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q,
                  max_seqlen_k=kv_len, blk_kv=page, causal=True)
    out_fp8pv, lse8 = F.sparse_atten_func(q8, k8, v8, rp, qi, topk,
                                          return_softmax_lse=True, **common)
    extra = {}
    try:
        out_bf16pv = F.sparse_atten_func(q8, k8, v8, rp, qi, topk,
                                         pv_dtype=torch.bfloat16, **common)
        extra["out_bf16pv"] = out_bf16pv
        print("  J pv_dtype=bf16 staging: ok")
    except Exception as e:  # noqa: BLE001
        print(f"  J pv_dtype=bf16 FAIL: {type(e).__name__}: {str(e)[:100]}")
    # fp8 through the PLAN surface (dense + maxscore on fp8 cache)?
    try:
        plan = F.fmha_sm100_plan(lens(total_q), lens(kv_len), hq, hkv,
                                 page_size=page, output_maxscore=True, causal=True)
        d8, ms8 = F.fmha_sm100(q8, k8, v8, plan, output_maxscore=True)
        extra["dense_fp8_plan"] = d8
        extra["ms_fp8_plan"] = ms8
        print(f"  J plan-surface fp8 dense+maxscore: ok {tuple(d8.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  J plan-surface fp8 FAIL: {type(e).__name__}: {str(e)[:100]}")
    save("msa_sm100_golden_v3j_fp8prefill",
         dict(q_fp8=q8.view(torch.uint8), k_fp8=k8.view(torch.uint8),
              v_fp8=v8.view(torch.uint8), block_ids=kbi, row_ptr=rp, q_idx=qi,
              ms_bf16_proxy=ms, out_fp8pv=out_fp8pv, lse=lse8, **extra),
         dict(capture="J fp8 prefill CSR", seed=54, total_q=total_q, hq=hq, hkv=hkv,
              kv_len=kv_len, page_size=page, topk=topk, causal=True,
              note="fp8 storage; out_fp8pv = default (fp8 QK+PV); out_bf16pv = pv_dtype staging lever; fp8 uint8 views"))
    RESULTS["J"] = "ok"


# ---------------- K: big multi-CTA + determinism ----------------
def sec_k():
    torch.manual_seed(55)
    total_q, hq, hkv, hd, page, topk = 4096, 8, 1, 128, 128, 16
    n_pages = 256
    kv_len = n_pages * page
    q = torch.randn(total_q, hq, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    kp = torch.randn(n_pages, hkv, page, hd, device=DEV, dtype=torch.bfloat16) * 0.3
    vp = torch.randn_like(kp)
    kvi = torch.randperm(n_pages, device=DEV).to(torch.int32)
    plan = F.fmha_sm100_plan(lens(total_q), lens(kv_len), hq, hkv,
                             page_size=page, output_maxscore=True, causal=True)
    d1, ms1 = F.fmha_sm100(q, kp, vp, plan, kv_indices=kvi, output_maxscore=True)
    d2, ms2 = F.fmha_sm100(q, kp, vp, plan, kv_indices=kvi, output_maxscore=True)
    det = bool(torch.equal(d1, d2)) and bool(torch.equal(ms1, ms2))
    kbi = F.sparse_topk_select(ms1.contiguous(), topk, num_valid_pages=n_pages)
    splan = F.fmha_sm100_plan(lens(total_q), lens(kv_len), hq, hkv,
                              page_size=page, kv_block_num=topk, causal=True)
    s1, _ = F.fmha_sm100(q, kp, vp, splan, kv_indices=kvi,
                         kv_block_indexes=kbi[:, :hkv, :].contiguous())
    s2, _ = F.fmha_sm100(q, kp, vp, splan, kv_indices=kvi,
                         kv_block_indexes=kbi[:, :hkv, :].contiguous())
    det_s = bool(torch.equal(s1, s2))
    print(f"  K 4096q/256pages: dense det={det} sparse det={det_s}")
    save("msa_sm100_golden_v3k_bigshape",
         dict(q=q, k_pages=kp, v_pages=vp, kv_indices=kvi, dense_out=d1,
              max_score=ms1, kv_block_indexes=kbi, sparse_out=s1),
         dict(capture="K big multi-CTA", seed=55, total_q=total_q, hq=hq, hkv=hkv,
              kv_len=kv_len, n_pages=n_pages, page_size=page, topk=topk,
              causal=True, dense_deterministic=det, sparse_deterministic=det_s,
              note="shuffled kv_indices; exercises multi-CTA scheduling + 256-tile (2x128) maxscore axis"))
    RESULTS["K"] = f"ok det={det}/{det_s}"


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()}")
    ver = subprocess.run(["pip", "show", "fmha-sm100"], capture_output=True, text=True).stdout
    rev = subprocess.run(["git", "-C", "/workspace/MSA", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"MSA git rev: {rev}")
    for fn in (sec_f, sec_g, sec_h, sec_i, sec_j, sec_k):
        print(f"== {fn.__name__} ==")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            RESULTS[fn.__name__] = f"FAIL {type(e).__name__}: {e}"
            print(f"  SECTION FAIL: {type(e).__name__}: {e}")
    with open(f"{OUT}/v3_provenance.json", "w") as fh:
        json.dump(dict(msa_git_rev=rev, pip_show=ver, results=RESULTS), fh, indent=1)
    shutil.copy(__file__, f"{OUT}/capture_golden_v3.py")
    print("V3 RESULTS:", RESULTS)
