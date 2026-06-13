# SPDX-License-Identifier: MIT
"""SM100 golden capture E — paged sparse DECODE (the production-critical shape).

SM100 ABI: decode supports ONLY torch.float8_e4m3fn Q/K/V
(interface.py:_validate_sparse_decode_inputs) — the whole decode path is
fp8.  Variants: dense decode (q2k=None) and sparse decode (topk=16);
seqlen_q=1 (pure decode) and seqlen_q=4 (speculative/MTP shape).  All with
return_softmax_lse=True to pin the LSE convention.
q2k_indices: int32 [Hkv, total_q, topK], page ids LOCAL to each sequence.
"""
from __future__ import annotations

import json
import shutil

import torch
from safetensors.torch import save_file

import fmha_sm100 as F

OUT = "/workspace/msa_golden"
DEV = torch.device("cuda")
# MiniMax-M3 production shape: 64 q heads / 4 kv heads = qhead_per_kv 16
# (the ONLY ratio the SM100 decode kernel supports), D=128, fp8e4m3 only.
PAGE, TOPK, HQ, HKV, HD = 128, 16, 64, 4, 128
KV_LENS = [2048, 3072, 4096, 8192]  # 16/24/32/64 pages, all >= topk
N_POOL = 160


def build_case(seqlen_q: int, seed: int):
    torch.manual_seed(seed)
    batch = len(KV_LENS)
    pages_per = [n // PAGE for n in KV_LENS]
    max_pages = max(pages_per)
    kp = torch.randn(N_POOL, HKV, PAGE, HD, device=DEV, dtype=torch.bfloat16) * 0.3
    vp = torch.randn_like(kp)
    perm = torch.randperm(N_POOL, device=DEV).to(torch.int32)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=DEV)
    used = 0
    for b, np_ in enumerate(pages_per):
        page_table[b, :np_] = perm[used : used + np_]
        used += np_
    seqused = torch.tensor(KV_LENS, dtype=torch.int32, device=DEV)
    q = torch.randn(batch * seqlen_q, HQ, HD, device=DEV, dtype=torch.bfloat16) * 0.3

    # realistic top-16 LOCAL page selection per token PER KV HEAD: raw q.k
    # tile maxima (mean over the kv head's contiguous q-head group),
    # causal-limited.  Heads group contiguously: q heads [h*16, (h+1)*16) -> kv h.
    grp = HQ // HKV
    q2k = torch.zeros(HKV, batch * seqlen_q, TOPK, dtype=torch.int32, device=DEV)
    for b in range(batch):
        np_ = pages_per[b]
        for h in range(HKV):
            keys = kp[page_table[b, :np_].long(), h].reshape(-1, HD).float()
            for s in range(seqlen_q):
                t = b * seqlen_q + s
                qpos = KV_LENS[b] - seqlen_q + s
                logits = keys @ q[t, h * grp : (h + 1) * grp].float().T  # [kv_len, grp]
                logits[qpos + 1 :, :] = float("-inf")
                tile_scores = logits.reshape(np_, PAGE, grp).amax(1).mean(-1)
                q2k[h, t] = torch.topk(tile_scores, TOPK).indices.to(torch.int32)
    return q, kp, vp, page_table, seqused, q2k


def run_case(tag: str, seqlen_q: int, seed: int):
    q, kp, vp, pt, su, q2k = build_case(seqlen_q, seed)
    q8 = q.to(torch.float8_e4m3fn)
    k8 = kp.to(torch.float8_e4m3fn)
    v8 = vp.to(torch.float8_e4m3fn)
    common = dict(page_table=pt, seqused_k=su, seqlen_q=seqlen_q,
                  max_seqlen_k=max(KV_LENS), blk_kv=PAGE, causal=True,
                  return_softmax_lse=True)
    dense_out, dense_lse = F.sparse_decode_atten_func(q8, k8, v8, None, **common)
    # SPARSE decode (q2k gather) is a STUB on SM100 in the released repo:
    # interface.py raises NotImplementedError("SM100 paged fp8 sparse decode
    # forward is not implemented yet").  q2k_indices is still saved as the
    # documented INPUT spec for the future sparse kernel.
    try:
        F.sparse_decode_atten_func(q8, k8, v8, q2k, **common)
        sparse_stub = False
    except NotImplementedError as exc:
        sparse_stub = True
        print(f"  sparse decode stub confirmed: {exc}")
    print(f"  {tag}: dense nan={dense_out.isnan().any().item()} "
          f"out dtype={dense_out.dtype} lse shape={tuple(dense_lse.shape)}")
    tensors = dict(
        q_fp8=q8.view(torch.uint8), k_pages_fp8=k8.view(torch.uint8),
        v_pages_fp8=v8.view(torch.uint8), q_bf16=q, k_pages_bf16=kp,
        v_pages_bf16=vp, page_table=pt, seqused_k=su, q2k_indices=q2k,
        dense_out=dense_out, dense_lse=dense_lse,
    )
    tensors = {k: v.contiguous().cpu() for k, v in tensors.items()}
    name = f"msa_sm100_golden_v2e_decode_s{seqlen_q}"
    save_file(tensors, f"{OUT}/{name}.safetensors")
    meta = dict(
        capture=f"E decode {tag}", seed=seed, batch=len(KV_LENS), kv_lens=KV_LENS,
        n_pool_pages=N_POOL, hq=HQ, hkv=HKV, head_dim=HD, page_size=PAGE,
        topk=TOPK, seqlen_q=seqlen_q, causal=True,
        sparse_decode_is_stub=True,
        note="SM100 decode ABI: fp8e4m3-ONLY q/k/v, qhead_per_kv MUST be 16 (M3 production 64q/4kv), D=128, causal only, cta_tile_q=128 => seqlen_q=8; SPARSE decode (q2k gather) is NotImplementedError on SM100 in the released repo — dense only here; *_fp8 tensors are uint8 views of float8_e4m3fn (bf16 originals included for reference); q2k_indices LOCAL page ids [hkv, total_q, topk] saved as the input spec for the future sparse kernel; LSE [total_q, hq]",
        tensors={k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()},
    )
    with open(f"{OUT}/{name}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  saved {name}")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()}")
    run_case("seqlen_q=8", 8, 46)
    shutil.copy(__file__, f"{OUT}/capture_golden_v2e.py")
    print("DECODE CAPTURES DONE")
