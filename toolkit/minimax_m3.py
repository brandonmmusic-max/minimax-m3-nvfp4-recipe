from .base import ModelQuantConfig


MinimaxM3Config = ModelQuantConfig(
    model_id="/workspace/MiniMax-M3",
    trust_remote_code=False,  # native in transformers main (minimax_m3_vl)
    streaming=True,           # Luke topology: GPU0 executes, GPUs 1-7 store layers
    extra_quant_overrides={
        "*mlp.gate.weight_quantizer": {"enable": False},
        "*mlp.gate.input_quantizer": {"enable": False},
        "*multi_modal_projector*quantizer": {"enable": False},
        "*gate_up_proj*quantizer": {"enable": False},
        "*patch_merge*quantizer": {"enable": False},
        "*router*": {"enable": False},
        "*lm_head*weight_quantizer": {"enable": False},
        "*lm_head*input_quantizer": {"enable": False},
        "*embed*": {"enable": False},
        "*vision*": {"enable": False},
        "*visual*": {"enable": False},
        "*mtp*": {"enable": False},
    },
)


def _register_moe():
    from moe_registry import register_minimax_m3_moe_for_quantization
    register_minimax_m3_moe_for_quantization()


def _get_model_cls():
    # streaming loader requires a concrete class (reads .config_class)
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration,
    )
    return MiniMaxM3SparseForConditionalGeneration


MinimaxM3Config.register_moe = _register_moe
MinimaxM3Config.get_model_cls = _get_model_cls


import re as _re


def _m3_key_translate(k):
    """Checkpoint layout (repo) -> native transformers module paths.

    Mirrors transformers/conversion_mapping.py (the canonical table) plus
    w1/w3/w2 -> gate_proj/up_proj/down_proj so the loader's existing
    3D expert fuser (gate-first concat, matching Concatenate(dim=1)) fires.
    """
    # language tower prefix + lm_head
    k = _re.sub(r"^language_model\.model\.", "model.language_model.", k)
    k = _re.sub(r"^language_model\.lm_head", "lm_head", k)
    # vision tower
    k = _re.sub(r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.", "model.vision_tower.embeddings.proj.", k)
    k = _re.sub(r"^vision_tower\.vision_model\.encoder\.layers\.", "model.vision_tower.layers.", k)
    k = _re.sub(r"^vision_tower\.vision_model\.", "model.vision_tower.", k)
    # projector + patch merge mlp
    k = _re.sub(r"^multi_modal_projector\.", "model.multi_modal_projector.", k)
    k = _re.sub(r"^patch_merge_mlp\.linear_1\.", "model.multi_modal_projector.merge_linear_1.", k)
    k = _re.sub(r"^patch_merge_mlp\.linear_2\.", "model.multi_modal_projector.merge_linear_2.", k)
    # MoE block per-leaf renames
    k = _re.sub(r"\.block_sparse_moe\.experts\.", ".mlp.experts.", k)
    k = _re.sub(r"\.block_sparse_moe\.shared_experts\.", ".mlp.shared_experts.", k)
    k = _re.sub(r"\.block_sparse_moe\.gate\.weight", ".mlp.gate.weight", k)
    k = _re.sub(r"\.block_sparse_moe\.e_score_correction_bias", ".mlp.gate.e_score_correction_bias", k)
    # sparse attention indexer
    k = _re.sub(r"\.self_attn\.index_q_proj\.", ".self_attn.indexer.q_proj.", k)
    k = _re.sub(r"\.self_attn\.index_k_proj\.", ".self_attn.indexer.k_proj.", k)
    k = _re.sub(r"\.self_attn\.index_q_norm\.", ".self_attn.indexer.q_norm.", k)
    k = _re.sub(r"\.self_attn\.index_k_norm\.", ".self_attn.indexer.k_norm.", k)
    # expert weight names: w1=gate, w3=up, w2=down (canonical source order)
    k = _re.sub(r"(\.mlp\.experts\.\d+)\.w1\.weight$", r"\1.gate_proj.weight", k)
    k = _re.sub(r"(\.mlp\.experts\.\d+)\.w3\.weight$", r"\1.up_proj.weight", k)
    k = _re.sub(r"(\.mlp\.experts\.\d+)\.w2\.weight$", r"\1.down_proj.weight", k)
    return k


MinimaxM3Config.key_translate = staticmethod(_m3_key_translate)
