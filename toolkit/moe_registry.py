"""ModelOpt quantization support for MoE models with fused 3D expert weights.

Supports:
  - MiniMaxM2 (built-in transformers): MiniMaxM2Experts
  - GLM-5 / glm_moe_dsa (remote code): GlmMoeDsaMoE + GlmMoeDsaNaiveMoe
  - Qwen3.5 MoE: Qwen3_5MoeSparseMoeBlock + Qwen3_5MoeExperts

All models use natural top-k routing during calibration. Expert coverage is
achieved through a large, diverse calibration dataset rather than forcing
all-expert activation, which biases downstream scales.
"""
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
try:
    from modelopt.torch.quantization.plugins.huggingface import _QuantSparseMoe
except ImportError:
    from modelopt.torch.quantization.plugins.huggingface import _QuantSparseSequentialMoe as _QuantSparseMoe  # 0.44 rename; only GLM5/Qwen regs use it


def patch_glm5_attention_indexer():
    """Patch GLM-5 attention indexer layout to match checkpoint keys/shapes."""
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention
    except ImportError:
        print("⚠ glm_moe_dsa attention class not found, skipping indexer patch")
        return

    if getattr(GlmMoeDsaAttention, "_glm5_indexer_patch_applied", False):
        return

    original_init = GlmMoeDsaAttention.__init__

    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)

        if hasattr(self, "indexer"):
            return

        index_n_heads = getattr(config, "index_n_heads", self.num_heads)
        index_head_dim = getattr(config, "index_head_dim", self.qk_head_dim)

        for name in ("wq_b", "wk", "k_norm", "weights_proj"):
            if hasattr(self, name):
                delattr(self, name)

        self.indexer = nn.Module()
        self.indexer.wq_b = nn.Linear(config.q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.indexer.wk = nn.Linear(config.hidden_size, index_head_dim, bias=config.attention_bias)
        self.indexer.k_norm = nn.LayerNorm(index_head_dim)
        self.indexer.weights_proj = nn.Linear(config.hidden_size, index_n_heads, bias=False)

    GlmMoeDsaAttention.__init__ = patched_init
    GlmMoeDsaAttention._glm5_indexer_patch_applied = True
    print("✓ Patched GLM-5 attention indexer layout")


# ---------------------------------------------------------------------------
# Shared: unfuse 3D expert params into per-expert nn.Linear.
# ---------------------------------------------------------------------------

class _QuantFusedExperts(QuantModule):
    """Unfuse 3D expert params into per-expert nn.Linear for quantization."""

    def _setup(self):
        from accelerate import init_empty_weights

        dtype, device = self.gate_up_proj.dtype, self.gate_up_proj.device
        I = self.intermediate_dim
        H = self.hidden_dim

        def _copy_weight(module, weight):
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().data.to(dtype=dtype, device=device)

        with init_empty_weights():
            gate_proj = nn.ModuleList(
                [nn.Linear(H, I, bias=False) for _ in range(self.num_experts)]
            )
            up_proj = nn.ModuleList(
                [nn.Linear(H, I, bias=False) for _ in range(self.num_experts)]
            )
            down_proj = nn.ModuleList(
                [nn.Linear(I, H, bias=False) for _ in range(self.num_experts)]
            )

        for idx in range(self.num_experts):
            _copy_weight(gate_proj[idx], self.gate_up_proj[idx, :I, :])
            _copy_weight(up_proj[idx], self.gate_up_proj[idx, I:, :])
            _copy_weight(down_proj[idx], self.down_proj[idx])

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


# ---------------------------------------------------------------------------
# GLM-5 (glm_moe_dsa).
# ---------------------------------------------------------------------------

class _QuantGlmMoeDsaMoE(_QuantSparseMoe):
    @property
    def num_experts(self):
        return self.n_routed_experts

    def forward(self, hidden_states):
        return super(_QuantSparseMoe, self).forward(hidden_states)


def register_glm5_moe_for_quantization():
    """Register GLM-5 (glm_moe_dsa) MoE classes with modelopt."""
    patch_glm5_attention_indexer()
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaMoE,
            GlmMoeDsaNaiveMoe,
        )
    except ImportError:
        print("⚠ glm_moe_dsa not in transformers, skipping GLM-5 registration")
        return

    if QuantModuleRegistry.get(GlmMoeDsaMoE) is None:
        QuantModuleRegistry.register(
            {GlmMoeDsaMoE: "GlmMoeDsaMoE"}
        )(_QuantGlmMoeDsaMoE)

    if QuantModuleRegistry.get(GlmMoeDsaNaiveMoe) is None:
        QuantModuleRegistry.register(
            {GlmMoeDsaNaiveMoe: "GlmMoeDsaNaiveMoe"}
        )(_QuantFusedExperts)

    print("✓ Registered GLM-5 MoE for quantization")


# ---------------------------------------------------------------------------
# Qwen3.5 MoE.
# ---------------------------------------------------------------------------

class _QuantQwen35MoeSparseMoeBlock(_QuantSparseMoe):
    @property
    def num_experts(self):
        return self.experts.num_experts

    def forward(self, hidden_states):
        return super(_QuantSparseMoe, self).forward(hidden_states)


def register_qwen35_moe_for_quantization():
    """Register Qwen3.5 MoE (qwen3_5_moe) classes with modelopt."""
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeSparseMoeBlock,
            Qwen3_5MoeExperts,
        )
    except ImportError:
        print("⚠ qwen3_5_moe not in transformers, skipping Qwen3.5 registration")
        return

    if QuantModuleRegistry.get(Qwen3_5MoeSparseMoeBlock) is None:
        QuantModuleRegistry.register(
            {Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"}
        )(_QuantQwen35MoeSparseMoeBlock)

    if QuantModuleRegistry.get(Qwen3_5MoeExperts) is None:
        QuantModuleRegistry.register(
            {Qwen3_5MoeExperts: "Qwen3_5MoeExperts"}
        )(_QuantFusedExperts)

    print("✓ Registered Qwen3.5 MoE for quantization")


# ---------------------------------------------------------------------------
# MiniMax M3 (minimax_m3_vl, native transformers).
# Experts: fused 3D gate_up_proj/down_proj, CONTIGUOUS halves (the modeling
# code says "same as GPT OSS, but the weights are not interleaved"), clamped
# swiglu: gate.clamp(max=L); up.clamp(-L, L); (up+1) * gate*sigmoid(gate*a).
# ---------------------------------------------------------------------------

class _QuantM3FusedExperts(_QuantFusedExperts):
    """M3 experts: unfused per-expert Linears + M3's clamped swiglu."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # 2026-06-12 vectorized routing: argsort-by-expert + contiguous
        # slices replaces the one_hot[T,k,E] + per-expert where-scan
        # (verified bitwise-identical outputs AND identical per-expert
        # token sets -> identical amax; 2x+ on the token-scaling term).
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            Tn, Kn = top_k_index.shape
            flat_e = top_k_index.reshape(-1)
            flat_t = torch.arange(Tn, device=hidden_states.device).repeat_interleave(Kn)
            order = torch.argsort(flat_e, stable=True)
            se = flat_e[order]
            st = flat_t[order]
            counts = torch.bincount(se, minlength=self.num_experts + 1).cpu().tolist()
        sw = top_k_weights.reshape(-1)[order]
        start = 0
        for e in range(self.num_experts):
            n = counts[e]
            if n == 0:
                continue
            sl = slice(start, start + n)
            start += n
            tok = st[sl]
            x = hidden_states[tok]
            gate = self.gate_proj[e](x)
            up = self.up_proj[e](x)
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
            cur = (up + 1.0) * glu
            cur = self.down_proj[e](cur) * sw[sl, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def register_minimax_m3_moe_for_quantization():
    """Register MiniMax-M3 VL experts with modelopt."""
    try:
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
            MiniMaxM3VLExperts,
        )
    except ImportError:
        print("\u26a0 minimax_m3_vl not in transformers, skipping M3 registration")
        return

    if QuantModuleRegistry.get(MiniMaxM3VLExperts) is None:
        QuantModuleRegistry.register(
            {MiniMaxM3VLExperts: "MiniMaxM3VLExperts"}
        )(_QuantM3FusedExperts)

    print("\u2713 Registered MiniMax-M3 MoE for quantization")
