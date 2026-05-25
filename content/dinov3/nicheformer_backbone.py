"""
nicheformer_backbone.py
-----------------------
Wraps NicheformerModel so it satisfies the DINOv3 backbone interface.

DINOv3 (ssl_meta_arch.py) expects backbone.forward() to return:
    {
        "x_norm_clstoken"    : Tensor [B, D]       – used by DINOHead + DINO loss
        "x_norm_patchtokens" : Tensor [B, seq, D]  – used by iBOT head (stubbed here)
        "x_storage_tokens"   : Tensor [B, 0,   D]  – register tokens  (empty stub)
    }

NicheformerModel.get_embeddings() returns Tensor [B, D=512]
Run Nicheformer, stuff that [B,512] into x_norm_clstoken, stub the rest.

Architecture used (random weights, no checkpoint):
    nlayers        = 12
    dim_model      = 512
    nheads         = 16
    dim_feedforward= 1024
    context_length = 1500
    n_tokens       = 25000
    masking_p      = 0.15
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── reproduced here so this file is self-contained ──────────────────────────
# TO-DO lale : from hescape.models.gexp_models._nicheformer import NicheformerModel)

MASK_TOKEN = 0
CLS_TOKEN  = 2


def _complete_masking_inference(batch: dict, n_tokens: int) -> dict:
    """
    Minimal version of complete_masking with masking_p=0.0 (inference mode).
    Replaces padding zeros with token-id 1, builds the attention mask.
    No random masking — we want clean embeddings.
    """
    padding_token = 1
    indices = batch["X"].clone()

    indices = torch.where(indices == 0,
                          torch.tensor(padding_token, device=indices.device),
                          indices)
    batch["X"] = indices
    batch["masked_indices"] = indices
    batch["mask"] = torch.ones_like(indices)

    attention_mask = (indices == padding_token)
    batch["attention_mask"] = attention_mask.bool()
    return batch


class _NicheformerCore(nn.Module):
    """
    Standalone Nicheformer transformer with random weights.
    Mirrors the architecture in nicheformer/models/_nicheformer.py exactly.
    Does NOT inherit from pl.LightningModule so there is zero Lightning overhead.
    """

    def __init__(
        self,
        dim_model: int      = 512,
        nheads: int         = 16,
        dim_feedforward: int= 1024,
        nlayers: int        = 12,
        dropout: float      = 0.1,
        n_tokens: int       = 25000,
        context_length: int = 1500,
        aux_tokens: int     = 3,          # specie + assay + modality prepended
    ):
        super().__init__()
        self.dim_model      = dim_model
        self.context_length = context_length
        self.aux_tokens     = aux_tokens

        # Token + positional embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=n_tokens + 5,
            embedding_dim=dim_model,
            padding_idx=1,
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=dim_model,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Register positional indices as a buffer so they move with .to(device)
        self.register_buffer(
            "pos",
            torch.arange(0, context_length, dtype=torch.long),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            layer_norm_eps=1e-12,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=False,
        )

    def get_embeddings(
        self,
        batch: dict,
        with_context: bool = False,
    ) -> torch.Tensor:
        """
        batch must contain:
            "masked_indices"  : LongTensor [B, seq]
            "attention_mask"  : BoolTensor [B, seq]  True = padding (ignore)

        Returns: [B, dim_model]  — mean-pooled over non-auxiliary token positions.
        """
        x            = batch["masked_indices"]
        attn_mask    = batch["attention_mask"]

        # Token embedding + positional embedding
        tok_emb = self.embeddings(x)
        pos_emb = self.positional_embedding(self.pos[:x.size(1)])
        hidden  = self.dropout(tok_emb + pos_emb)

        # Full transformer pass
        hidden = self.encoder(
            hidden,
            src_key_padding_mask=attn_mask,
            is_causal=False,
        )

        if not with_context:
            hidden = hidden[:, self.aux_tokens:, :]

        # Mean-pool (ignore padding positions)
        # Build a mask: True = valid token (not padding)
        if not with_context:
            valid = ~attn_mask[:, self.aux_tokens:]
        else:
            valid = ~attn_mask

        valid_f = valid.float().unsqueeze(-1)
        summed  = (hidden * valid_f).sum(dim=1)
        count   = valid_f.sum(dim=1).clamp(min=1.0)
        emb     = summed / count

        return emb


# ── Token-ID constants (match Nicheformer's on_after_batch_transfer) ─────────
_MODALITY_SPATIAL = 4
_SPECIE_HUMAN     = 5
_ASSAY_XENIUM     = 9


class NicheformerBackbone(nn.Module):
    """
    DINOv3-compatible backbone wrapping the Nicheformer transformer.

    Input  (forward):
        x            : LongTensor [B, seq_len]  – gene token IDs (from NicheformerTransform)
        is_training  : bool                     – passed by SSLMetaArch (unused here)
        masks        : optional                 – iBOT masks (ignored in this stub)

    Output (forward) — the dict DINOv3 backbone interface requires:
        "x_norm_clstoken"    : [B, D]
        "x_norm_patchtokens" : [B, 1, D]   ← stub; iBOT loss disabled in train_gene.py
        "x_storage_tokens"   : [B, 0, D]   ← stub; no register tokens
    """

    # embed_dim is read by SSLMetaArch right after build_model_from_cfg returns
    num_features: int = 512

    def __init__(
        self,
        dim_model: int       = 512,
        nheads: int          = 16,
        dim_feedforward: int = 1024,
        nlayers: int         = 12,
        dropout: float       = 0.1,
        n_tokens: int        = 25000,
        context_length: int  = 1500,
    ):
        super().__init__()

        self.dim_model      = dim_model
        self.context_length = context_length
        self.num_features   = dim_model

        self.trunk = _NicheformerCore(
            dim_model=dim_model,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            nlayers=nlayers,
            dropout=dropout,
            n_tokens=n_tokens,
            context_length=context_length,
            aux_tokens=3,
        )

    # ── called once by SSLMetaArch.init_weights() ───────────────────────────
    def init_weights(self) -> None:
        """Xavier-init all Linear layers, zero all biases. Matches Nicheformer."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _prepend_aux_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend [modality | assay | specie] tokens to every sequence.
        Mirrors Nicheformer's on_after_batch_transfer logic.

        x : [B, seq]
        returns : [B, seq+3]
        """
        B = x.size(0)
        device = x.device

        modality = torch.full((B, 1), _MODALITY_SPATIAL, dtype=torch.long, device=device)
        assay    = torch.full((B, 1), _ASSAY_XENIUM,     dtype=torch.long, device=device)
        specie   = torch.full((B, 1), _SPECIE_HUMAN,     dtype=torch.long, device=device)


        return torch.cat([modality, assay, specie, x], dim=1)  # [B, seq+3]

    def _build_batch(self, x: torch.Tensor) -> dict:
        """
        Build the dict that _NicheformerCore.get_embeddings() expects.
        x : [B, seq]  (already has aux tokens prepended)
        """
        batch = {"X": x}
        return _complete_masking_inference(batch, n_tokens=self.trunk.embeddings.num_embeddings)

    # ── main forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        x,
        is_training: bool = True,
        masks=None,               # ignored — iBOT stub
    ) -> dict:
        """
        Accepts either:
          • a single tensor   [B, seq]          — teacher call
          • a list of tensors [[B,seq],[B,seq]]  — student call (global+local)

        Always returns the three-key dict DINOv3 expects.
        """
        if isinstance(x, (list, tuple)):
            outs = [self._forward_single(xi) for xi in x]
            return outs   # ssl_meta_arch unpacks this as global_out, local_out

        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> dict:
        """
        x : LongTensor [B, seq]
        Returns the three-key dict.
        """
        # 1. Clamp to context_length (safety — dummy data may exceed it)
        x = x[:, :self.context_length]
        # 2. Prepend auxiliary tokens → [B, seq+3]
        x = self._prepend_aux_tokens(x)
        x = x[:, :self.context_length]
        batch = self._build_batch(x)
        cls_emb = self.trunk.get_embeddings(batch, with_context=False)  # [B, 512]
        cls_emb = F.normalize(cls_emb, p=2, dim=-1)

        B, D = cls_emb.shape

        return {
            "x_norm_clstoken":    cls_emb,
            # iBOT stub — one fake patch token per sample
            "x_norm_patchtokens": cls_emb.unsqueeze(1),
            # Register tokens — Nicheformer has none
            "x_storage_tokens":   cls_emb.new_zeros(B, 0, D),
        }


# ── pretrained weight loader ─────────────────────────────────────────────────

# Architecture config from theislab/Nicheformer HuggingFace hub (config.json)
# n_tokens=20340 is the pretrained vocab size — MUST match to load weights
_PRETRAINED_CFG = dict(
    dim_model       = 512,
    nheads          = 16,
    dim_feedforward = 1024,
    nlayers         = 12,
    dropout         = 0.0,
    n_tokens        = 20340,   # pretrained vocab — different from random (25000)
    context_length  = 1500,
)

# Architecture config for random-weight training
_RANDOM_CFG = dict(
    dim_model       = 512,
    nheads          = 16,
    dim_feedforward = 1024,
    nlayers         = 12,
    dropout         = 0.1,
    n_tokens        = 25000,
    context_length  = 1500,
)


def _load_pretrained_weights(backbone: "NicheformerBackbone", checkpoint: str) -> None:
    """
    Load pretrained Nicheformer weights into a NicheformerBackbone.

    Supports two sources:
        HuggingFace hub  : checkpoint = "theislab/Nicheformer"
        Local safetensors: checkpoint = "/path/to/model.safetensors"
        Local .ckpt      : checkpoint = "/path/to/model.ckpt"

    The HuggingFace model uses a different key naming convention than our
    _NicheformerCore. We remap keys automatically:
        HF key                          → our key
        encoder.embeddings.*            → trunk.embeddings.*
        encoder.positional_embedding.*  → trunk.positional_embedding.*
        encoder.encoder.*               → trunk.encoder.*
    """
    import os

    # ── load raw state dict ───────────────────────────────────────────────────
    if not os.path.exists(checkpoint):
        # Treat as HuggingFace hub repo ID
        print(f"  Downloading weights from HuggingFace: {checkpoint}")
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=checkpoint,
                filename="model.safetensors",
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not download from HuggingFace hub '{checkpoint}'."
                f"Install huggingface_hub: pip install huggingface_hub{e}"
            )
        checkpoint = local_path

    ext = os.path.splitext(checkpoint)[1].lower()

    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
            raw_sd = load_file(checkpoint, device="cpu")
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors files."
                "pip install safetensors"
            )
    elif ext in (".ckpt", ".pt", ".pth"):
        raw_sd = torch.load(checkpoint, map_location="cpu")
        # Lightning checkpoints wrap state_dict under a key
        if "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]
    else:
        raise ValueError(f"Unsupported checkpoint format: {ext}")

    # ── remap HuggingFace key names → our _NicheformerCore key names ──────────
    # We map these to our _NicheformerCore which lives under "trunk.*"
    KEY_MAP = [
        ("nicheformer.embeddings.",          "trunk.embeddings."),
        ("nicheformer.positional_embedding.","trunk.positional_embedding."),
        ("nicheformer.encoder.",             "trunk.encoder."),
        ("nicheformer.dropout.",             "trunk.dropout."),
        ("nicheformer.pos",                  "trunk.pos"),
        # Fallback: plain Nicheformer state dict (no wrapper prefix)
        ("encoder.embeddings.",              "trunk.embeddings."),
        ("encoder.positional_embedding.",    "trunk.positional_embedding."),
        ("encoder.encoder.",                 "trunk.encoder."),
        ("encoder.dropout.",                 "trunk.dropout."),
    ]

    remapped = {}
    skipped  = []
    for k, v in raw_sd.items():
        new_k = k
        for hf_prefix, our_prefix in KEY_MAP:
            if k.startswith(hf_prefix):
                new_k = our_prefix + k[len(hf_prefix):]
                break
        if any(skip in new_k for skip in [
            "classifier_head", "pooler_head", "cls_head",
            "lm_head", "cls_loss", "activation",
            "nicheformer.classifier", "nicheformer.pooler",
            "encoder_layer.",
        ]):
            skipped.append(new_k)
            continue
        remapped[new_k] = v

    # ── load into backbone ────────────────────────────────────────────────────
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    missing   = [k for k in missing   if "pos" not in k]  # pos buffer rebuilt
    unexpected = [k for k in unexpected if k not in skipped]

    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    n_loaded = len(remapped) - len(unexpected)
    print(f"  Loaded {n_loaded} weight tensors from checkpoint.")
    print(f"  Skipped {len(skipped)} head layers (not needed for DINO backbone).")

    # Sanity check: verify the embedding weights are non-random
    # (if loading failed silently, embeddings would have xavier init ~0)
    emb_key = "trunk.embeddings.weight"
    if emb_key in remapped:
        emb_std = remapped[emb_key].std().item()
        print(f"  Embedding weight std = {emb_std:.6f}  (random xavier ~0.04, pretrained ~0.02-0.09)")


# ── factory function called by train_gene.py ─────────────────────────────────

def build_nicheformer_backbone(checkpoint: str = "none") -> tuple:
    """
    Build student and teacher NicheformerBackbone.

    Args:
        checkpoint : one of:
            "none"                   → random weights (default, no files needed)
            "theislab/Nicheformer"   → download pretrained from HuggingFace hub
            "/path/to/model.safetensors" → load from local safetensors file
            "/path/to/model.ckpt"    → load from local Lightning checkpoint

    Returns:
        (student_backbone, teacher_backbone, embed_dim)
        — same signature as dinov3.models.build_model_from_cfg

    Notes:
        - Pretrained weights use n_tokens=20340 (from config.json on HF hub)
        - Random weights use n_tokens=25000
        - embed_dim=512 in both cases (unchanged)
        - dropout=0.0 for pretrained (matches original training), 0.1 for random
    """
    use_pretrained = checkpoint.lower() != "none"
    cfg = _PRETRAINED_CFG if use_pretrained else _RANDOM_CFG

    print(f"  Building backbone — mode: {'pretrained' if use_pretrained else 'random weights'}")
    print(f"  n_tokens={cfg['n_tokens']}  dropout={cfg['dropout']}")

    student = NicheformerBackbone(**cfg)
    teacher = NicheformerBackbone(**cfg)

    if use_pretrained:
        print(f"  Loading pretrained weights: {checkpoint}")
        _load_pretrained_weights(student, checkpoint)
        # Teacher starts as exact copy of pretrained student
        teacher.load_state_dict(student.state_dict())
        print("  Teacher initialised from pretrained student weights.")
    else:
        print("  Using random weight initialisation.")
        # Teacher starts as exact copy of student even for random weights
        # (EMA update requires identical starting point)
        teacher.load_state_dict(student.state_dict())

    embed_dim = 512
    return student, teacher, embed_dim


# ── sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default="none", type=str,
        help=(
            "Weights to load. Options:\n"
            "  'none'                      → random weights (default)\n"
            "  'theislab/Nicheformer'      → download from HuggingFace\n"
            "  '/path/to/model.safetensors → local file"
        )
    )
    args = p.parse_args()

    print("=" * 60)
    print(f"  nicheformer_backbone.py sanity check")
    print(f"  checkpoint = {args.checkpoint}")
    print("=" * 60)

    student, teacher, embed_dim = build_nicheformer_backbone(args.checkpoint)
    print(f"  embed_dim = {embed_dim}")

    # Use n_tokens matching the built backbone
    vocab = student.trunk.embeddings.num_embeddings
    B, SEQ = 3, 1500
    fake_tokens = torch.randint(0, vocab - 5, (B, SEQ), dtype=torch.long)
    mask = torch.rand(B, SEQ) > 0.8
    fake_tokens = fake_tokens * mask.long()

    print(f"  Input shape : {fake_tokens.shape}  dtype={fake_tokens.dtype}")

    with torch.no_grad():
        out = student(fake_tokens, is_training=True)

    print(f"  x_norm_clstoken    : {out['x_norm_clstoken'].shape}")
    print(f"  x_norm_patchtokens : {out['x_norm_patchtokens'].shape}")
    print(f"  x_storage_tokens   : {out['x_storage_tokens'].shape}")

    norms = out["x_norm_clstoken"].norm(dim=-1)
    print(f"  CLS norms (should be ~1.0): {norms}")

    # Verify student and teacher start with identical weights
    for (n_s, p_s), (n_t, p_t) in zip(
        student.named_parameters(), teacher.named_parameters()
    ):
        assert torch.equal(p_s, p_t), f"Weight mismatch: {n_s}"
    print("  Student == Teacher weights at init: OK")
    print("  Backbone sanity check PASSED.")
    print()
    print("  To test pretrained weights:")
    print("    python nicheformer_backbone.py --checkpoint theislab/Nicheformer")
