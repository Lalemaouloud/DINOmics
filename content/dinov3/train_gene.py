"""
train_gene.py
-------------
Minimal standalone training script that wires:
    DummyGeneCropDataset → gene_collate_fn → NicheformerBackbone → SSLMetaArch

No distributed training. No FSDP. No config files.
Goal: verify the full forward pass runs end-to-end on CPU or single GPU.

Run:
    python train_gene.py
    python train_gene.py --device cuda   # if GPU available
    python train_gene.py --steps 5       # run only 5 iterations
"""

import argparse
import math
import sys
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── our files ────────────────────────────────────────────────────────────────
from nicheformer_backbone import NicheformerBackbone, build_nicheformer_backbone
from gene_collate import gene_collate_fn
from gene_dataset_1 import make_gene_dataset

# ── DINOv3 components — import from wherever dinov3 is on PYTHONPATH ────
# If running from the dinov3 repo root:  export PYTHONPATH=.
from dinov3.layers.dino_head import DINOHead
from dinov3.loss.dino_clstoken_loss import DINOLoss



# ─────────────────────────────────────────────────────────────────────────────
# Minimal SSLMetaArch replacement
# We don't use the real SSLMetaArch because it requires FSDP + distributed (To be done later).
# This class implements the same teacher-student logic cleanly.
# ─────────────────────────────────────────────────────────────────────────────

class GeneDINO(nn.Module):
    """
    Teacher-student DINO training for gene expression data.

    Student: NicheformerBackbone + DINOHead (trainable)
    Teacher: NicheformerBackbone + DINOHead (EMA, no gradient)

    Loss: DINO CLS-token loss only (iBOT disabled for the prototype).
    """

    def __init__(
        self,
        embed_dim:        int   = 512,
        dino_out_dim:     int   = 4096,    # number of prototypes (K)
        head_hidden_dim:  int   = 2048,
        head_bottleneck:  int   = 256,
        head_nlayers:     int   = 3,
        n_local_crops:    int   = 8,
        n_global_crops:   int   = 2,
        checkpoint:       str   = "none",  # "none" | "theislab/Nicheformer" | local path
    ):
        super().__init__()

        self.n_local_crops  = n_local_crops
        self.n_global_crops = n_global_crops

        # ── Student ──────────────────────────────────────────────────────────
        s_backbone, t_backbone, _ = build_nicheformer_backbone(checkpoint)

        self.student_backbone = s_backbone
        self.student_head     = DINOHead(
            in_dim=embed_dim,
            out_dim=dino_out_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck,
            nlayers=head_nlayers,
        )

        # ── Teacher (EMA, no grad) ───────────────────────────────────────────
        self.teacher_backbone = t_backbone
        self.teacher_head     = DINOHead(
            in_dim=embed_dim,
            out_dim=dino_out_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck,
            nlayers=head_nlayers,
        )
        self.teacher_backbone.requires_grad_(False)
        self.teacher_head.requires_grad_(False)

        # ── DINO loss ────────────────────────────────────────────────────────
        self.dino_loss = DINOLoss(out_dim=dino_out_dim)

        # ── Init weights ─────────────────────────────────────────────────────
        self.student_backbone.init_weights()
        self.student_head.init_weights()
        # Teacher starts as a copy of the student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.dino_loss.init_weights()

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        """EMA update: teacher = m*teacher + (1-m)*student"""
        for s_p, t_p in zip(self.student_backbone.parameters(),
                             self.teacher_backbone.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)
        for s_p, t_p in zip(self.student_head.parameters(),
                             self.teacher_head.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

    def forward(self, batch: dict, teacher_temp: float = 0.04) -> torch.Tensor:
        """
        batch:
            "collated_global_crops" : [n_global*B, seq_g]
            "collated_local_crops"  : [n_local*B,  seq_l]
            "global_batch_size"     : int B

        Returns: scalar loss
        """
        B            = batch["global_batch_size"]
        # Read actual crop counts from the batch — works for both
        # single-transform (n_global=2, n_local=8) and
        # multi-transform  (n_global=8, n_local=32 for 4 transforms)
        n_global     = batch.get("n_global_crops", self.n_global_crops)
        n_local      = batch.get("n_local_crops",  self.n_local_crops)

        global_crops = batch["collated_global_crops"]   # [n_global*B, seq_g]
        local_crops  = batch["collated_local_crops"]    # [n_local*B,  seq_l]

        # ── Teacher forward (global crops only, no grad) ─────────────────────
        with torch.no_grad():
            t_out   = self.teacher_backbone(global_crops, is_training=True)
            t_cls   = t_out["x_norm_clstoken"]                  # [n_global*B, D]
            t_logit = self.teacher_head(t_cls)                  # [n_global*B, K]
            t_probs = self.dino_loss.sinkhorn_knopp_teacher(
                t_logit, teacher_temp=teacher_temp
            )                                                   # [n_global*B, K]
            t_probs = t_probs.unflatten(0, (n_global, B))       # [n_global, B, K]

        # ── Student forward (global + local crops) ───────────────────────────
        # Global and local have DIFFERENT sequence lengths (1500 vs 500),
        # so they must be processed in separate backbone calls — never cat'd.
        s_global_out    = self.student_backbone(global_crops, is_training=True)
        s_global_cls    = s_global_out["x_norm_clstoken"]               # [n_g*B, D]
        s_global_logit  = self.student_head(s_global_cls)               # [n_g*B, K]
        s_global_logit  = s_global_logit.unflatten(0, (n_global, B))    # [n_g, B, K]

        s_local_out     = self.student_backbone(local_crops, is_training=True)
        s_local_cls     = s_local_out["x_norm_clstoken"]                # [n_l*B, D]
        s_local_logit   = self.student_head(s_local_cls)                # [n_l*B, K]
        s_local_logit   = s_local_logit.unflatten(0, (n_local, B))      # [n_l, B, K]

        # Combine → [n_g+n_l, B, K]
        s_all_logit = torch.cat([s_global_logit, s_local_logit], dim=0)

        # ── DINO loss ─────────────────────────────────────────────────────────
        loss = self.dino_loss(
            student_logits=s_all_logit,   # [n_g+n_l, B, K]
            teacher_probs=t_probs,        # [n_global, B, K]
        )

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  Gene-DINO prototype training")
    transforms_display = args.transform if len(args.transform) > 1 else args.transform[0]
    print(f"  device       : {device}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  steps        : {args.steps}")
    print(f"  embed_dim    : 512")
    print(f"  prototypes K : {args.dino_out_dim}")
    print(f"  transform(s) : {transforms_display}")
    n_t = len(args.transform)
    print(f"  total views  : {n_t} transform(s) × 10 crops = {n_t*10} ({n_t*2}g + {n_t*8}l)")
    print(f"{'='*60}\n")

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    # Synthetic raw counts: N cells x 2000 genes, sparse (simulates real data)
    N_samples = args.batch_size * args.steps
    raw_counts = torch.zeros(N_samples, 2000)
    for i in range(N_samples):
        idx = torch.randperm(2000)[:400]
        raw_counts[i, idx] = torch.randint(1, 500, (400,)).float()

    # Single or multiple transforms — both work identically
    # Single:  --transform lognorm           → 2 global  + 8  local = 10 views
    # Multi:   --transform lognorm normalize → 4 global  + 16 local = 20 views
    # Full:    --transform lognorm normalize nicheformer scfoundation → 8 global  + 32 local = 40 views
    transforms = args.transform  # list from nargs='+'
    if len(transforms) == 1:
        transforms = transforms[0]   # single string for backward compat

    dataset = make_gene_dataset(
        raw_counts,
        transform_name   = transforms,
        n_global_crops   = 2,          # per transform # TO-DO-lale : Check with Rushin later if I change of keep 2 for global and 8 for local or if I put 4 global and 16 local when I use 2 transforms? 
        n_local_crops    = 8,          # per transform
        global_crop_size = 1500,
        local_crop_size  = 500,
        vocab_size       = 25000,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=gene_collate_fn,  # auto-detects n_global/local from batch shape
        drop_last=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = GeneDINO(
        embed_dim=512,
        dino_out_dim=args.dino_out_dim,
        head_hidden_dim=2048,
        head_bottleneck=256,
        head_nlayers=3,
        n_local_crops=8,
        n_global_crops=2,
        checkpoint=args.checkpoint,
    ).to(device)

    n_params = sum(p.numel() for p in model.student_backbone.parameters())
    print(f"Student backbone parameters : {n_params:,}")
    print(f"Student head parameters     : "
          f"{sum(p.numel() for p in model.student_head.parameters()):,}")
    print()

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.student_backbone.parameters()) +
        list(model.student_head.parameters()),
        lr=1e-4,
        weight_decay=0.04,
    )

    # ── Teacher temp schedule: warm up from 0.02 → 0.04 over first half ─────
    def teacher_temp_schedule(step: int, total: int) -> float:
        warmup = total // 2
        if step < warmup:
            return 0.02 + (0.04 - 0.02) * step / warmup
        return 0.04

    # ── Momentum schedule: 0.996 → 0.9999 ───────────────────────────────────
    def momentum_schedule(step: int, total: int) -> float:
        return 0.996 + (0.9999 - 0.996) * step / max(total - 1, 1)

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    model.teacher_backbone.eval()
    model.teacher_head.eval()

    total_steps = min(args.steps, len(loader))
    print(f"Starting training for {total_steps} steps...\n")

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break

        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        t_temp = teacher_temp_schedule(step, total_steps)
        mom    = momentum_schedule(step, total_steps)

        # Forward
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, teacher_temp=t_temp)

        # Backward
        loss.backward()

        # Gradient clip
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.student_backbone.parameters()) +
            list(model.student_head.parameters()),
            max_norm=3.0,
        )

        optimizer.step()

        # EMA teacher update
        model.update_teacher(momentum=mom)

        print(
            f"step {step+1:>3}/{total_steps} | "
            f"loss={loss.item():.4f} | "
            f"grad_norm={grad_norm:.3f} | "
            f"t_temp={t_temp:.4f} | "
            f"momentum={mom:.4f}"
        )

    print(f"\n{'='*60}")
    print("  Training loop completed successfully.")
    print(f"  Final loss : {loss.item():.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Gene-DINO prototype training")
    p.add_argument("--device",       default="cpu",  type=str)
    p.add_argument("--batch_size",   default=3,      type=int)
    p.add_argument("--steps",        default=10,     type=int)
    p.add_argument("--dino_out_dim", default=4096,   type=int,
                   help="number of DINO prototypes (K)")
    p.add_argument("--transform",    default=["lognorm"], type=str,
                   nargs="+",
                   choices=["lognorm", "normalize", "nicheformer", "scfoundation"],
                   help=(
                       "Augmentation transform(s). One or more: "
                       "lognorm normalize nicheformer scfoundation. "
                       "Single: --transform lognorm  "
                       "Multi:  --transform lognorm normalize nicheformer scfoundation"
                   ))
    p.add_argument("--checkpoint",   default="none",    type=str,
                   help=(
                       "Backbone weights to load. Options: "
                       "'none' (random, default) | "
                       "'theislab/Nicheformer' (download from HuggingFace) | "
                       "'/path/to/model.safetensors' (local file)"
                   ))
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
