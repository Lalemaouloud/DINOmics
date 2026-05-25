"""
gene_collate.py
---------------
Replaces DINOv3's collate_data_and_cast (dinov3/data/collate.py) for gene data.

DINOv3's forward_backward() expects a batch dict with these keys:
    "collated_global_crops"  : Tensor [n_global*B, seq]
    "collated_local_crops"   : Tensor [n_local*B,  seq]
    "collated_masks"         : BoolTensor [n_global*B, n_patches]  ← iBOT (stubbed)
    "mask_indices_list"      : LongTensor [n_masked]               ← iBOT (stubbed)
    "masks_weight"           : Tensor [n_masked]                   ← iBOT (stubbed)
    "n_masked_patches"       : Tensor scalar                       ← iBOT (stubbed)
    "upperbound"             : int                                  ← iBOT (stubbed)
    "global_batch_size"      : int

Our dummy dataset (DummyGeneCropDataset) produces per-sample:
    "global_crops" : [2, seq_global]   e.g. [2, 1500]
    "local_crops"  : [8, seq_local]    e.g. [8, 500]

After DataLoader collation with batch_size=B those become:
    batch["global_crops"] : [B, 2, seq_global]
    batch["local_crops"]  : [B, 8, seq_local]

This function reshapes them into what forward_backward() expects.
"""

import torch
from torch import Tensor
from typing import List, Dict, Any


def gene_collate_fn(
    samples: List[Dict[str, Any]],
    n_global_crops: int = None,   # auto-detected from data if None
    n_local_crops:  int = None,   # auto-detected from data if None
    dtype: torch.dtype  = torch.long,
) -> Dict[str, Any]:
    """
    Custom collate for gene expression DINO training.
    Works in both single-transform and multi-transform mode.

    Single-transform (original):
        n_global_crops = 2,  n_local_crops = 8
        global_crops per sample : [2, 1500]
        local_crops  per sample : [8, 500]

    Multi-transform (new — 4 transforms):
        n_global_crops = 8,  n_local_crops = 32
        global_crops per sample : [8, 1500]   (4 transforms × 2 global)
        local_crops  per sample : [32, 500]   (4 transforms × 8 local)

    n_global_crops and n_local_crops are auto-detected from the first sample
    so the same collate function works for any number of transforms without
    any argument changes.
    """
    B = len(samples)

    # Auto-detect crop counts from the actual data shape
    # samples[0]["global_crops"] shape: [n_global, seq_global]
    _n_global = samples[0]["global_crops"].shape[0]
    _n_local  = samples[0]["local_crops"].shape[0]
    n_global_crops = n_global_crops or _n_global
    n_local_crops  = n_local_crops  or _n_local

    # ── Stack global crops ────────────────────────────────────────────────────
    # [B, n_global, seq_g] → permute → [n_global, B, seq_g] → reshape → [n_global*B, seq_g]
    global_crops = torch.stack(
        [s["global_crops"] for s in samples], dim=0
    ).to(dtype)
    seq_global    = global_crops.shape[-1]
    collated_global = (
        global_crops
        .permute(1, 0, 2)
        .reshape(n_global_crops * B, seq_global)
        .contiguous()
    )

    # ── Stack local crops ─────────────────────────────────────────────────────
    local_crops = torch.stack(
        [s["local_crops"] for s in samples], dim=0
    ).to(dtype)
    seq_local    = local_crops.shape[-1]
    collated_local = (
        local_crops
        .permute(1, 0, 2)
        .reshape(n_local_crops * B, seq_local)
        .contiguous()
    )

    # ── iBOT stubs ────────────────────────────────────────────────────────────
    # Scaled to match actual n_global_crops (8 in multi-transform mode)
    n_patches         = 1
    n_masked          = B * n_global_crops
    collated_masks    = torch.ones(n_global_crops * B, n_patches, dtype=torch.bool)
    mask_indices_list = torch.arange(n_masked, dtype=torch.long)
    masks_weight      = torch.ones(n_masked, dtype=torch.float32) / n_masked
    n_masked_patches  = torch.tensor(n_masked, dtype=torch.long)
    upperbound        = n_masked

    return {
        "collated_global_crops": collated_global,
        "collated_local_crops":  collated_local,
        "collated_masks":        collated_masks,
        "mask_indices_list":     mask_indices_list,
        "masks_weight":          masks_weight,
        "n_masked_patches":      n_masked_patches,
        "upperbound":            upperbound,
        "global_batch_size":     B,
        "n_global_crops":        n_global_crops,   # passed through for GeneDINO.forward()
        "n_local_crops":         n_local_crops,
    }


# ── quick sanity check ────────────────────────────────────────────────────────
# Inline dummy dataset so this file is self-contained for testing while actually running the code the following lines have no affect

if __name__ == "__main__":
    from torch.utils.data import DataLoader



    import torch
    from torch.utils.data import Dataset

    class _TinyDummy(Dataset):
        def __init__(self, n=9):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "global_crops": torch.randint(0, 25000, (2, 1500), dtype=torch.long),
                "local_crops":  torch.randint(0, 25000, (8, 500),  dtype=torch.long),
            }

    from functools import partial

    ds     = _TinyDummy(n=9)
    loader = DataLoader(
        ds,
        batch_size=3,
        collate_fn=partial(gene_collate_fn, n_global_crops=2, n_local_crops=8),
    )

    batch = next(iter(loader))
    print("collated_global_crops :", batch["collated_global_crops"].shape)   # [6, 1500]
    print("collated_local_crops  :", batch["collated_local_crops"].shape)    # [24, 500]
    print("collated_masks        :", batch["collated_masks"].shape)          # [6, 1]
    print("mask_indices_list     :", batch["mask_indices_list"].shape)       # [6]
    print("masks_weight          :", batch["masks_weight"].shape)            # [6]
    print("n_masked_patches      :", batch["n_masked_patches"])
    print("global_batch_size     :", batch["global_batch_size"])
    print("Collate sanity check PASSED.")
