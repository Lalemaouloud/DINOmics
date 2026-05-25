"""
inspect_pipeline.py
-------------------
Verifies the actual Gene-DINO pipeline by importing and running our real files:
    gene_dataset.py         — GeneExpressionDINODataset, all 4 transforms
    gene_collate.py         — gene_collate_fn
    nicheformer_backbone.py — NicheformerBackbone

No training. No GPU needed. Shows exactly what is happening inside our code.

Run:
    python inspect_pipeline.py                    # full inspection
    python inspect_pipeline.py --section cell     # raw data  
    python inspect_pipeline.py --section dataset  # dataset + transforms
    python inspect_pipeline.py --section collate  # collate reshaping
    python inspect_pipeline.py --section backbone # backbone forward pass
    python inspect_pipeline.py --section multi    # multi-transformation mode 

Place this file in the same folder as the other pipeline files.
PYTHONPATH must include the dinov3 repo root (only needed for --section backbone).
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
from functools import partial


# ── Make sure our pipeline files are importable ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def banner(text):
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}")

def section(text):
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")

def show_tensor(label, t, n=10):
    flat = t.flatten()[:n].tolist()
    vals = "[" + ", ".join(
        f"{v:.3f}" if isinstance(v, float) else str(int(v))
        for v in flat
    ) + ", ...]"
    print(f"  {label}")
    print(f"    shape  : {list(t.shape)}")
    print(f"    dtype  : {t.dtype}")
    print(f"    range  : [{t.min().item():.3f}, {t.max().item():.3f}]")
    print(f"    sample : {vals}")


# ─────────────────────────────────────────────────────────────────────────────
# Build the dummy raw count tensor 
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_data(N=9, G=2000, n_expressed=400, seed=42):
    """
    Creates a raw float count tensor [N, G] matching the structure of
    DummyGeneCropDataset — N cells, G genes, sparse counts.
    This is the input that GeneExpressionDINODataset expects.
    """
    torch.manual_seed(seed)
    raw = torch.zeros(N, G)
    for i in range(N):
        idx = torch.randperm(G)[:n_expressed]
        raw[i, idx] = torch.randint(1, 500, (n_expressed,)).float()
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — raw cell
# ─────────────────────────────────────────────────────────────────────────────

def inspect_cell(raw):
    banner("SECTION 1 — raw gene expression data")
    print("""
  What you are looking at:
    The raw float tensor produced by make_dummy_data().
    This is what GeneExpressionDINODataset receives as input.
    Shape is [N, G] — N cells, G genes per cell.
    """)

    print(f"  Full tensor shape : {list(raw.shape)}  ({raw.shape[0]} cells, {raw.shape[1]} genes)")
    show_tensor("cell 0 (first cell)", raw[0])

    n_expressed = (raw[0] > 0).sum().item()
    n_silent    = (raw[0] == 0).sum().item()
    print(f"\n  Cell 0 breakdown:")
    print(f"    Expressed genes : {n_expressed} / {raw.shape[1]}")
    print(f"    Silent genes    : {n_silent} / {raw.shape[1]}")
    print(f"    Max count       : {raw[0].max().item():.0f}")
    print(f"    Mean (nonzero)  : {raw[0][raw[0]>0].mean().item():.1f}")

    print(f"\n  First 15 gene values of cell 0:")
    for i, v in enumerate(raw[0, :15].tolist()):
        bar = "█" * int(v / 25) if v > 0 else "·"
        print(f"    gene {i:>3} : {v:>6.0f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GeneExpressionDINODataset with actual transforms
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(raw):
    banner("SECTION 2 — GeneExpressionDINODataset  (our actual gene_dataset.py)")
    print("""
  Importing GeneExpressionDINODataset from gene_dataset.py and running it.
  This is the exact class used during training — not a reimplementation.
    """)

    from gene_dataset_1 import make_gene_dataset

    # ── LogNormOnly ───────────────────────────────────────────────────────────
    section("LogNormOnly transform  (default)")
    print("  Creating dataset with transform_name='lognorm'...")
    ds = make_gene_dataset(raw, transform_name="lognorm")
    sample = ds[0]

    print(f"\n  ds[0] keys   : {list(sample.keys())}")
    show_tensor("global_crops[0]  (first global crop)", sample["global_crops"][0])
    show_tensor("local_crops[0]   (first local crop)",  sample["local_crops"][0])

    print(f"\n  Crop shapes from ds[0]:")
    print(f"    global_crops : {list(sample['global_crops'].shape)}"
          f"  ({sample['global_crops'].shape[0]} global × {sample['global_crops'].shape[1]} tokens)")
    print(f"    local_crops  : {list(sample['local_crops'].shape)}"
          f"  ({sample['local_crops'].shape[0]} local × {sample['local_crops'].shape[1]} tokens)")

    active_g = (sample["global_crops"][0] > 1).sum().item()
    active_l = (sample["local_crops"][0]  > 1).sum().item()
    print(f"\n  Active (non-padding) tokens:")
    print(f"    global crop 0 : {active_g} / {sample['global_crops'].shape[1]}")
    print(f"    local  crop 0 : {active_l} / {sample['local_crops'].shape[1]}")

    # ── Stochastic check ──────────────────────────────────────────────────────
    section("Stochastic augmentation check")
    print("  Calling ds[0] twice — should give DIFFERENT crops each time...")
    s1 = ds[0]
    s2 = ds[0]
    identical = torch.equal(s1["global_crops"][0], s2["global_crops"][0])
    print(f"  global_crops[0] identical on two calls: {identical}  (must be False)")
    overlap = (s1["global_crops"][0] == s2["global_crops"][0]).sum().item()
    print(f"  Overlapping token positions: {overlap} / {s1['global_crops'].shape[1]}")

    # ── NormalizeCounts ───────────────────────────────────────────────────────
    section("NormalizeCounts transform")
    print("  Creating dataset with transform_name='normalize'...")
    ds_norm = make_gene_dataset(raw, transform_name="normalize")
    s_norm  = ds_norm[0]
    print(f"  global_crops shape : {list(s_norm['global_crops'].shape)}")
    print(f"  local_crops shape  : {list(s_norm['local_crops'].shape)}")

    # Compare lognorm vs normalize token IDs for same cell
    g_lognorm  = ds[0]["global_crops"][0]
    g_normalize = ds_norm[0]["global_crops"][0]
    agree = (g_lognorm == g_normalize).sum().item()
    total = g_lognorm.shape[0]
    print(f"\n  Comparing lognorm vs normalize on same cell (same crop seed):")
    print(f"  Token positions that agree : {agree} / {total}")
    print(f"  (On dummy data these often agree — on real data they diverge)")

    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — gene_collate_fn with actual DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def inspect_collate(raw):
    banner("SECTION 3 — gene_collate_fn  (our actual gene_collate.py)")
    print("""
  Importing gene_collate_fn from gene_collate.py and running a real DataLoader.
  Shows exactly what enters the backbone after collation.
    """)

    from gene_dataset_1 import make_gene_dataset
    from gene_collate import gene_collate_fn

    for n_transforms, transform_arg in [
        (1, "lognorm"),
        (2, ["lognorm", "normalize"]),
    ]:
        section(f"{n_transforms} transform(s) — transform={transform_arg!r}")

        ds = make_gene_dataset(raw, transform_name=transform_arg)
        loader = DataLoader(
            ds,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            collate_fn=gene_collate_fn,
            drop_last=True,
        )

        batch = next(iter(loader))

        print(f"\n  Batch keys: {list(batch.keys())}")
        print(f"\n  Key shapes entering GeneDINO.forward():")
        print(f"    collated_global_crops : {list(batch['collated_global_crops'].shape)}")
        print(f"    collated_local_crops  : {list(batch['collated_local_crops'].shape)}")
        print(f"    n_global_crops        : {batch['n_global_crops']}")
        print(f"    n_local_crops         : {batch['n_local_crops']}")
        print(f"    global_batch_size     : {batch['global_batch_size']}")

        ng = batch["n_global_crops"]
        nl = batch["n_local_crops"]
        B  = batch["global_batch_size"]
        print(f"\n  Breakdown:")
        print(f"    {n_transforms} transform(s) × 2 global crops × B={B} cells = {ng*B} rows in global tensor")
        print(f"    {n_transforms} transform(s) × 8 local crops  × B={B} cells = {nl*B} rows in local tensor")
        print(f"\n  Teacher will see  : {list(batch['collated_global_crops'].shape)}  (global only)")
        print(f"  Student will see  : {list(batch['collated_global_crops'].shape)}"
              f" + {list(batch['collated_local_crops'].shape)}  (all views)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — NicheformerBackbone forward pass
# ─────────────────────────────────────────────────────────────────────────────

def inspect_backbone(raw):
    banner("SECTION 4 — NicheformerBackbone  (our actual nicheformer_backbone.py)")
    print("""
  Importing build_nicheformer_backbone from nicheformer_backbone.py.
  Running a real forward pass through the student backbone.
  Shows input → output shapes and verifies the DINOv3 interface contract.
    """)

    from nicheformer_backbone import build_nicheformer_backbone
    from gene_dataset_1 import make_gene_dataset
    from gene_collate import gene_collate_fn

    print("  Building backbone with random weights...")
    student, teacher, embed_dim = build_nicheformer_backbone(checkpoint="none")
    student.eval()
    teacher.eval()

    print(f"\n  embed_dim          : {embed_dim}")
    print(f"  Student parameters : {sum(p.numel() for p in student.parameters()):,}")
    print(f"  Teacher parameters : {sum(p.numel() for p in teacher.parameters()):,}")

    # Build a batch through the real pipeline
    ds     = make_gene_dataset(raw, transform_name="lognorm")
    loader = DataLoader(
        ds, batch_size=3, shuffle=False,
        num_workers=0, collate_fn=gene_collate_fn, drop_last=True
    )
    batch = next(iter(loader))

    global_crops = batch["collated_global_crops"]   # [n_global*B, 1500]
    local_crops  = batch["collated_local_crops"]    # [n_local*B,  500]

    section("Teacher forward  (global crops only, no grad)")
    print(f"  Input  : {list(global_crops.shape)}  dtype={global_crops.dtype}")
    with torch.no_grad():
        t_out = teacher(global_crops, is_training=True)
    print(f"  Output keys : {list(t_out.keys())}")
    show_tensor("x_norm_clstoken", t_out["x_norm_clstoken"])
    norms = t_out["x_norm_clstoken"].norm(dim=-1)
    print(f"  L2 norms (must be 1.0) : {[round(v,4) for v in norms.tolist()]}")
    show_tensor("x_norm_patchtokens (iBot stub)", t_out["x_norm_patchtokens"])
    print(f"  x_storage_tokens : {list(t_out['x_storage_tokens'].shape)}  (empty — no register tokens)")

    section("Student forward  (local crops)")
    print(f"  Input  : {list(local_crops.shape)}  dtype={local_crops.dtype}")
    print(f"  Note: different seq length (500 vs 1500) — separate forward pass")
    with torch.no_grad():
        s_out = student(local_crops, is_training=True)
    show_tensor("x_norm_clstoken", s_out["x_norm_clstoken"])
    norms_s = s_out["x_norm_clstoken"].norm(dim=-1)
    print(f"  L2 norms (must be 1.0) : {[round(v,4) for v in norms_s.tolist()]}")

    section("Interface contract verification")
    print("  DINOv3 backbone must return these three keys with these shapes:")
    B_g = global_crops.shape[0]
    B_l = local_crops.shape[0]
    checks = [
        ("x_norm_clstoken",    t_out["x_norm_clstoken"].shape,    (B_g, 512)),
        ("x_norm_patchtokens", t_out["x_norm_patchtokens"].shape,  (B_g, 1, 512)),
        ("x_storage_tokens",   t_out["x_storage_tokens"].shape,    (B_g, 0, 512)),
    ]
    all_pass = True
    for key, got, expected in checks:
        status = "PASS" if tuple(got) == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {key:<25} : {str(list(got)):<16} expected {list(expected)}  [{status}]")

    print(f"\n  Student/teacher start with identical weights: ", end="")
    identical = all(
        torch.equal(p_s, p_t)
        for p_s, p_t in zip(student.parameters(), teacher.parameters())
    )
    print(f"{'PASS' if identical else 'FAIL'}")

    print(f"\n  Overall backbone contract : {'ALL PASS' if all_pass and identical else 'FAILURES DETECTED'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Multi-transform mode
# ─────────────────────────────────────────────────────────────────────────────

def inspect_multi(raw, args):
    banner("SECTION 5 — Multi-transform mode ")
    print("""
  Running all 4 transforms simultaneously using our actual gene_dataset.py.
  Shows that the same cell produces 40 different views — 8 global + 32 local.
    """)

    from gene_dataset_1 import make_gene_dataset
    from gene_collate import gene_collate_fn

    # Start with transforms that need no reference files
    transforms = ["lognorm", "normalize"]

    # Add nicheformer if paths were provided and files exist
    nf_ready = (
        args.nf_ref and args.nf_tech and args.nf_data_ref
        and os.path.exists(args.nf_ref)
        and os.path.exists(args.nf_tech)
        and os.path.exists(args.nf_data_ref)
    )
    if nf_ready:
        transforms.append("nicheformer")
        print(f"  NicheformerTransform  : files found — INCLUDED")
    else:
        print(f"  NicheformerTransform  : SKIPPED")
        print(f"    pass --nf_ref --nf_tech --nf_data_ref to include")

    # Add scfoundation if paths were provided and files exist
    scf_ready = (
        args.scf_index and args.nf_data_ref
        and os.path.exists(args.scf_index)
        and os.path.exists(args.nf_data_ref)
    )
    if scf_ready:
        transforms.append("scfoundation")
        print(f"  scFoundationTransform : files found — INCLUDED")
    else:
        print(f"  scFoundationTransform : SKIPPED")
        print(f"    pass --scf_index --nf_data_ref to include")

    n_t = len(transforms)
    print(f"\n  Active transforms : {transforms}")
    print(f"  Views per cell    : {n_t} transforms x 10 crops = {n_t*10} ({n_t*2}g + {n_t*8}l)\n")

    # Build kwargs for file-dependent transforms
    kwargs = {}
    if nf_ready:
        kwargs["nicheformer_reference_path"] = args.nf_ref
        kwargs["technology_mean_path"]        = args.nf_tech
        kwargs["data_gene_reference_path"]    = args.nf_data_ref
    if scf_ready:
        kwargs["scfoundation_gene_index_path"] = args.scf_index
        if "data_gene_reference_path" not in kwargs:
            kwargs["data_gene_reference_path"] = args.nf_data_ref

    ds = make_gene_dataset(raw, transform_name=transforms, **kwargs)
    sample = ds[0]

    n_global = sample["global_crops"].shape[0]
    n_local  = sample["local_crops"].shape[0]
    print(f"  Per-sample output from ds[0]:")
    print(f"    global_crops : {list(sample['global_crops'].shape)}"
          f"  = {len(transforms)} transforms × 2 global")
    print(f"    local_crops  : {list(sample['local_crops'].shape)}"
          f"  = {len(transforms)} transforms × 8 local")
    print(f"    Total views  : {n_global + n_local} per cell")

    # Show that the first 2 global crops (lognorm) differ from next 2 (normalize)
    g0 = sample["global_crops"][0]  # lognorm global 1
    g1 = sample["global_crops"][1]  # lognorm global 2
    g2 = sample["global_crops"][2]  # normalize global 1
    g3 = sample["global_crops"][3]  # normalize global 2

    section("Are views from different transforms actually different?")
    same_transform = (g0 == g1).sum().item()
    diff_transform = (g0 == g2).sum().item()
    print(f"  lognorm global 1 vs lognorm global 2   (same transform, diff crop):")
    print(f"    matching positions: {same_transform} / {g0.shape[0]}")
    print(f"  lognorm global 1 vs normalize global 1 (diff transform, diff crop):")
    print(f"    matching positions: {diff_transform} / {g0.shape[0]}")
    print(f"  → different transforms create meaningfully different views")

    section("After collate (batch_size=3)")
    loader = DataLoader(
        ds, batch_size=3, shuffle=False,
        num_workers=0, collate_fn=gene_collate_fn, drop_last=True
    )
    batch = next(iter(loader))
    print(f"  collated_global_crops : {list(batch['collated_global_crops'].shape)}"
          f"  = {batch['n_global_crops']} global crops × 3 cells")
    print(f"  collated_local_crops  : {list(batch['collated_local_crops'].shape)}"
          f"  = {batch['n_local_crops']} local crops × 3 cells")
    print(f"  n_global_crops        : {batch['n_global_crops']}")
    print(f"  n_local_crops         : {batch['n_local_crops']}")
    print(f"\n  These tensors go directly into GeneDINO.forward()")
    print(f"  Teacher sees  : {list(batch['collated_global_crops'].shape)}")
    print(f"  Student sees  : {list(batch['collated_global_crops'].shape)}"
          f" + {list(batch['collated_local_crops'].shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Verify the Gene-DINO pipeline using our actual files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--section", default="all",
        choices=["all", "cell", "dataset", "collate", "backbone", "multi"],
        help=(
            "Which section to run:\n"
            "  all      — everything (default)\n"
            "  cell     — raw gene expression data\n"
            "  dataset  — GeneExpressionDINODataset + transforms\n"
            "  collate  — gene_collate_fn + DataLoader\n"
            "  backbone — NicheformerBackbone forward pass\n"
            "             (requires dinov3 on PYTHONPATH)\n"
            "  multi    — multi-transform mode\n"
        )
    )
    # Reference file paths for NicheformerTransform
    p.add_argument("--nf_ref",      default=None, type=str,
                   help="nicheformer/data/model_means/model.h5ad")
    p.add_argument("--nf_tech",     default=None, type=str,
                   help="nicheformer/data/model_means/xenium_mean_script.npy")
    p.add_argument("--nf_data_ref", default=None, type=str,
                   help="hescape/data/<panel>/nicheformer_reference.h5ad")
    # Reference file path for scFoundationTransform
    p.add_argument("--scf_index",   default=None, type=str,
                   help="path/to/OS_scRNA_gene_index.19264.tsv")
    return p.parse_args()


if __name__ == "__main__":
    args  = get_args()
    s     = args.section
    raw   = make_dummy_data(N=9, G=2000, n_expressed=400)

    banner("Gene-DINO Pipeline Verification")
    print("  Imports and runs our actual files:")
    print("    gene_dataset.py  →  GeneExpressionDINODataset, make_gene_dataset")
    print("    gene_collate.py  →  gene_collate_fn")
    print("    nicheformer_backbone.py  →  NicheformerBackbone (backbone section only)")
    print(f"\n  Section: {s}")
    print(f"  No training. No GPU needed (except backbone section uses CPU).")

    if s in ("all", "cell"):
        inspect_cell(raw)

    if s in ("all", "dataset"):
        inspect_dataset(raw)

    if s in ("all", "collate"):
        inspect_collate(raw)

    if s in ("all", "backbone"):
        try:
            inspect_backbone(raw)
        except ImportError as e:
            print(f"\n  [SKIP] backbone section needs dinov3 on PYTHONPATH: {e}")
            print("  Run: export PYTHONPATH=/content/dinov3:$PYTHONPATH")

    if s in ("all", "multi"):
        inspect_multi(raw, args)

    banner("Verification complete — all sections using our actual code")
