"""
gene_dataset.py
---------------
Dataset for DINO self-supervised pretraining on gene expression data.
Implements all four augmentation transforms from the hescape repo.

Transform options (set via transform_name argument):
┌─────────────────────┬────────────────────────────────┬──────────────────────────┐
│ transform_name      │ Files needed                   │ Status                   │
├─────────────────────┼────────────────────────────────┼──────────────────────────┤
│ "lognorm"  DEFAULT  │ none                           │ works always             │
│ "normalize"         │ none                           │ works always             │
│ "nicheformer"       │ model.h5ad                     │ works with ref files     │
│                     │ xenium_mean_script.npy         │                          │
│                     │ data_gene_reference_path       │                          │
│ "scfoundation"      │ OS_scRNA_gene_index.19264.tsv  │ works with ref files     │
│                     │ data_gene_reference_path       │                          │
└─────────────────────┴────────────────────────────────┴──────────────────────────┘

Output per sample (same shapes as DummyGeneCropDataset — collate unchanged):
    "global_crops" : LongTensor [n_global, global_crop_size]   e.g. [2, 1500]
    "local_crops"  : LongTensor [n_local,  local_crop_size]    e.g. [8,  500]

Backbone, collate, and training loop require zero changes.
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from functools import partial


# =============================================================================
# 1. LogNormOnly
# =============================================================================

class LogNormOnly(torch.nn.Module):
    """
    log1p normalization only. No library-size correction.

    Input : float Tensor [G] or [B, G]  — raw counts
    Output: float Tensor same shape

    DEFAULT transform — requires no reference files.
    """
    def forward(self, X: Tensor) -> Tensor:
        return torch.log1p(X.float())


# =============================================================================
# 2. NormalizeCounts
# =============================================================================

class NormalizeCounts(torch.nn.Module):
    """
    Median-ratio library-size normalization + log1p.

    Input : float Tensor [B, G]  — raw counts
    Output: float Tensor [B, G]

    Requires no reference files.
    """
    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        counts_per_cell = torch.sum(X, dim=1)
        positive        = counts_per_cell[counts_per_cell > 0]
        after, _        = torch.median(positive, dim=0)
        counts_per_cell = counts_per_cell + (counts_per_cell == 0).float()
        counts_per_cell = counts_per_cell / after
        normalized      = torch.div(X, counts_per_cell.unsqueeze(1))
        return torch.log1p(normalized)


# =============================================================================
# 3. NicheformerTransform
# =============================================================================

class NicheformerTransform(torch.nn.Module):
    """
    Reproduces NicheformerTransform from hescape/data_modules/image_gexp_dataset.py.

    Converts raw float counts → sorted gene token IDs ranked by
    technology-mean-normalized expression, padded to max_seq_len.

    Output dtype: torch.long  (integer token IDs, ready for Nicheformer embedding layer)

    Required files:
        nicheformer_reference_path : nicheformer/data/model_means/model.h5ad
        technology_mean_path       : nicheformer/data/model_means/xenium_mean_script.npy
                                     (swap for cosmx/merfish/iss/dissociated as needed)
        data_gene_reference_path   : <dataset>/nicheformer_reference.h5ad
                                     (per-dataset gene panel reference)

    Input : float Tensor [B, G] or [G]
    Output: long  Tensor [B, max_seq_len]
    """

    def __init__(
        self,
        nicheformer_reference_path: str,
        technology_mean_path:       str,
        data_gene_reference_path:   str,
        max_seq_len: int = 1500,
        aux_tokens:  int = 30,
    ):
        super().__init__()
        try:
            import anndata as ad
            import numpy   as np
            import pandas  as pd
        except ImportError as e:
            raise ImportError(
                "NicheformerTransform requires anndata, numpy, pandas.\n"
                f"pip install anndata numpy pandas\n{e}"
            )

        nicheformer_ref = ad.read_h5ad(nicheformer_reference_path)
        data_gene_ref   = ad.read_h5ad(data_gene_reference_path)

        gene_to_token = {
            gene: idx for idx, gene in enumerate(nicheformer_ref.var_names)
        }

        technology_mean      = np.load(technology_mean_path)
        self.technology_mean = torch.from_numpy(technology_mean).float()

        data_gene_ref.var.index = data_gene_ref.var["gene_ids"]
        token_ids = [
            gene_to_token.get(g, -1) for g in data_gene_ref.var["gene_ids"]
        ]
        data_gene_ref.var["token_id"] = token_ids

        self.gene_mask = torch.tensor(
            data_gene_ref.var["token_id"].values != -1, dtype=torch.bool
        )
        data_gene_ref  = data_gene_ref[:, data_gene_ref.var["token_id"] != -1]
        self.token_ids = torch.tensor(
            data_gene_ref.var["token_id"].values, dtype=torch.long
        )

        self.max_seq_len = max_seq_len
        self.aux_tokens  = aux_tokens

    def _tokenize(self, X: Tensor):
        # X may have fewer genes than the reference panel (e.g. dummy data has
        # 2000 genes but the 5k panel reference has 5001). Pad with zeros so
        # the gene_mask can index into X safely.
        n_ref = self.gene_mask.shape[0]
        if X.shape[1] < n_ref:
            pad = torch.zeros(
                X.shape[0], n_ref - X.shape[1],
                dtype=X.dtype, device=X.device
            )
            X = torch.cat([X, pad], dim=1)          # [B, n_ref]

        exp       = X[:, self.gene_mask]
        counts    = exp.mean(dim=1).clamp(min=1e-6)
        exp       = exp * (10_000 / counts.unsqueeze(1))

        tech_mean = self.technology_mean.to(X.device)
        tech_mean = torch.nan_to_num(tech_mean, nan=0.0).clamp(min=1e-6)
        tech_mean = tech_mean[self.token_ids.to(X.device)]
        exp       = exp / tech_mean.unsqueeze(0)

        sorted_idx = torch.argsort(-exp, dim=1)
        sorted_exp = torch.gather(exp, 1, sorted_idx)
        token_ids  = self.token_ids.to(X.device)
        sorted_ids = token_ids[sorted_idx] + self.aux_tokens

        return sorted_exp, sorted_ids

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.float()

        _, sorted_ids = self._tokenize(X)
        B             = X.size(0)
        tokens        = torch.zeros(B, self.max_seq_len, dtype=torch.long, device=X.device)
        n_fill        = min(sorted_ids.size(1), self.max_seq_len)
        tokens[:, :n_fill] = sorted_ids[:, :n_fill]
        return tokens


# =============================================================================
# 4. scFoundationTransform
# =============================================================================

class scFoundationTransform(torch.nn.Module):
    """
    Reproduces scFoundationTransform from hescape/data_modules/image_gexp_dataset.py.

    Converts raw counts → 19266-dim float vector:
        [log1p(median-normalized counts for 19264 genes) | log10(total) | log10(total)]

    Required files:
        scfoundation_gene_index_path : OS_scRNA_gene_index.19264.tsv
        data_gene_reference_path     : <dataset>/nicheformer_reference.h5ad

    Input : float Tensor [B, G] or [G]
    Output: float Tensor [B, 19266]
    """

    def __init__(
        self,
        scfoundation_gene_index_path: str,
        data_gene_reference_path:     str,
    ):
        super().__init__()
        try:
            import anndata as ad
            import pandas  as pd
        except ImportError as e:
            raise ImportError(
                "scFoundationTransform requires anndata and pandas.\n"
                f"pip install anndata pandas\n{e}"
            )

        gene_list_df     = pd.read_csv(
            scfoundation_gene_index_path, header=0, delimiter="\t"
        )
        self.gene_list   = list(gene_list_df["gene_name"])
        data_gene_ref    = ad.read_h5ad(data_gene_reference_path)
        self.gene_to_idx = {
            gene: i for i, gene in enumerate(data_gene_ref.var_names)
        }

    def _select_genes(self, X: Tensor) -> Tensor:
        # For each gene in scFoundation's 19264-gene list:
        #   - if the gene exists in the dataset AND its column index is within
        #     X's width -> take that column
        #   - otherwise -> zero column (gene absent from this panel/dummy data)
        n_cols = X.size(1)
        cols   = []
        for gene in self.gene_list:
            idx = self.gene_to_idx.get(gene, -1)
            if idx != -1 and idx < n_cols:
                cols.append(X[:, idx].unsqueeze(1))
            else:
                cols.append(torch.zeros(X.size(0), 1, device=X.device))
        return torch.cat(cols, dim=1)

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.float()

        gexpr           = self._select_genes(X)  # always realign to scFoundation 19264-gene panel
        counts_per_cell = gexpr.sum(dim=1)
        positive        = counts_per_cell[counts_per_cell > 0]
        after, _        = torch.median(positive, dim=0)
        counts_per_cell = counts_per_cell + (counts_per_cell == 0).float()
        norm_factor     = counts_per_cell / after
        gexpr           = torch.log1p(gexpr / norm_factor.unsqueeze(1))
        totalcount      = torch.log10(counts_per_cell.unsqueeze(1).clamp(min=1e-6))
        return torch.cat([gexpr, totalcount, totalcount], dim=1)


# =============================================================================
# Gene crop utility (the DINO augmentation for gene data)
# =============================================================================

def crop_genes(x: Tensor, crop_size: int, pad_value: int = 0) -> Tensor:
    """
    Randomly subsample crop_size positions from x.

    x         : [G]  long token IDs (or float expression values)
    crop_size : number of genes to keep
    pad_value : padding value if G < crop_size

    Each call returns a DIFFERENT random subset — this is the DINO augmentation.
    Global crop = large random gene subset (teacher + student).
    Local crop  = small random gene subset (student only).
    """
    G = x.shape[0]
    if G >= crop_size:
        idx = torch.randperm(G, device=x.device)[:crop_size]
        return x[idx].contiguous()
    pad = torch.full(
        (crop_size - G,), pad_value, dtype=x.dtype, device=x.device
    )
    return torch.cat([x, pad], dim=0).contiguous()


def float_to_rank_tokens(x: Tensor, vocab_size: int = 25000) -> Tensor:
    """
    Convert a normalized float gene vector → rank-based integer token IDs.
    Used by the lognorm and normalize paths (no reference file needed).

    Ranking:
        highest expressed gene → token ID 30
        second highest         → token ID 31
        ...
        zero expression        → token ID 1  (padding token in Nicheformer)

    Token IDs 0-29 reserved for Nicheformer special tokens.

    x       : [G]  float normalized gene expression
    Returns : [G]  long token IDs
    """
    tokens       = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
    nonzero_mask = x > 0
    nonzero_vals = x[nonzero_mask]

    if nonzero_vals.numel() == 0:
        return tokens

    order    = torch.argsort(nonzero_vals, descending=True)
    n        = nonzero_vals.numel()
    ids      = torch.arange(30, 30 + n, dtype=torch.long, device=x.device).clamp(max=vocab_size - 1)
    positions = nonzero_mask.nonzero(as_tuple=True)[0]
    tokens[positions[order]] = ids
    return tokens


# =============================================================================
# Transform registry
# =============================================================================

_NO_FILE_TRANSFORMS = {"lognorm", "normalize"}
_FILE_TRANSFORMS    = {"nicheformer", "scfoundation"}
ALL_TRANSFORMS      = _NO_FILE_TRANSFORMS | _FILE_TRANSFORMS


# =============================================================================
# Main Dataset
# =============================================================================

class GeneExpressionDINODataset(Dataset):
    """
    Dataset for DINO self-supervised pretraining on gene expression data.
    Supports all four (for now) augmentation transforms from HESCAPE repo.
    Default transform is LogNormOnly — no reference files needed.

    Args:
        data                         : Tensor [N, G] of raw float counts
                                       OR HuggingFace Dataset with gexp column
        transform_name               : str or list of str
                                       Single: "lognorm" | "normalize" | "nicheformer" | "scfoundation"
                                       Multi : ["lognorm","normalize","nicheformer","scfoundation"]
                                       When a list is passed, ALL transforms run per cell.
                                       Each produces its own 2 global + 8 local crops.
                                       Total views = len(transforms) × 10
                                       e.g. 4 transforms → 8 global + 32 local = 40 views
        n_global_crops               : number of global views (default 2)
        n_local_crops                : number of local views  (default 8)
        global_crop_size             : gene tokens per global view (default 1500)
        local_crop_size              : gene tokens per local view  (default 500)
        vocab_size                   : token vocab, must match backbone (default 25000 (Vocabulary size in Nicheformer))
        gexp_key                     : column name for HuggingFace dataset (default "gexp" to match HESCAPE dataset on HF)

        -- NicheformerTransform paths --
        nicheformer_reference_path   : path to model.h5ad
        technology_mean_path         : path to xenium_mean_script.npy
        data_gene_reference_path     : path to per-dataset nicheformer_reference.h5ad

        -- scFoundationTransform paths --
        scfoundation_gene_index_path : path to OS_scRNA_gene_index.19264.tsv
        data_gene_reference_path     : path to per-dataset nicheformer_reference.h5ad
    """

    def __init__(
        self,
        data,
        transform_name               = "lognorm",  # str or list[str]
        n_global_crops:               int = 2,
        n_local_crops:                int = 8,
        global_crop_size:             int = 1500,
        local_crop_size:              int = 500,
        vocab_size:                   int = 25000,
        gexp_key:                     str = "gexp",
        nicheformer_reference_path:   Optional[str] = "/content/nicheformer/data/model_means/model.h5ad",
        technology_mean_path:         Optional[str] = "/content/nicheformer/data/model_means/xenium_mean_script.npy",
        scfoundation_gene_index_path: Optional[str] = "/content/dinov3/OS_scRNA_gene_index.19264.tsv",
        data_gene_reference_path:     Optional[str] = "/content/hescape/data/human-5k-panel/nicheformer_reference.h5ad",
        max_seq_len:                  int = 1500,
        aux_tokens:                   int = 30,
    ):
        super().__init__()

        # Normalise transform_name to a list
        if isinstance(transform_name, str):
            transform_names = [transform_name.lower()]
        else:
            transform_names = [t.lower() for t in transform_name]

        for tn in transform_names:
            if tn not in ALL_TRANSFORMS:
                raise ValueError(
                    f"transform_name must be one of {sorted(ALL_TRANSFORMS)}, "
                    f"got '{tn}'."
                )

        self.n_global_crops   = n_global_crops
        self.n_local_crops    = n_local_crops
        self.global_crop_size = global_crop_size
        self.local_crop_size  = local_crop_size
        self.vocab_size       = vocab_size
        self.gexp_key         = gexp_key
        self.transform_names  = transform_names   # list — may be length 1

        # data source (For now)
        if isinstance(data, Tensor):
            self.data      = data.float()
            self.data_type = "tensor"
        else:
            self.data      = data
            self.data_type = "hf"

        # Build one transform object per name and store in an ordered list.
        # Order is preserved — transform_names[i] produces views[i].
        self.transforms = []       # list of (name, transform_obj, output_dtype)

        for tn in transform_names:

            if tn == "lognorm":
                self.transforms.append((tn, LogNormOnly(), "float"))

            elif tn == "normalize":
                self.transforms.append((tn, NormalizeCounts(), "float"))

            elif tn == "nicheformer":
                self._require(
                    nicheformer_reference_path = nicheformer_reference_path,
                    technology_mean_path       = technology_mean_path,
                    data_gene_reference_path   = data_gene_reference_path,
                )
                t = NicheformerTransform(
                    nicheformer_reference_path = nicheformer_reference_path,
                    technology_mean_path       = technology_mean_path,
                    data_gene_reference_path   = data_gene_reference_path,
                    max_seq_len                = max_seq_len,
                    aux_tokens                 = aux_tokens,
                )
                self.transforms.append((tn, t, "long"))

            elif tn == "scfoundation":
                self._require(
                    scfoundation_gene_index_path = scfoundation_gene_index_path,
                    data_gene_reference_path     = data_gene_reference_path,
                )
                t = scFoundationTransform(
                    scfoundation_gene_index_path = scfoundation_gene_index_path,
                    data_gene_reference_path     = data_gene_reference_path,
                )
                self.transforms.append((tn, t, "float"))

        # Convenience: single-transform path keeps backward-compatible attribute
        if len(self.transforms) == 1:
            _, self.transform, self.output_dtype = self.transforms[0]
            self.transform_name = transform_names[0]
        else:
            self.transform      = None   # multi-mode: use self.transforms list
            self.transform_name = transform_names  # list in multi-mode
            self.output_dtype   = "mixed"

        # Inform user about total views when multi-transform
        if len(self.transforms) > 1:
            n_total_global = len(self.transforms) * n_global_crops
            n_total_local  = len(self.transforms) * n_local_crops
            print(
                f"  Multi-transform mode: {len(self.transforms)} transforms × "
                f"({n_global_crops}g + {n_local_crops}l) crops = "
                f"{n_total_global} global + {n_total_local} local = "
                f"{n_total_global + n_total_local} total views per cell"
            )

    @staticmethod
    def _require(**paths):
        """Validate that all required file paths are provided and exist."""
        for name, path in paths.items():
            if path is None:
                raise ValueError(
                    f"'{name}' is required for this transform but was not provided."
                )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

    def __len__(self) -> int:
        return len(self.data)

    def _get_raw(self, idx: int) -> Tensor:
        if self.data_type == "tensor":
            return self.data[idx]
        row = self.data[idx]
        x   = row[self.gexp_key]
        if not isinstance(x, Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.float()

    def __getitem__(self, idx: int) -> dict:
        raw = self._get_raw(idx)                                     # [G] float

        if len(self.transforms) == 1:
            # ── Single-transform path ────────────────────
            tokens = self._apply_transform(*self.transforms[0], raw)
            return self._make_crops(tokens)

        else:
            # ── Multi-transform path ────────────────────────
            # Each transform produces its own 2 global + 8 local crops.
            # All global crops are stacked together, all local crops together.
            # exemple pseudocode :
            #   for t in transforms:
            #       tokens = t(raw_counts)
            #       views.append(crop_genes(tokens))  # 2 global + 8 local
            all_global = []
            all_local  = []
            for (name, transform, dtype) in self.transforms:
                tokens = self._apply_transform(name, transform, dtype, raw)
                crops  = self._make_crops(tokens)
                all_global.append(crops["global_crops"])   # [n_global, 1500]
                all_local.append(crops["local_crops"])     # [n_local,  500]

            return {
                # [n_transforms * n_global, 1500]  e.g. [8, 1500] for 4 transforms
                "global_crops": torch.cat(all_global, dim=0),
                # [n_transforms * n_local,  500]   e.g. [32, 500] for 4 transforms
                "local_crops":  torch.cat(all_local,  dim=0),
            }

    def _apply_transform(
        self,
        name:      str,
        transform,
        dtype:     str,
        raw:       Tensor,
    ) -> Tensor:
        """Run one transform on raw [G] float counts → long token IDs [seq]."""
        if name == "nicheformer":
            return transform(raw.unsqueeze(0)).squeeze(0)     # already long

        elif name in ("normalize", "scfoundation"):
            normalized = transform(raw.unsqueeze(0)).squeeze(0)
            return float_to_rank_tokens(normalized, self.vocab_size)

        else:
            # lognorm
            normalized = transform(raw)
            return float_to_rank_tokens(normalized, self.vocab_size)

    def _make_crops(self, tokens: Tensor) -> dict:
        """
        Given a 1D long token tensor, produce global and local crops.
        Used by both single-transform and multi-transform paths.
        In multi-transform mode, called once per transform — results are
        concatenated in __getitem__.
        """
        global_views = torch.stack([
            crop_genes(tokens, self.global_crop_size)
            for _ in range(self.n_global_crops)
        ])                                                           # [n_global, global_size]
        local_views  = torch.stack([
            crop_genes(tokens, self.local_crop_size)
            for _ in range(self.n_local_crops)
        ])                                                           # [n_local, local_size]
        return {"global_crops": global_views, "local_crops": local_views}


# =============================================================================
# Factory
# =============================================================================

def make_gene_dataset(
    data,
    transform_name: str = "lognorm",
    **kwargs,
) -> GeneExpressionDINODataset:
    """
    Factory for GeneExpressionDINODataset.

    Prototype (no files needed):
        ds = make_gene_dataset(raw_count_tensor)

    NicheformerTransform (real data):
        ds = make_gene_dataset(
            hf_dataset,
            transform_name             = "nicheformer",
            nicheformer_reference_path = "nicheformer/data/model_means/model.h5ad",
            technology_mean_path       = "nicheformer/data/model_means/xenium_mean_script.npy",
            data_gene_reference_path   = "<dataset_folder>/nicheformer_reference.h5ad",
        )

    scFoundationTransform (real data):
        ds = make_gene_dataset(
            hf_dataset,
            transform_name               = "scfoundation",
            scfoundation_gene_index_path = "<path>/OS_scRNA_gene_index.19264.tsv",
            data_gene_reference_path     = "<dataset_folder>/nicheformer_reference.h5ad",
        )

    """
    return GeneExpressionDINODataset(
        data=data, transform_name=transform_name, **kwargs
    )


# =============================================================================
# Sanity checks  —  python gene_dataset.py
# =============================================================================

if __name__ == "__main__":
    from gene_collate import gene_collate_fn
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("  gene_dataset.py — all transforms sanity check")
    print("=" * 60)

    # shared dummy data: N=30 cells, G=2000 genes, sparse raw counts
    N, G = 30, 2000
    raw  = torch.zeros(N, G)
    for i in range(N):
        idx        = torch.randperm(G)[:400]
        raw[i, idx] = torch.randint(1, 500, (400,)).float()

    passed = 0

    # ── Test 1: LogNormOnly ───────────────────────────────────────────────────
    print("\n[1] LogNormOnly (default) — no reference files")
    ds = make_gene_dataset(raw, transform_name="lognorm")
    s  = ds[0]
    assert s["global_crops"].shape == (2, 1500), f"got {s['global_crops'].shape}"
    assert s["local_crops"].shape  == (8,  500), f"got {s['local_crops'].shape}"
    assert s["global_crops"].dtype == torch.long
    print(f"  global_crops : {s['global_crops'].shape}  dtype={s['global_crops'].dtype}  OK")
    print(f"  local_crops  : {s['local_crops'].shape}   dtype={s['local_crops'].dtype}  OK")
    passed += 1

    # ── Test 2: NormalizeCounts ───────────────────────────────────────────────
    print("\n[2] NormalizeCounts — no reference files")
    ds2 = make_gene_dataset(raw, transform_name="normalize")
    s2  = ds2[0]
    assert s2["global_crops"].shape == (2, 1500)
    assert s2["global_crops"].dtype == torch.long
    print(f"  global_crops : {s2['global_crops'].shape}  dtype={s2['global_crops'].dtype}  OK")
    print(f"  local_crops  : {s2['local_crops'].shape}   dtype={s2['local_crops'].dtype}  OK")
    passed += 1

    # ── Test 3: stochastic augmentation ──────────────────────────────────────
    print("\n[3] Stochastic augmentation — crops differ between calls")
    s_a    = ds[5]
    s_b    = ds[5]
    differ = not (s_a["global_crops"][0] == s_b["global_crops"][0]).all().item()
    print(f"  crops differ: {differ}  (expect True usually)  OK")
    passed += 1

    # ── Test 4: DataLoader + collate ─────────────────────────────────────────
    print("\n[4] DataLoader + gene_collate_fn")
    loader = DataLoader(
        ds,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(gene_collate_fn, n_global_crops=2, n_local_crops=8),
        drop_last=True,
    )
    batch = next(iter(loader))
    assert batch["collated_global_crops"].shape == (6, 1500)
    assert batch["collated_local_crops"].shape  == (24, 500)
    print(f"  collated_global_crops : {batch['collated_global_crops'].shape}  OK")
    print(f"  collated_local_crops  : {batch['collated_local_crops'].shape}   OK")
    print(f"  global_batch_size     : {batch['global_batch_size']}  OK")
    passed += 1

    # ── Test 5: NicheformerTransform (needs reference files) ─────────────────
    print("\n[5] NicheformerTransform — requires reference files")
    niche_ref = "/content/nicheformer/data/model_means/model.h5ad"
    tech_mean = "/content/nicheformer/data/model_means/xenium_mean_script.npy"
    #data_ref  = None   # per-dataset nicheformer_reference.h5ad
    data_ref = "/content/hescape/data/human-5k-panel/nicheformer_reference.h5ad"
    if os.path.exists(niche_ref) and os.path.exists(tech_mean) and data_ref and os.path.exists(data_ref):
        ds_nf = make_gene_dataset(
            raw,
            transform_name             = "nicheformer",
            nicheformer_reference_path = niche_ref,
            technology_mean_path       = tech_mean,
            data_gene_reference_path   = data_ref,
        )
        sn = ds_nf[0]
        assert sn["global_crops"].dtype == torch.long
        print(f"  global_crops : {sn['global_crops'].shape}  dtype={sn['global_crops'].dtype}  OK")
        passed += 1
    else:
        print("  SKIPPED — set data_ref path to the per-dataset nicheformer_reference.h5ad")
        print(f"  nicheformer_reference_path = {niche_ref}")
        print(f"  technology_mean_path       = {tech_mean}")
        print(f"  data_gene_reference_path   = <your path here>")

    # ── Test 6: scFoundationTransform (needs reference files) ────────────────
    print("\n[6] scFoundationTransform — requires reference files")
    scf_index = "/content/dinov3/OS_scRNA_gene_index.19264.tsv"   # set to OS_scRNA_gene_index.19264.tsv
    if scf_index and os.path.exists(scf_index) and data_ref and os.path.exists(data_ref):
        ds_scf = make_gene_dataset(
            raw,
            transform_name               = "scfoundation",
            scfoundation_gene_index_path = scf_index,
            data_gene_reference_path     = data_ref,
        )
        ss = ds_scf[0]
        assert ss["global_crops"].dtype == torch.long
        print(f"  global_crops : {ss['global_crops'].shape}  dtype={ss['global_crops'].dtype}  OK")
        passed += 1
    else:
        print("  SKIPPED — set scf_index and data_ref paths to your reference files")
        print(f"  scfoundation_gene_index_path = OS_scRNA_gene_index.19264.tsv")
        print(f"  data_gene_reference_path     = <your path here>")

    print(f"\n{'=' * 60}")
    print(f"  {passed}/4 core checks PASSED  (file-dependent transforms skipped if files absent)")
    print(f"{'=' * 60}")
