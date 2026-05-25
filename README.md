# 🧬 Gene-DINO (DINOmics step 1)
### DINOv3 Adapted for Transcriptomics Gene Expression Data

> **Step 1 Prototype — Self-Supervised Learning for Gene Expression using DINOv3 framework + Nicheformer's transformer**

---

# 🚀 Overview

**Gene-DINO ** is a prototype framework that adapts the self-supervised learning paradigm of DINOv3 to transcriptomics gene expression data.

Instead of images, the model operates on **gene-token sequences** generated from gene expression matrices, enabling transformer-based representation learning directly on single-cell and spatial transcriptomics data.

The current implementation focuses on:

- ✅ Adapting DINOv3 teacher–student learning to gene expression
- ✅ Integrating the Nicheformer transformer backbone
- ✅ Multi-croping augmentation for transcriptomics data structure
- ✅ Multi-transform self-supervision
- ✅ Compatibility with real HESCAPE dataloader's output (TO-DO lale might will change that later as we use another data or scale)
- ✅ Preserving original DINOv3 core logic wherever possible

---

# 🧠 Core Idea

DINOv3 learns invariant representations from different augmented *views* of the same image.

Gene-DINO extends this idea to transcriptomics:

```text
Same Cell
   ↓
Different preprocessing transforms
   ↓
Different random gene crops
   ↓
Different “views” of the same biological state
   ↓
Teacher–Student self-supervised alignment
```

The model learns that:

- partial gene subsets
- differently normalized counts
- different preprocessing pipelines (See Figure 1)

…should still map to similar embeddings if they originate from the same biological cell.

<p align="center">
  <img src="./documentation/example_brainstorming.png" alt="Gene-DINO preprocessing views" width="560" />
</p>

---

# 🏗️ Architecture

## 🔹 Backbone

The backbone is based on **Nicheformer**, wrapped into a DINOv3-compatible interface.

### Backbone Configuration

| Component | Value |
|---|---|
| Transformer Layers | 12 |
| Hidden Dimension | 512 |
| Attention Heads | 16 |
| Feedforward Dimension | 1024 |
| Max Sequence Length | 1500 |
| Vocabulary Size | 25000 (random) / 20340 (pretrained) |

---

## 🔹 Input Format

```python
LongTensor [B, seq_len]
```

Where:
- `B` = batch size
- `seq_len` = gene token sequence length

---

## 🔹 Output Interface (DINOv3 Compatible)

```python
{
    "x_norm_clstoken":    [B, 512],
    "x_norm_patchtokens": [B, 1, 512],
    "x_storage_tokens":   [B, 0, 512]
}
```

This matches exactly what DINOv3 expects internally.

---

# 📁 Project Structure

```text
content/
├── documentation/
├── reference_files/
├── nicheformer/
├── hescape/
├── evaluation/
├── genexp_dino_walkthrough.ipynb
└── dinov3/
    ├── nicheformer_backbone.py
    ├── gene_collate.py
    ├── gene_dataset.py
    ├── train_gene.py
    └── inspect_pipeline.py
    └── ... (The rest of the dinov3 code unchanged)
```

--- 

# 📦 Files Explained

| File | Purpose |
|---|---|
| `nicheformer_backbone.py` | Wraps Nicheformer into a DINOv3-compatible backbone |
| `gene_collate.py` | Handles crop collation for transcriptomics |
| `gene_dataset.py` | Dataset + augmentation transforms |
| `train_gene.py` | Teacher–student DINO training loop |
| `inspect_pipeline.py` | Full pipeline verification/debugging |

---

# ✅ What Remains Original From DINOv3

The prototype intentionally preserves the original DINOv3 learning mechanics.

### Reused Directly

- `DINOHead`
- `DINOLoss`
- Sinkhorn-Knopp centering
- Teacher temperature sharpening
- EMA teacher update logic

No modifications were made to these components.

---

# ⚙️ Installation

## 1️⃣ Clone Repositories (DINOv3, Nicheformer and Hescape) 
---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔬 Augmentation Strategy

## 🧬 Gene Cropping

Instead of image crops, Gene-DINO performs **random gene subsampling**.

### Global Crop
- 1500 tokens
- Seen by teacher + student

### Local Crop
- 500 tokens
- Seen by student only

Each crop randomly selects different genes from the same cell.

---

# 🔄 Supported Transforms

Currently implemented:

| Transform | Description |
|---|---|
| `lognorm` | Log normalization |
| `normalize` | Count normalization |
| `nicheformer` | Nicheformer preprocessing |
| `scfoundation` | scFoundation preprocessing |


---

# 🧰 Running on Dummy Data

The entire pipeline can run immediately without any dataset.

## Random Initialization

```bash
python train_gene.py \
    --steps 10 \
    --batch_size 3 \
    --device cuda \
    --checkpoint none \
    --transform lognorm
```

---

## Pretrained Nicheformer Initialization

```bash
python train_gene.py \
    --steps 10 \
    --batch_size 3 \
    --device cuda \
    --checkpoint theislab/Nicheformer \
    --transform lognorm
```

---

## All Four Transforms

```bash
python train_gene.py \
    --steps 5 \
    --batch_size 1 \
    --device cuda \
    --transform lognorm normalize nicheformer scfoundation
```

---

# 🔍 Pipeline Verification

`inspect_pipeline.py` verifies:

- ✅ dataset logic
- ✅ transform correctness
- ✅ collate outputs
- ✅ backbone interface
- ✅ multi-transform compatibility

---

# 📊 Expected Shapes

## Single Cell

```python
global_crops : [8, 1500]
local_crops  : [32, 500]
```

---

## Batch Size = 3

```python
collated_global_crops : [24, 1500]
collated_local_crops  : [96, 500]
```



---


# ⚠️ Current Limitations

- Prototype stage only
- Single-GPU training
- No iBOT branch yet
- No distributed FSDP integration
- Crop sizes still under experimentation
- Multi-transform memory cost is high

---

# 📚 🙌 Acknowledgements

- DINOv3 — Meta AI
- Nicheformer — Theis Lab
- HESCAPE — Peng Lab

---

# Notes

This prototype is intentionally designed to:

- preserve original DINOv3 logic
- minimize invasive modifications
- validate the biological SSL hypothesis first
- prepare for scalable distributed training later

---

# 🧬 GENE-DINO (DINOmics STEP 1)

> “Different views of the same cell should encode the same biology.”