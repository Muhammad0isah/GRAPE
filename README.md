<div align="center">

# GRAPE

### **G**raph **R**etinal **A**nalysis for **P**rediction and **E**valuation

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-blue)](https://muhammad0isah.github.io/GRAPE/leaderboard.html) [![Dataset](https://img.shields.io/badge/Dataset-DRIVE_|_STARE_|_HRF-green)](#data-sources) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![CI](https://img.shields.io/badge/CI-Automated_Scoring-orange)](.github/workflows/score_submission.yml) [![Encryption](https://img.shields.io/badge/Submissions-RSA_Encrypted-red)](encryption/)

</div>

## Overview

**GRAPE** is a GNN benchmark competition for diabetic retinopathy classification from retinal vessel graphs.

**Task:** Binary graph classification (healthy vs DR)  
**Metric:** Macro F1 Score (leaderboard_score), AUROC  
**Data:** 70 retinal vessel graphs from DRIVE[<sup>1</sup>](#data-sources) + STARE[<sup>2</sup>](#data-sources) + HRF[<sup>3</sup>](#data-sources)  
**Leaderboard:** [muhammad0isah.github.io/GRAPE/leaderboard.html](https://muhammad0isah.github.io/GRAPE/leaderboard.html)

---

## Background

Diabetic retinopathy (DR) is the leading cause of blindness in working-age adults. Retinal blood vessels form natural graphs where bifurcation points are nodes and vessel segments are edges. Changes in vessel topology (branching patterns, tortuosity, connectivity) indicate disease progression. This competition benchmarks GNN methods on classifying these graphs as healthy or DR-positive.

---

## Data Sources

| Dataset | Graphs | Healthy | DR | Source |
|---------|--------|---------|-----|--------|
| DRIVE | 20 | 17 | 3 | [drive.grand-challenge.org](https://drive.grand-challenge.org/) |
| STARE | 20 | 16 | 4 | [cecas.clemson.edu/~ahoover/stare](https://cecas.clemson.edu/~ahoover/stare/) |
| HRF | 30 | 15 | 15 | [www5.cs.fau.de/research/data/fundus-images](https://www5.cs.fau.de/research/data/fundus-images/) |
| **Total** | **70** | **48** | **22** | |

**Split:** 55 train / 15 test (stratified).

**Graph ID prefixes:** `D_XX` = DRIVE, `S_XX` = STARE, `H_XX` = HRF healthy, `R_XX` = HRF DR.

---

## Dataset Challenges

- **Class imbalance** — 48 healthy vs 22 DR (~69%/31%)
- **Cross-domain shift** — three imaging sources with different resolutions and protocols
- **Variable graph sizes** — ~30 to 500+ nodes per graph
- **Noisy topology** — graph extraction from segmentation introduces structural noise

---

## Repository Structure

```
GRAPE/
├── data/
│   └── public/
│       ├── train_data.csv          # 55 graphs (nodes + edges)
│       ├── train_labels.csv        # training labels
│       ├── test_data.csv           # 15 graphs for prediction
│       └── sample_submission.csv   # expected output format
├── encryption/
│   ├── public_key.pem              # RSA public key (for encrypting submissions)
│   ├── encrypt.py                  # encryption script
│   └── decrypt.py                  # decryption (CI-only)
├── competition/
│   ├── evaluate.py                 # scoring script
│   ├── validate_submission.py      # format validation
│   └── metrics.py                  # macro F1, AUROC
├── baseline.py                     # GAT baseline model
├── submissions/
│   └── inbox/<team>/               # place your .enc file here
└── leaderboard/
    └── leaderboard.csv             # auto-updated scores
```

---

## Graph Specification (A, X)

Each graph is defined by a **node feature matrix X** and an **adjacency matrix A**.

### Node Feature Matrix X

Each node has 4 features:

| Feature | Column | Description |
|---------|--------|-------------|
| $x_1$ | `x` | horizontal coordinate (pixels) |
| $x_2$ | `y` | vertical coordinate (pixels) |
| $x_3$ | `width` | vessel width at the node |
| $x_4$ | `type` | junction or endpoint |

### Adjacency Matrix A

The `edges` column encodes adjacency. Each node lists its neighbors as semicolon-separated IDs. This defines an undirected, unweighted adjacency matrix:

If node 0 has `edges = "3;9;121"`, then $A_{0,3} = A_{0,9} = A_{0,121} = 1$.

### CSV Columns

**train_data.csv / test_data.csv:**

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier (e.g. `D_21`, `S_44`) |
| `node_id` | node index within graph |
| `x`, `y` | pixel coordinates |
| `width` | vessel width |
| `type` | `junction` or `endpoint` |
| `edges` | adjacent node IDs (semicolon-separated) |

**train_labels.csv:**

| Column | Description |
|--------|-------------|
| `graph_id` | graph identifier |
| `label` | `0` = healthy, `1` = diabetic retinopathy |

---

## How to Participate (Step by Step)

### Step 1: Clone the Repository

```bash
git clone https://github.com/muhammad0isah/GRAPE.git
cd GRAPE
```

### Step 2: Install Dependencies

```bash
pip install pandas scikit-learn cryptography torch torch-geometric
```

### Step 3: Train Your Model and Generate Predictions

Use `data/public/train_data.csv` and `data/public/train_labels.csv` to train a GNN, then predict labels for each graph in `data/public/test_data.csv`.

A baseline GAT model is provided. Running it trains the model and generates `submission.csv` automatically:

```bash
python baseline.py
```

This outputs a file called `submission.csv` in the project root with the required format:

```csv
graph_id,label
D_25,0
R_2,1
S_235,0
H_10,0
...
```

You can build your own model — just make sure the output CSV has exactly these two columns, includes all 15 test graph IDs, and labels are `0` or `1`.

### Step 4: Encrypt Your Predictions

Submissions are encrypted so that other participants cannot see your predictions.

```bash
mkdir -p submissions/inbox/YOUR_TEAM_NAME
python encryption/encrypt.py submission.csv submissions/inbox/YOUR_TEAM_NAME/submission.csv.enc
```

Replace `YOUR_TEAM_NAME` with your team name (no spaces, use underscores).

Add a `meta.yaml` in your team folder to display your model name and notes on the leaderboard:

```yaml
model: VesselGCN                          # Name of your model (optional)
type: human                               # human, llm, or human+llm
notes: 3-layer GCN with skip connections  # Brief description (optional)
```

Example folder structure:

```
submissions/inbox/<team>/
├── submission.csv.enc   # Required (encrypted predictions)
└── meta.yaml            # Describes your submission
```

### Step 5: Fork, Commit, and Open a Pull Request

```bash
# Fork this repo on GitHub first, then:
git checkout -b submission/YOUR_TEAM_NAME
git add submissions/inbox/YOUR_TEAM_NAME/
git commit -m "Submission: YOUR_TEAM_NAME"
git push origin submission/YOUR_TEAM_NAME
```

Then open a Pull Request from your fork to the main repository.

### Step 6: Wait for Automated Scoring

The CI pipeline will automatically:
1. Decrypt your `.enc` file using the organizer's private key
2. Validate the submission format
3. Score against the hidden test labels
4. Report your Macro F1 and AUROC in the PR

Scores are published on the [leaderboard](https://muhammad0isah.github.io/GRAPE/leaderboard.html) after the PR is merged.

---

## Rules

- **One submission per team.** Only your first submission is scored.
- **Predictions must be encrypted.** Raw `.csv` files are gitignored and will not be accepted.
- **Training must complete within 3 hours on CPU.**
- **No access to test labels.** They are stored as a GitHub Secret and never exposed.

---

## Evaluation

Submissions are ranked by **Macro F1 Score** on the hidden test set. AUROC is reported as a secondary metric. Tied scores share the same rank.

---

## Baseline

The provided baseline (`baseline.py`) uses a 3-layer GAT with multi-head attention, multi-pool readout, and graph-level topological features. It achieves a Macro F1 of **0.830**.

Dependencies: `torch`, `torch-geometric`, `pandas`, `numpy`.

---

## License

MIT

---

## Citation

```bibtex
@misc{grape_2025,
  title={GRAPE:Graph Retinal Analysis for Prediction and Evaluation},
  author={Muhammad Ibrahim Isah},
  year={2026},
  url={https://github.com/Muhammad0isah/GRAPE}
}
```