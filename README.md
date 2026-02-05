# Video Compositional Understanding with GDE/LDE

This project evaluates **compositional generalization** for video understanding using **Generalized Decomposition of Embeddings (GDE)** and **Linear Decomposition of Embeddings (LDE)** on the **Something-Something V2** dataset.

---

## 📋 Overview

The goal is to test whether video models can decompose and recombine actions (verbs) and objects in novel ways. For example, if a model learns from "pushing box" and "lifting chair", can it recognize "pushing chair" (an unseen combination)?

### Key Features

- **Two Video Encoders**: DiST and InternVideo2 for extracting video embeddings
- **Two Factorization Methods**: GDE (Riemannian geometry) and LDE (Euclidean)
- **Compositional Evaluation**: Tests on seen and unseen verb-object combinations
- **Zero-Shot Generalization**: Measures performance on novel compositions

---

## 🚀 QUICK START (Step-by-Step for Beginners)

> **If embeddings are already extracted (check `embeddings/` folder), skip to Step 4!**

### Step 0: Check Your Current Location

```bash
# First, always make sure you're in the right place
pwd
# Should show something like: /scratch/rpp/vlm_image_compositionality
```

### Step 1: Navigate to Project Directory

```bash
cd /scratch/rpp/vlm_image_compositionality/video_compositionality_project
```

### Step 2: Activate Python Environment

```bash
source /scratch/rpp/vlm_image_compositionality/bin/activate
```

You should see `(vlm_image_compositionality)` at the start of your terminal prompt.

### Step 3: Check if Embeddings Already Exist

```bash
ls -la embeddings/
```

**If you see these files, skip to Step 4:**
```
dist_train.pt
dist_val.pt  
dist_test.pt
```

**If files are missing**, you need to extract embeddings first (see Section "Extract Video Embeddings" below).

> **Current Status:** DiST embeddings are complete (38,034 train / 18,774 val / 22,657 test videos).  
> InternVideo2 embeddings need to be extracted.

### Step 4: Run Evaluation (The Main Test!)

```bash
# Navigate to evaluation folder
cd /scratch/rpp/vlm_image_compositionality/video_compositionality_project/evaluation

# Run GDE evaluation on test set
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase test \
    --result_path ../results/gde_test_results.json
```

### Step 5: View Your Results

```bash
cat ../results/gde_test_results.json
```

**Congratulations! 🎉 You ran the evaluation!**

---

## 📁 Project Structure

```
video_compositionality_project/
├── README.md                          # This file (you are here!)
├── requirements.txt                   # Python packages needed
│
├── data/
│   └── sth-sth-v2/                   # Something-Something V2 dataset
│       ├── train_pairs.json          # 38,034 training annotations
│       ├── val_pairs.json            # 18,774 validation annotations
│       ├── test_pairs.json           # 22,657 test annotations
│       └── s2s clips/                # Video files (.webm format)
│           ├── 142241.webm
│           ├── 50520.webm
│           └── ...
│
├── embeddings/                        # Pre-computed video embeddings
│   ├── dist_train.pt                 # DiST embeddings (512-dim each)
│   ├── dist_val.pt
│   ├── dist_test.pt
│   ├── internvideo2_train.pt         # InternVideo2 embeddings (768-dim each)
│   ├── internvideo2_val.pt
│   └── internvideo2_test.pt
│
├── factorizers/                       # Decomposition algorithms
│   ├── factorizers.py                # GDE & LDE code
│   └── sphere.py                     # Math for Riemannian geometry
│
├── evaluation/                        # Evaluation scripts
│   └── evaluate_video_composition.py # <-- THE MAIN SCRIPT YOU RUN
│
├── models/                           # Video embedding extraction
│   └── internvideo2/
│       └── extract_internvideo2_final.py
│
└── results/                          # Your results go here
    └── *.json
```

---

## 🔧 Full Installation (If Starting Fresh)

### Step 1: Go to Workspace

```bash
cd /scratch/rpp/vlm_image_compositionality
```

### Step 2: Create Virtual Environment (One Time Only)

```bash
python3 -m venv vlm_env
source vlm_env/bin/activate
```

### Step 3: Install All Packages

```bash
cd video_compositionality_project
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Step 4: Verify Installation

```bash
python -c "import torch; import clip; print('✓ PyTorch:', torch.__version__); print('✓ CLIP installed')"
```

---

## 🎬 Extract Video Embeddings (Only If Not Already Done!)

> **Check first:** `ls embeddings/` - if you see `.pt` files, embeddings are ready!

### Option A: DiST Model (Recommended - Faster)

**DiST** uses CLIP ViT-B/16 + 3D Conv. Output: 512-dim embeddings.

```bash
# Step 1: Go to DiST folder
cd /scratch/rpp/vlm_image_compositionality/DiST_model

# Step 2: Activate environment
source /scratch/rpp/vlm_image_compositionality/bin/activate

# Step 3: Run extraction script
# NOTE: This script processes ALL splits automatically (no command-line arguments)
# It will skip splits that already have output files and resume from checkpoints
python extract_embeddings.py

# The script automatically:
# - Processes val and test splits (train already complete)
# - Saves checkpoints every 100 videos (resumable if interrupted)
# - Outputs: s2s_train_dist.pt, s2s_val_dist.pt, s2s_test_dist.pt

# Step 4: Copy embeddings to project folder
cp embeddings/s2s_train_dist.pt ../video_compositionality_project/embeddings/dist_train.pt
cp embeddings/s2s_val_dist.pt ../video_compositionality_project/embeddings/dist_val.pt
cp embeddings/s2s_test_dist.pt ../video_compositionality_project/embeddings/dist_test.pt

# Step 5: Verify
ls -la ../video_compositionality_project/embeddings/dist_*.pt
```

**Expected Processing Time:**
- Train: ~2-3 hours (38,034 videos)
- Val: ~1-2 hours (18,774 videos) 
- Test: ~1.5-2 hours (22,657 videos)

### Option B: InternVideo2 Model (Better Quality - Slower)

**InternVideo2** is a 6B parameter model. Output: 768-dim embeddings.

```bash
# Step 1: Go to InternVideo2 folder
cd /scratch/rpp/vlm_image_compositionality/video_compositionality_project/models/internvideo2

# Step 2: Activate environment
source /scratch/rpp/vlm_image_compositionality/bin/activate

# Step 3: Extract embeddings (specify split and device)
# Extract each split individually:
python extract_internvideo2_final.py --split train --device cuda
python extract_internvideo2_final.py --split val --device cuda  
python extract_internvideo2_final.py --split test --device cuda

# OR extract all splits at once:
python extract_internvideo2_final.py --split all --device cuda

# For CPU usage (slower but works without GPU):
python extract_internvideo2_final.py --split train --device cpu

# Step 4: Verify embeddings are created
ls -la ../../embeddings/internvideo2_*.pt
```

**Command Options:**
| Argument | Options | Description |
|----------|---------|-------------|
| `--split` | `train`, `val`, `test`, `all` | Which split(s) to process |
| `--device` | `cuda`, `cpu` | Use GPU or CPU (cuda recommended) |

**Expected Processing Time (on GPU):**
- Train: ~3-4 hours (38,034 videos)
- Val: ~2-3 hours (18,774 videos)
- Test: ~2-3 hours (22,657 videos)

**Requirements:**
- GPU Memory: ~15GB for 6B model
- CPU Alternative: Use `--device cpu` (much slower)
- Saves checkpoints every 100 videos (resumable)

---

## 📊 Running Evaluation (Complete Guide)

### Navigate to Evaluation Folder

```bash
cd /scratch/rpp/vlm_image_compositionality/video_compositionality_project/evaluation
```

### Test 1: GDE on Test Set (Closed World)

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase test \
    --result_path ../results/gde_test_closed.json
```

### Test 2: LDE on Test Set (Closed World)

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name LDE \
    --test_phase test \
    --result_path ../results/lde_test_closed.json
```

### Test 3: GDE on Validation Set

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase val \
    --result_path ../results/gde_val_closed.json
```

### Test 4: Open World Evaluation

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase test \
    --open_world \
    --result_path ../results/gde_test_open.json
```

### Test 5: With AUC Metric (Slower but More Complete)

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase test \
    --compute_auc \
    --result_path ../results/gde_test_with_auc.json
```

### Test 6: Multiple Runs for Statistics

```bash
python evaluate_video_composition.py \
    --data_dir ../data \
    --embeddings_dir ../embeddings \
    --experiment_name GDE \
    --test_phase test \
    --n_exp 3 \
    --result_path ../results/gde_test_3runs.json
```

---

## 📋 All Command Arguments Explained

| Argument | What It Does | Options | Default |
|----------|--------------|---------|---------|
| `--data_dir` | Where your JSON files are | Path | `./data` |
| `--embeddings_dir` | Where your .pt files are | Path | `./embeddings` |
| `--experiment_name` | Which method to use | `GDE` or `LDE` | `GDE` |
| `--test_phase` | Which set to evaluate | `test` or `val` | `test` |
| `--open_world` | Allow all pair predictions | Flag (no value) | Off |
| `--compute_auc` | Calculate AUC metric | Flag (no value) | Off |
| `--n_exp` | Number of runs | Integer | `1` |
| `--result_path` | Where to save results | Path | None |

---

## 📈 Understanding Results

### Output Metrics

The evaluation script outputs the following metrics:

```json
{
  "unbiased_pair_acc": 0.4523,      // Overall pair accuracy (no bias)
  "attr_acc": 0.6821,                // Verb accuracy (action recognition)
  "obj_acc": 0.7134,                 // Object accuracy (object recognition)
  "best_seen_acc": 0.8945,           // Best accuracy on seen compositions
  "best_unseen_acc": 0.2341,         // Best accuracy on unseen compositions
  "best_harmonic_mean": 0.3678,      // Harmonic mean of seen/unseen
  "auc": 0.1234                      // Area under seen-unseen curve (if --compute_auc)
}
```

### Key Metrics Explained

1. **Pair Accuracy**: Correct prediction of both verb AND object
2. **Attribute (Verb) Accuracy**: Correct action prediction (ignoring object)
3. **Object Accuracy**: Correct object prediction (ignoring action)
4. **Seen Accuracy**: Performance on training compositions
5. **Unseen Accuracy**: **Zero-shot performance** on novel compositions
6. **Harmonic Mean**: Balance between seen and unseen performance
7. **AUC**: Trade-off between seen and unseen accuracy

### Interpreting Results

**Good Compositional Understanding:**
- High unseen accuracy (>30%)
- Balanced seen/unseen performance
- High harmonic mean

**Memorization (Poor Compositionality):**
- High seen accuracy (>80%)
- Low unseen accuracy (<10%)
- Low harmonic mean

---

## 🧮 How It Works

### 1. Data Format

**Annotation Format:**
```json
{
  "id": "142241",
  "action": "approaching chair with your camera",
  "verb": "Approaching [something] with your camera",
  "object": "chair"
}
```

**Embedding Format:**
```python
# dist_train.pt
{
    "142241": tensor([0.123, -0.456, ..., 0.789]),  # 512-dim for DiST
    "50520": tensor([0.234, 0.567, ..., -0.123]),
    ...
}
```

### 2. Training Phase

**Input:** Training video embeddings + (verb, object) labels

**Process:**
1. **Compute global mean** μ on unit sphere (Riemannian mean)
2. **Project to tangent space** at μ using logarithmic map
3. **Compute verb primitives**: Mean embedding per verb
4. **Compute object primitives**: Mean embedding per object
5. **Center primitives**: Subtract global mean

**Output:** Learned verb and object primitives in tangent space

### 3. Test Phase

**For each test video:**
1. **Compose all pairs**: verb_primitive + object_primitive
2. **Map to sphere**: Exponential map from tangent space
3. **Compute similarities**: Cosine similarity with test embedding
4. **Predict**: Pair with highest similarity

### 4. Example

**Training:**
- Video 1: "Pushing box" → z₁
- Video 2: "Lifting chair" → z₂

**Learn:**
- Verb primitive: "Pushing" = direction in tangent space
- Object primitive: "box" = direction in tangent space

**Test on unseen "Pushing chair":**
- Compose: v_pushing + v_chair
- Compare with test video embedding
- Predict if similarity is high!

---

## 🔬 Methods

### GDE (Generalized Decomposition of Embeddings)

**Approach:** Riemannian geometry on unit hypersphere

**Steps:**
1. Compute intrinsic (Fréchet) mean on sphere
2. Logarithmic map to tangent space
3. Linear decomposition in tangent space
4. Exponential map back to sphere

**Advantages:**
- Respects geometry of normalized embeddings
- Better for CLIP/DiST embeddings (unit norm)
- More principled composition

**Implementation:** `factorizers/factorizers.py` (GDE class)

---

### LDE (Linear Decomposition of Embeddings)

**Approach:** Euclidean space decomposition

**Steps:**
1. Compute arithmetic mean
2. Linear combination: z ≈ μ + v_verb + v_object
3. Direct composition in embedding space

**Advantages:**
- Simpler implementation
- Faster computation
- Baseline comparison

**Implementation:** `factorizers/factorizers.py` (LDE class)

---

## 📊 Expected Performance

### Typical Results on Something-Something V2

| Method | Pair Acc | Verb Acc | Object Acc | Unseen Acc |
|--------|----------|----------|------------|------------|
| **GDE** | 45-50% | 65-70% | 70-75% | 25-35% |
| **LDE** | 40-45% | 60-65% | 65-70% | 20-30% |

**Notes:**
- Random baseline: ~0.1% (thousands of possible pairs)
- Seen accuracy typically: 80-90%
- Unseen accuracy: 20-35% (zero-shot!)

---

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory (GPU)**
```bash
# Reduce batch size for InternVideo2
python extract_internvideo2_final.py --split train --device cuda --batch_size 1

# Or use CPU (slower)
python extract_internvideo2_final.py --split train --device cpu
```

**2. Video Files Not Found**
```bash
# Check video path
ls data/sth-sth-v2/s2s\ clips/ | head -5

# Update path in extraction scripts if needed
VIDEO_FOLDER = "/path/to/your/videos"
```

**3. Missing Checkpoints**
```bash
# DiST checkpoint
mkdir -p /scratch/rpp/vlm_image_compositionality/DiST_model/checkpoints
# Download and place: dist_k710_vit_b_16.pth

# InternVideo2 checkpoint
mkdir -p models/internvideo2/checkpoints
# Download and place: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt
```

**4. Import Errors**
```bash
# Make sure you're in the correct directory
cd video_compositionality_project

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting (Common Problems & Solutions)

### Problem 1: "ModuleNotFoundError: No module named 'factorizers'"

**Solution:**
```bash
# Make sure you're running from the correct directory
cd /scratch/rpp/vlm_image_compositionality/video_compositionality_project/evaluation
python evaluate_video_composition.py ...
```

### Problem 2: "FileNotFoundError: embeddings not found"

**Solution:**
```bash
# Check if embeddings exist
ls -la ../embeddings/

# If missing, either:
# A) Extract them (see "Extract Video Embeddings" section)
# B) Make sure path is correct in command:
python evaluate_video_composition.py --embeddings_dir ../embeddings ...
```

### Problem 3: "CUDA out of memory"

**Solution:**
```bash
# Use CPU instead (slower but works)
python extract_internvideo2_final.py --split train --device cpu

# Or reduce batch size
python extract_internvideo2_final.py --split train --device cuda --batch_size 1
```

### Problem 4: "No such file or directory: train_pairs.json"

**Solution:**
```bash
# Check the data directory structure
ls ../data/sth-sth-v2/

# IMPORTANT: The script automatically adds "sth-sth-v2" to the path!
# So use --data_dir ../data (NOT ../data/sth-sth-v2)
python evaluate_video_composition.py --data_dir ../data ...
```

### Problem 5: "Permission denied"

**Solution:**
```bash
# Make scripts executable
chmod +x extract_embeddings.py
chmod +x evaluate_video_composition.py
```

### Problem 6: Python version issues

**Solution:**
```bash
# Check Python version (need 3.8+)
python --version

# Use specific Python
python3.12 evaluate_video_composition.py ...
```

---

## ✅ Checklist Before Running

Use this checklist to make sure everything is ready:

- [ ] **Environment activated:** `source /scratch/rpp/vlm_image_compositionality/bin/activate`
- [ ] **In correct directory:** `cd .../video_compositionality_project/evaluation`
- [ ] **Embeddings exist:** `ls ../embeddings/*.pt`
- [ ] **Data files exist:** `ls ../data/sth-sth-v2/*.json`
- [ ] **Results folder exists:** `mkdir -p ../results`

---

---

## 📚 References

### Papers

1. **GDE Method**: Generalized Decomposition of Embeddings for Compositional Learning
2. **DiST Model**: "Disentangling Spatial and Temporal Learning for Efficient Image-to-Video Transfer Learning"
3. **InternVideo2**: "InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding"
4. **Something-Something V2**: "The 'Something Something' Video Database for Learning and Evaluating Visual Common Sense"

### Links

- Something-Something V2: https://developer.qualcomm.com/software/ai-datasets/something-something
- CLIP: https://github.com/openai/CLIP
- InternVideo2: https://github.com/OpenGVLab/InternVideo

---

## 📧 Contact

For questions or issues, open an issue on GitHub.

---

## 📄 License

This project is licensed under the MIT License.
