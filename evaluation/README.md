# Video Composition Evaluation

This project evaluates **compositional generalization** on video embeddings from the **Something-Something V2** dataset using factorization methods (GDE and LDE). It tests whether video models can understand and combine verbs (actions) and objects in novel ways by decomposing video embeddings into compositional components.

## Project Overview

The project trains factorizers (GDE/LDE) on DiST video embeddings to learn verb-object decompositions, then evaluates how well these learned components can be recombined to recognize novel verb-object compositions in test videos.

**Key Insight**: If a model truly understands compositionality, it should be able to combine learned components (e.g., "pushing" + "bottle") to recognize actions it has never seen before.

## Project Structure

```
video_compositionality_project/
├── data/
│   └── sth-sth-v2/
│       ├── train_pairs.json       # Training video annotations (38,034 videos)
│       ├── val_pairs.json         # Validation annotations (18,774 videos)
│       └── test_pairs.json        # Test annotations (22,657 videos)
├── embeddings/
│   ├── dist_train.pt              # DiST embeddings for training (38,034 x 512)
│   ├── dist_val.pt                # DiST embeddings for validation (18,774 x 512)
│   └── dist_test.pt               # DiST embeddings for test (22,657 x 512)
├── models/
│   ├── DiST_model/
│   │   ├── checkpoints/
│   │   │   └── dist_vit-b16_32+64f_ssv2_70.9.pth  # Pre-trained DiST model
│   │   ├── configs/                # Model configuration files
│   │   ├── extract_embeddings.py   # Script to extract video embeddings
│   │   └── README.md               # DiST model documentation
│   └── InternVideo2-Stage2_6B-224p-f4/
│       ├── README.md               # InternVideo2 model documentation
│       ├── configs/                # Model configuration files
│       ├── data/                   # Dataset configs or metadata
│       └── checkpoints/            # Pre-trained InternVideo2 checkpoints
├── factorizers/
│   ├── factorizers.py             # GDE and LDE implementations
│   └── sphere.py                  # Riemannian geometry utilities for GDE
├── evaluation/
│   ├── evaluate_video_composition.py  # Main evaluation script
│   └── README.md                      # This file
└── results/
    ├── gde_val_closed.json        # Results files (auto-generated)
    ├── gde_val_open.json
    ├── lde_test_closed.json
    └── ...
```

## Data Structure

### Video Embeddings
Embeddings are stored as `.pt` files containing dictionaries:
```python
{
    'video_id': torch.Tensor([512]),  # e.g., '80715': tensor of 512 dimensions
    ...
}
```

Supported naming: `dist_train.pt`, `dist_val.pt`, `dist_test.pt` or `internvideo2_train.pt`, etc.

### Pair Annotations
JSON files with format:
```json
[
    {
        "id": "80715",
        "action": "lifting up one end of bottle, then letting it drop down",
        "verb": "Lifting up one end of [something], then letting it drop down",
        "object": "bottle"
    },
    ...
]
```

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install torch numpy scipy tqdm
```

### Setup

1. **Ensure your data structure matches:**
   ```
   data/sth-sth-v2/{train,val,test}_pairs.json
   embeddings/dist_{train,val,test}.pt
   ```

2. **Navigate to project directory:**
   ```bash
   cd /path/to/video_compositionality_project
   ```

### Run Evaluation

#### Quick Test (without AUC, faster)
```bash
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name GDE \
    --test_phase val
```

#### Complete Evaluation Commands

**GDE Method:**
```bash
# Validation - Closed World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name GDE \
    --test_phase val \
    --result_path ./results/gde_val_closed.json \
    --compute_auc

# Validation - Open World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name GDE \
    --test_phase val \
    --open_world \
    --result_path ./results/gde_val_open.json \
    --compute_auc

# Test - Closed World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name GDE \
    --test_phase test \
    --result_path ./results/gde_test_closed.json \
    --compute_auc
```

**LDE Method:**
```bash
# Validation - Closed World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name LDE \
    --test_phase val \
    --result_path ./results/lde_val_closed.json \
    --compute_auc

# Validation - Open World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name LDE \
    --test_phase val \
    --open_world \
    --result_path ./results/lde_val_open.json \
    --compute_auc

# Test - Closed World
python3 evaluation/evaluate_video_composition.py \
    --data_dir ./data \
    --embeddings_dir ./embeddings \
    --experiment_name LDE \
    --test_phase test \
    --result_path ./results/lde_test_closed.json \
    --compute_auc
```

#### Run All Evaluations (Batch Script)
```bash
# Create and run a batch script
cat > run_all_evals.sh << 'EOF'
#!/bin/bash
methods=("GDE" "LDE")
phases=("val" "test")
scenarios=("closed" "open")

for method in "${methods[@]}"; do
  for phase in "${phases[@]}"; do
    for scenario in "${scenarios[@]}"; do
      if [ "$scenario" == "closed" ]; then
        open_flag=""
      else
        open_flag="--open_world"
      fi
      
      echo "Running $method - $phase - $scenario"
      python3 evaluation/evaluate_video_composition.py \
        --data_dir ./data \
        --embeddings_dir ./embeddings \
        --experiment_name $method \
        --test_phase $phase \
        $open_flag \
        --result_path ./results/${method,,}_${phase}_${scenario}.json \
        --compute_auc
    done
  done
done
EOF

chmod +x run_all_evals.sh
./run_all_evals.sh
```

### View Results
```bash
# View all results
cat results/*.json | python3 -m json.tool

# View specific result
cat results/gde_val_closed.json

# Compare results
python3 << 'EOF'
import json
import glob

for file in sorted(glob.glob('results/*.json')):
    with open(file) as f:
        data = json.load(f)
        config = data['config']
        result = data['result']
        print(f"\n{file}:")
        print(f"  Method: {config['experiment_name']}, Phase: {config['test_phase']}, Open: {config.get('open_world', False)}")
        print(f"  Pair Acc: {result['unbiased_pair_acc']:.4f}")
        print(f"  Verb Acc: {result['attr_acc']:.4f}")
        print(f"  Object Acc: {result['obj_acc']:.4f}")
        if 'best_harmonic_mean' in result:
            print(f"  Harmonic Mean: {result['best_harmonic_mean']:.4f}")
EOF
```

## Usage

- `--data_dir`: Path to data directory containing `sth-sth-v2/` folder (default: `./data`)
- `--embeddings_dir`: Path to embeddings directory with `.pt` files (default: `./embeddings`)
- `--experiment_name`: Factorizer method: `GDE` or `LDE` (required)
- `--test_phase`: Evaluation phase: `val` or `test` (default: `test`)
- `--open_world`: Use open world evaluation (allows all possible pairs)
- `--compute_auc`: Compute AUC metric (slower but comprehensive)
- `--result_path`: Path to save results JSON file
- `--n_exp`: Number of experiment repetitions (default: 1)

## How It Works

1. **Load Video Embeddings**: Loads pre-computed video features from DiST or InternVideo2 models
   ```python
   # Embeddings: {video_id: tensor([512]), ...}
   train_embs = torch.load('embeddings/dist_train.pt')
   ```

2. **Train Factorizer**: Learns to decompose video embeddings into verb + object components
   ```python
   factorizer = GDE(train_embeddings, train_pairs, weights=None)
   ```

3. **Compose Embeddings**: Creates synthetic embeddings for all verb-object combinations
   ```python
   composed = factorizer.compute_ideal_words_approximation(test_pairs)
   ```

4. **Evaluate**: Measures how well composed embeddings match real test videos
   ```python
   similarity = video_embeddings @ composed_embeddings.T
   accuracy = compute_metrics(similarity, ground_truth)
   ```

## Metrics

- **Unbiased Pair Acc**: Accuracy without bias correction
- **Verb (Attr) Acc**: Verb recognition accuracy  
- **Object Acc**: Object recognition accuracy
- **Best Seen Acc**: Maximum accuracy on seen training pairs
- **Best Unseen Acc**: Maximum accuracy on novel unseen pairs
- **Harmonic Mean**: Harmonic mean of seen and unseen accuracy
- **AUC**: Area under the seen-unseen accuracy curve

## Results Structure

Results are saved as JSON files:
```json
{
    "config": {
        "experiment_name": "GDE",
        "test_phase": "val",
        "open_world": false,
        ...
    },
    "result": {
        "unbiased_pair_acc": 0.1184,
        "attr_acc": 0.8490,
        "obj_acc": 0.1281,
        "best_seen_acc": 0.1504,
        "best_unseen_acc": 0.4362,
        "best_harmonic_mean": 0.1482,
        "auc": 0.0444
    }
}
```

## Dataset Statistics

- **5,124** unique verb-object pairs
- **161** unique verbs (actions)
- **248** unique objects
- **38,034** training videos
- **18,774** validation videos
- **22,657** test videos

## Dependencies

- PyTorch
- NumPy
- SciPy
- tqdm

The factorizers (GDE/LDE) are imported from `factorizers/factorizers.py`.
