"""
Video Composition Evaluation Script
Evaluates video embeddings using compositional factorizers (GDE/LDE)
Adapted from the original image composition evaluation code for video data
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
from math import exp
from collections import defaultdict
from scipy.stats import hmean

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factorizers.factorizers import GDE, LDE

# Mapping of factorizer names to classes
FACTORIZERS = {
    'GDE': GDE,
    'LDE': LDE,
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VideoCompositionDataset:
    """
    Dataset class for Something-Something V2 video embeddings.
    
    Video embeddings are stored as .pt files containing dictionaries:
        {video_id: embedding_tensor, ...}
    where video_id is a string (e.g., '80715') and embedding is a torch.Tensor of shape [512]
    
    Pair annotations are JSON files with format:
        [{'id': '80715', 'action': '...', 'verb': '...', 'object': '...'}, ...]
    """
    
    def __init__(self, data_path, embeddings_path, phase='train', open_world=False):
        """
        Args:
            data_path: Path to data directory containing {train,val,test}_pairs.json
            embeddings_path: Path to embeddings directory containing dist_{train,val,test}.pt
            phase: 'train', 'val', or 'test'
            open_world: Whether to use open world evaluation (allows all pairs at test time)
        """
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.phase = phase
        self.open_world = open_world
        
        # Load pair annotations for all splits
        # Each entry: {'id': str, 'action': str, 'verb': str, 'object': str}
        self.train_pairs_data = self._load_pairs('train')
        self.val_pairs_data = self._load_pairs('val')
        self.test_pairs_data = self._load_pairs('test')
        
        # Extract (verb, object) pairs for compositional evaluation
        # Using verb as "attribute" and object as "object" to match original framework
        self.train_pairs = [(d['verb'], d['object']) for d in self.train_pairs_data]
        self.val_pairs = [(d['verb'], d['object']) for d in self.val_pairs_data]
        self.test_pairs = [(d['verb'], d['object']) for d in self.test_pairs_data]
        
        # Get all unique pairs across splits
        all_pairs = list(set(self.train_pairs + self.val_pairs + self.test_pairs))
        self.pairs = sorted(all_pairs)
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        
        # Get unique verbs (treated as "attributes") and objects
        all_verbs = sorted(set(p[0] for p in self.pairs))
        all_objs = sorted(set(p[1] for p in self.pairs))
        
        self.attrs = all_verbs  # verbs are "attributes" in composition
        self.objs = all_objs
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        
        # Load video embeddings from .pt files
        # Format: {video_id_str: torch.Tensor([512]), ...}
        self.train_embeddings = self._load_embeddings('train')
        self.val_embeddings = self._load_embeddings('val')
        self.test_embeddings = self._load_embeddings('test')
        
        # Build video ID to (verb, object) pair mapping
        self.train_id2pair = {d['id']: (d['verb'], d['object']) for d in self.train_pairs_data}
        self.val_id2pair = {d['id']: (d['verb'], d['object']) for d in self.val_pairs_data}
        self.test_id2pair = {d['id']: (d['verb'], d['object']) for d in self.test_pairs_data}
    
    def _load_pairs(self, split):
        """Load pair annotations from JSON file."""
        json_path = os.path.join(self.data_path, f'{split}_pairs.json')
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _load_embeddings(self, split):
        """
        Load video embeddings from .pt file.
        Returns a dictionary: {video_id_str: embedding_tensor, ...}
        """
        # Try different embedding file naming conventions
        for emb_name in [f'dist_{split}.pt', f'internvideo2_{split}.pt']:
            emb_path = os.path.join(self.embeddings_path, emb_name)
            if os.path.exists(emb_path):
                emb_dict = torch.load(emb_path)
                # Ensure keys are strings for consistent access
                return {str(k): v for k, v in emb_dict.items()}
        raise FileNotFoundError(f"No embedding file found for split: {split}")
    
    def load_all_image_embs(self):
        """
        Load all video embeddings for current phase (mimics original image dataset API).
        
        Returns:
            embeddings: torch.Tensor of shape [N, D] where N is number of videos
            all_pairs: list of (verb, object) tuples corresponding to each embedding
        """
        if self.phase == 'train':
            emb_dict = self.train_embeddings
            id2pair = self.train_id2pair
        elif self.phase == 'val':
            emb_dict = self.val_embeddings
            id2pair = self.val_id2pair
        else:  # test
            emb_dict = self.test_embeddings
            id2pair = self.test_id2pair
        
        # Extract embeddings and pairs in consistent order
        embeddings = []
        all_pairs = []
        
        for vid_id in sorted(emb_dict.keys(), key=int):  # Sort by numeric video ID
            if vid_id in id2pair:
                embeddings.append(emb_dict[vid_id])
                all_pairs.append(id2pair[vid_id])
        
        embeddings = torch.stack(embeddings)
        return embeddings, all_pairs
    
    def __str__(self):
        return (f"VideoCompositionDataset(phase={self.phase}, "
                f"n_pairs={len(self.pairs)}, n_verbs={len(self.attrs)}, n_objects={len(self.objs)}, "
                f"n_train={len(self.train_pairs_data)}, n_val={len(self.val_pairs_data)}, "
                f"n_test={len(self.test_pairs_data)})")


class Evaluator:
    """
    Evaluation class adapted from the original code.
    """
    def __init__(self, dset: VideoCompositionDataset):
        self.dset = dset

        if dset.phase == 'train':
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)
        
        # Labels in closed world scenario
        if not dset.open_world:
            self.closed_mask = torch.BoolTensor(
                [1 if pair in test_pair_set else 0 for pair in dset.pairs]
            )

        # Mask of seen concepts
        self.seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in self.seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Pairs as (attr_idx, obj_idx)
        self.pair_idx2ao_idx = torch.LongTensor([
            (dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs
        ])

    def get_attr_obj_from_pairs(self, pairs):
        attrs = self.pair_idx2ao_idx[pairs, 0]
        objs = self.pair_idx2ao_idx[pairs, 1]
        return attrs, objs

    def evaluate(self, y_pred_topk, y_true, seen_ids, unseen_ids):
        """Evaluate predictions."""
        correct = torch.eq(y_pred_topk, y_true.unsqueeze(1)).any(1).numpy()
        all_acc = np.mean(correct)
        seen_acc = np.mean(correct[seen_ids]) if len(seen_ids) > 0 else 0.0
        unseen_acc = np.mean(correct[unseen_ids]) if len(unseen_ids) > 0 else 0.0
        
        # Handle edge case where one of them is 0
        if seen_acc == 0 or unseen_acc == 0:
            harmonic = 0.0
        else:
            harmonic = hmean([seen_acc, unseen_acc])
            
        return {
            "all_acc": all_acc,
            "seen_acc": seen_acc,
            "unseen_acc": unseen_acc,
            "harmonic_mean": harmonic,
            "macro_average_acc": (seen_acc + unseen_acc) * 0.5,
        }
    
    def predict(self, scores, topk, bias=0.0):
        """Generate predictions from biased scores."""
        scores = scores.clone()
        scores[:, ~self.seen_mask] += bias

        # If closed world, exclude labels not in test_pair_set
        if not self.dset.open_world:
            scores[:, ~self.closed_mask] = -1e10

        _, pair_preds = scores.topk(topk, dim=1)
        attr_preds, obj_preds = self.get_attr_obj_from_pairs(pair_preds)

        return pair_preds, attr_preds, obj_preds
    
    def get_overall_metrics(self, features, all_pairs_true, topk_list=[1], progress_bar=True):
        """Compute comprehensive metrics including AUC."""
        labels = torch.LongTensor(
            [self.dset.pair2idx[pair] for pair in all_pairs_true]
        )

        # Seen/unseen samples
        seen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] in self.seen_pair_set
        ]
        unseen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] not in self.seen_pair_set
        ]

        overall_metrics = {}
        for topk in topk_list:
            # Get model's performance from unbiased features
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            attr_true, obj_true = self.get_attr_obj_from_pairs(labels)
            
            unbiased_pair_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['all_acc']
            
            attr_acc = self.evaluate(
                attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            
            obj_acc = self.evaluate(
                obj_preds, obj_true, seen_ids, unseen_ids)['all_acc']

            # Get model's performance on seen/unseen pairs
            pair_preds, _, _ = self.predict(features, topk=topk, bias=1e3)
            full_unseen_metrics = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)

            # Compute biases for AUC
            correct_scores = features[np.arange(len(features)), labels][unseen_ids]
            max_seen_scores = features[unseen_ids][:, self.seen_mask].topk(topk, dim=1)[0][:, topk-1]
            
            pairs_correct = torch.eq(pair_preds, labels.unsqueeze(1)).any(1).numpy()
            unseen_correct = pairs_correct[unseen_ids]
            unseen_score_diff = max_seen_scores - correct_scores
            correct_unseen_score_diff = unseen_score_diff[unseen_correct] - 1e-4
            correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
            magic_binsize = 20
            bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
            bias_list = correct_unseen_score_diff[::bias_skip]

            # Get biased predictions with different biases
            all_metrics = []
            for bias in tqdm(bias_list, disable=not progress_bar, desc="Computing AUC"):
                pair_preds, _, _ = self.predict(features, topk=topk, bias=bias)
                metrics = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)
                all_metrics.append(metrics)
            all_metrics.append(full_unseen_metrics)

            # Compute overall metrics
            seen_accs = np.array([m["seen_acc"] for m in all_metrics])
            unseen_accs = np.array([m["unseen_acc"] for m in all_metrics])
            best_seen_acc = max([m["seen_acc"] for m in all_metrics])
            best_unseen_acc = max([m["unseen_acc"] for m in all_metrics])
            best_harmonic_mean = max([m["harmonic_mean"] for m in all_metrics])
            auc = np.trapz(seen_accs, unseen_accs)

            overall_metrics[topk] = {
                "unbiased_pair_acc": unbiased_pair_acc,
                "attr_acc": attr_acc,
                "obj_acc": obj_acc,
                "best_seen_acc": best_seen_acc,
                "best_unseen_acc": best_unseen_acc,
                "best_harmonic_mean": best_harmonic_mean,
                "auc": auc,
            }
        return overall_metrics
    
    def get_fast_metrics(self, features, all_pairs_true, topk_list=[1]):
        """Compute metrics without AUC (faster)."""
        labels = torch.LongTensor(
            [self.dset.pair2idx[pair] for pair in all_pairs_true]
        )
        attr_true, obj_true = self.get_attr_obj_from_pairs(labels)

        # Seen/unseen samples
        seen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] in self.seen_pair_set
        ]
        unseen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] not in self.seen_pair_set
        ]

        fast_metrics = {}
        for topk in topk_list:
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            
            unbiased_pair_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['all_acc']
            
            attr_acc = self.evaluate(
                attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            
            obj_acc = self.evaluate(
                obj_preds, obj_true, seen_ids, unseen_ids)['all_acc']
            
            # Best seen (bias = -inf)
            pair_preds, _, _ = self.predict(features, topk=topk, bias=-1e3)
            best_seen_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['seen_acc']
            
            # Best unseen (bias = +inf)
            pair_preds, _, _ = self.predict(features, topk=topk, bias=1e3)
            best_unseen_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['unseen_acc']

            fast_metrics[topk] = {
                "unbiased_pair_acc": unbiased_pair_acc,
                "attr_acc": attr_acc,
                "obj_acc": obj_acc,
                "best_seen_acc": best_seen_acc,
                "best_unseen_acc": best_unseen_acc,
            }
        return fast_metrics


def compute_logits(video_embs, label_embs):
    """Compute similarity logits between video and label embeddings."""
    logit_scale = exp(0.07)
    logit_scale = min(logit_scale, 100.0)
    logits = logit_scale * video_embs @ label_embs.t()
    return logits.to('cpu')


def main(config: argparse.Namespace, verbose=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    # Paths
    data_path = os.path.join(config.data_dir, 'sth-sth-v2')
    
    # Load test dataset
    test_dataset = VideoCompositionDataset(
        data_path=data_path,
        embeddings_path=config.embeddings_dir,
        phase=config.test_phase,
        open_world=config.open_world
    )
    
    # Load train dataset for factorizer training
    train_dataset = VideoCompositionDataset(
        data_path=data_path,
        embeddings_path=config.embeddings_dir,
        phase='train',
        open_world=False
    )
    
    if verbose:
        scenario = 'open world' if config.open_world else 'closed world'
        print(f"="*60)
        print(f"Video Composition Evaluation on Something-Something V2")
        print(f"="*60)
        print(f"Factorizer      : {config.experiment_name}")
        print(f"Scenario        : {scenario}")
        print(f"Test Phase      : {config.test_phase}")
        print(f"Device          : {device}")
        print(f"Dataset         : {test_dataset}")
        print(f"="*60)

    all_results = []
    for e in range(config.n_exp):
        if verbose: 
            print(f'\nExperiment {e+1}/{config.n_exp}')
        
        # Step 1: Load training video embeddings from .pt file (dict of {video_id: embedding})
        # These are pre-computed video features from the video model (DiST/InternVideo2)
        embs_for_IW, all_pairs_IW = train_dataset.load_all_image_embs()
        embs_for_IW = embs_for_IW.to(device)
        
        # Step 2: Normalize video embeddings (L2 normalization)
        embs_for_IW = embs_for_IW / embs_for_IW.norm(dim=1, keepdim=True)
        
        # Step 3: Train compositional factorizer (GDE or LDE)
        # Factorizer learns to decompose video embeddings into verb + object components
        Factorizer = FACTORIZERS[config.experiment_name]
        if verbose:
            print(f"Training {config.experiment_name} factorizer on {len(all_pairs_IW)} training videos...")
        factorizer = Factorizer(embs_for_IW, all_pairs_IW, weights=None)
        
        # Step 4: Compose embeddings for all possible verb-object pairs
        # This creates synthetic embeddings by combining learned verb and object components
        if verbose:
            print(f"Composing embeddings for {len(test_dataset.pairs)} verb-object pairs...")
        test_pair_embs = factorizer.compute_ideal_words_approximation(
            target_pairs=test_dataset.pairs
        )
        
        # Step 5: Load test video embeddings and their ground truth labels
        image_embs, all_pairs_true = test_dataset.load_all_image_embs()
        image_embs = image_embs.to(device)
        test_pair_embs = test_pair_embs.to(device)
        
        # Step 6: Normalize all embeddings
        image_embs = image_embs / image_embs.norm(dim=1, keepdim=True)
        test_pair_embs = test_pair_embs / test_pair_embs.norm(dim=1, keepdim=True)
        
        # Step 7: Compute similarity scores (logits) between videos and composed pairs
        if verbose:
            print(f"Computing similarities for {len(image_embs)} test videos...")
        logits = compute_logits(image_embs, test_pair_embs)
        
        # Step 8: Evaluate compositional generalization
        evaluator = Evaluator(test_dataset)
        
        if config.compute_auc:
            result = evaluator.get_overall_metrics(
                logits, all_pairs_true, progress_bar=verbose)[1]  # topk=1
        else:
            result = evaluator.get_fast_metrics(
                logits, all_pairs_true)[1]  # topk=1
        
        all_results.append(result)

    # Combine results from multiple experiments
    if config.n_exp > 1:
        all_stats = list(all_results[0].keys())
        result = defaultdict(list)
        for res in all_results:
            for stat in all_stats:
                result[stat + ' (list)'].append(res[stat])
        for stat in all_stats:
            result[stat + ' (mean)'] = np.mean(result[stat + ' (list)'])
            result[stat + ' (std)'] = np.std(result[stat + ' (list)'])
        result = dict(result)
    else:
        result = all_results[0]

    # Save results
    if config.result_path is not None:
        os.makedirs(os.path.dirname(config.result_path), exist_ok=True)
        with open(config.result_path, 'w') as fp:
            experiment_details = {'config': vars(config), 'result': result}
            json.dump(experiment_details, fp, indent=4)
        if verbose:
            print(f"\nResults saved to: {config.result_path}")

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, list):
                print(f"  {k}: {[f'{x:.4f}' for x in v]}")
            else:
                print(f"  {k}: {v}")
        print(f"{'='*60}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Composition Evaluation')
    parser.add_argument(
        "--data_dir",
        help="Path to data directory",
        type=str,
        default="./data"
    )
    parser.add_argument(
        "--embeddings_dir",
        help="Path to embeddings directory",
        type=str,
        default="./embeddings"
    )
    parser.add_argument(
        "--experiment_name",
        help="Factorizer name (GDE or LDE)",
        type=str,
        choices=['GDE', 'LDE'],
        default='GDE'
    )
    parser.add_argument(
        "--open_world",
        help="Evaluate on open world setup",
        action="store_true"
    )
    parser.add_argument(
        "--n_exp",
        help="Number of times to repeat experiment",
        default=1,
        type=int
    )
    parser.add_argument(
        "--test_phase",
        help="Evaluation phase: 'test' or 'val'",
        default="test",
        type=str,
        choices=['test', 'val']
    )
    parser.add_argument(
        "--result_path",
        help="Path to save results JSON file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--compute_auc",
        help="Compute AUC metric (slower)",
        action="store_true"
    )
    
    config = parser.parse_args()
    main(config, verbose=True)
