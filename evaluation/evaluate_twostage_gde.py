"""
Two-Stage GDE Evaluation Script (Solution 2: Residual-Based Sequential Decomposition)

Problem: Standard GDE decomposes video embeddings into verb + object components jointly.
Since DiST embeddings are action-biased, verb signals dominate and object signals are weak.

Solution: Two-stage sequential decomposition:
  Stage 1: Extract verb components first (they are strong and easy to find)
  Stage 2: Compute residuals (remove verb), then extract object components from residuals
  Stage 3: Optional joint refinement starting from these better initializations

This forces the factorizer to find meaningful object features in the residual space
where verb signal has been removed.

Uses the same evaluation pipeline as evaluate_video_composition.py.
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
from factorizers.sphere import (
    calculate_intrinstic_mean,
    logarithmic_map,
    exponential_map,
)
from factorizers.factorizers import GDE, compute_group_means, weighted_mean


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Two-Stage GDE Factorizer
# ============================================================

class TwoStageGDE(GDE):
    """
    Two-Stage Geodesic Decomposition with Residual-Based Object Learning.
    
    Standard GDE:
        v ≈ exp_μ(A[verb] + O[obj])
        Problem: A dominates, O is weak
    
    Two-Stage GDE:
        Stage 1: Compute verb means A[verb] = mean(log_μ(v_i) for verb_i = verb)
        Stage 2: Compute residuals r_i = log_μ(v_i) - A[verb_i]
                 Then O[obj] = mean(r_i for obj_i = obj)
        Stage 3: (Optional) Joint refinement from this initialization
    
    This ensures object embeddings are learned from residuals where 
    the strong verb signal has already been removed.
    """
    name = 'TwoStageGDE'
    
    def __init__(self, embs_for_IW, all_pairs_gt, weights=None, 
                 obj_weight=1.0, refine_iters=0, verbose=True):
        """
        Args:
            embs_for_IW: Video embeddings [N, D] on the unit sphere
            all_pairs_gt: List of (verb, object) tuples
            weights: Optional weights for embeddings
            obj_weight: Weight multiplier for object component (higher = more emphasis)
            refine_iters: Number of joint refinement iterations (0 = no refinement)
            verbose: Print progress information
        """
        self.obj_weight = obj_weight
        self.refine_iters = refine_iters
        self.verbose = verbose
        
        # This calls compute_ideal_words internally via parent __init__
        # We override compute_ideal_words below
        super().__init__(embs_for_IW, all_pairs_gt, weights)
    
    def compute_ideal_words(self, embeddings, all_pairs_gt, weights=None):
        """
        Two-stage decomposition on the Riemannian manifold (sphere).
        
        Stage 1: Map to tangent space, compute verb means
        Stage 2: Compute residuals, extract object means from residuals
        Stage 3: Optional iterative refinement
        
        Returns:
            context: Intrinsic mean (μ) on the sphere
            attr_IW: Verb ideal words in tangent space T_μS^n
            obj_IW: Object ideal words in tangent space T_μS^n
        """
        if self.verbose:
            print(f"  [TwoStageGDE] Starting two-stage decomposition...")
        
        # Compute intrinsic mean on the sphere (same as standard GDE)
        intrinsic_mean = calculate_intrinstic_mean(embeddings, weights, init='normalized mean')
        
        # Map all embeddings to tangent space T_μS^n
        embeddings_T = logarithmic_map(intrinsic_mean, embeddings)
        
        attrs, objs = zip(*all_pairs_gt)
        unique_attrs = sorted(set(attrs))
        unique_objs = sorted(set(objs))
        
        # ============================================================
        # STAGE 1: Extract verb components (strong signal, easy)
        # ============================================================
        # For each verb, compute the mean of all video embeddings with that verb
        # This captures the dominant action pattern
        if self.verbose:
            print(f"  [Stage 1] Extracting verb components from {len(unique_attrs)} unique verbs...")
        
        attr_IW = compute_group_means(embeddings_T, attrs, unique_attrs, weights)
        
        # Center verb means (subtract global mean, which should be ~0 in tangent space)
        global_mean = weighted_mean(embeddings_T, weights)
        attr_IW_centered = attr_IW - global_mean
        
        # ============================================================
        # STAGE 2: Extract object components from RESIDUALS
        # ============================================================
        # Compute residuals: r_i = embedding_i - A[verb_i]
        # This removes the verb signal, leaving mainly object information
        if self.verbose:
            print(f"  [Stage 2] Computing residuals and extracting object components from {len(unique_objs)} unique objects...")
        
        # Build verb index mapping
        attr2idx = {a: i for i, a in enumerate(unique_attrs)}
        
        # Compute residuals for each video embedding
        residuals = torch.zeros_like(embeddings_T)
        for i, (attr, obj) in enumerate(all_pairs_gt):
            verb_idx = attr2idx[attr]
            # Remove verb component from this embedding
            # r_i = v_i - A[verb_i]
            residuals[i] = embeddings_T[i] - attr_IW_centered[verb_idx]
        
        # Now compute object means from residuals
        # O[obj] = mean(r_i for all i where obj_i = obj)
        obj_IW = compute_group_means(residuals, objs, unique_objs, weights)
        
        # Center object means (subtract residual global mean)
        residual_global_mean = weighted_mean(residuals, weights)
        obj_IW_centered = obj_IW - residual_global_mean
        
        if self.verbose:
            # Report signal strengths
            verb_norm = torch.norm(attr_IW_centered, dim=1).mean().item()
            obj_norm = torch.norm(obj_IW_centered, dim=1).mean().item()
            print(f"  [Stage 2] Verb signal strength: {verb_norm:.4f}")
            print(f"  [Stage 2] Object signal strength (from residuals): {obj_norm:.4f}")
            print(f"  [Stage 2] Object/Verb ratio: {obj_norm/verb_norm:.4f}")
        
        # ============================================================
        # STAGE 3: Optional joint refinement
        # ============================================================
        if self.refine_iters > 0:
            if self.verbose:
                print(f"  [Stage 3] Running {self.refine_iters} refinement iterations...")
            
            attr_IW_refined = attr_IW_centered.clone()
            obj_IW_refined = obj_IW_centered.clone()
            
            obj2idx = {o: i for i, o in enumerate(unique_objs)}
            
            for iteration in range(self.refine_iters):
                # Refine verbs: A[verb] = mean(v_i - O[obj_i]) for verb_i = verb
                verb_residuals = torch.zeros_like(embeddings_T)
                for i, (attr, obj) in enumerate(all_pairs_gt):
                    obj_idx = obj2idx[obj]
                    verb_residuals[i] = embeddings_T[i] - obj_IW_refined[obj_idx]
                
                attr_IW_refined = compute_group_means(verb_residuals, attrs, unique_attrs, weights)
                attr_IW_refined = attr_IW_refined - weighted_mean(verb_residuals, weights)
                
                # Refine objects: O[obj] = mean(v_i - A[verb_i]) for obj_i = obj
                obj_residuals = torch.zeros_like(embeddings_T)
                for i, (attr, obj) in enumerate(all_pairs_gt):
                    verb_idx = attr2idx[attr]
                    obj_residuals[i] = embeddings_T[i] - attr_IW_refined[verb_idx]
                
                obj_IW_refined = compute_group_means(obj_residuals, objs, unique_objs, weights)
                obj_IW_refined = obj_IW_refined - weighted_mean(obj_residuals, weights)
                
                if self.verbose:
                    verb_norm = torch.norm(attr_IW_refined, dim=1).mean().item()
                    obj_norm = torch.norm(obj_IW_refined, dim=1).mean().item()
                    
                    # Compute reconstruction error
                    recon_error = 0.0
                    for i, (attr, obj) in enumerate(all_pairs_gt):
                        v_idx = attr2idx[attr]
                        o_idx = obj2idx[obj]
                        recon = attr_IW_refined[v_idx] + obj_IW_refined[o_idx]
                        recon_error += torch.norm(embeddings_T[i] - recon).item()
                    recon_error /= len(all_pairs_gt)
                    
                    print(f"    Iter {iteration+1}: verb_norm={verb_norm:.4f}, "
                          f"obj_norm={obj_norm:.4f}, ratio={obj_norm/verb_norm:.4f}, "
                          f"recon_error={recon_error:.4f}")
            
            attr_IW_centered = attr_IW_refined
            obj_IW_centered = obj_IW_refined
        
        # Apply object weight: scale object components to give them more influence
        if self.obj_weight != 1.0:
            if self.verbose:
                print(f"  [TwoStageGDE] Applying object weight: {self.obj_weight}")
            obj_IW_centered = obj_IW_centered * self.obj_weight
        
        if self.verbose:
            print(f"  [TwoStageGDE] Decomposition complete!")
        
        context = intrinsic_mean
        return context, attr_IW_centered, obj_IW_centered


# ============================================================
# Reuse Dataset and Evaluator from original script
# ============================================================

class VideoCompositionDataset:
    """Dataset class for Something-Something V2 video embeddings."""
    
    def __init__(self, data_path, embeddings_path, phase='train', open_world=False):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.phase = phase
        self.open_world = open_world
        
        self.train_pairs_data = self._load_pairs('train')
        self.val_pairs_data = self._load_pairs('val')
        self.test_pairs_data = self._load_pairs('test')
        
        self.train_pairs = [(d['verb'], d['object']) for d in self.train_pairs_data]
        self.val_pairs = [(d['verb'], d['object']) for d in self.val_pairs_data]
        self.test_pairs = [(d['verb'], d['object']) for d in self.test_pairs_data]
        
        all_pairs = list(set(self.train_pairs + self.val_pairs + self.test_pairs))
        self.pairs = sorted(all_pairs)
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        
        all_verbs = sorted(set(p[0] for p in self.pairs))
        all_objs = sorted(set(p[1] for p in self.pairs))
        
        self.attrs = all_verbs
        self.objs = all_objs
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        
        self.train_embeddings = self._load_embeddings('train')
        self.val_embeddings = self._load_embeddings('val')
        self.test_embeddings = self._load_embeddings('test')
        
        self.train_id2pair = {d['id']: (d['verb'], d['object']) for d in self.train_pairs_data}
        self.val_id2pair = {d['id']: (d['verb'], d['object']) for d in self.val_pairs_data}
        self.test_id2pair = {d['id']: (d['verb'], d['object']) for d in self.test_pairs_data}
    
    def _load_pairs(self, split):
        json_path = os.path.join(self.data_path, f'{split}_pairs.json')
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _load_embeddings(self, split):
        for emb_name in [f'dist_{split}.pt', f'internvideo2_{split}.pt']:
            emb_path = os.path.join(self.embeddings_path, emb_name)
            if os.path.exists(emb_path):
                emb_dict = torch.load(emb_path)
                return {str(k): v for k, v in emb_dict.items()}
        raise FileNotFoundError(f"No embedding file found for split: {split}")
    
    def load_all_image_embs(self):
        if self.phase == 'train':
            emb_dict = self.train_embeddings
            id2pair = self.train_id2pair
        elif self.phase == 'val':
            emb_dict = self.val_embeddings
            id2pair = self.val_id2pair
        else:
            emb_dict = self.test_embeddings
            id2pair = self.test_id2pair
        
        embeddings = []
        all_pairs = []
        for vid_id in sorted(emb_dict.keys(), key=int):
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
    """Evaluation class (same as original)."""
    def __init__(self, dset: VideoCompositionDataset):
        self.dset = dset

        if dset.phase == 'train':
            test_pair_set = set(dset.train_pairs)
        elif dset.phase == 'val':
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
        else:
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
        
        if not dset.open_world:
            self.closed_mask = torch.BoolTensor(
                [1 if pair in test_pair_set else 0 for pair in dset.pairs]
            )

        self.seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in self.seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        self.pair_idx2ao_idx = torch.LongTensor([
            (dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs
        ])

    def get_attr_obj_from_pairs(self, pairs):
        attrs = self.pair_idx2ao_idx[pairs, 0]
        objs = self.pair_idx2ao_idx[pairs, 1]
        return attrs, objs

    def track_wrong_predictions(self, y_pred_topk, y_true, all_pairs_true):
        wrong_predictions = []
        correct = torch.eq(y_pred_topk, y_true.unsqueeze(1)).any(1).numpy()
        for i, is_correct in enumerate(correct):
            if not is_correct:
                gt_pair = all_pairs_true[i]
                pred_idx = y_pred_topk[i, 0].item()
                pred_pair = self.dset.pairs[pred_idx]
                wrong_predictions.append({
                    'video_idx': i,
                    'ground_truth': gt_pair,
                    'predicted': pred_pair
                })
        return wrong_predictions

    def evaluate(self, y_pred_topk, y_true, seen_ids, unseen_ids):
        correct = torch.eq(y_pred_topk, y_true.unsqueeze(1)).any(1).numpy()
        all_acc = np.mean(correct)
        seen_acc = np.mean(correct[seen_ids]) if len(seen_ids) > 0 else 0.0
        unseen_acc = np.mean(correct[unseen_ids]) if len(unseen_ids) > 0 else 0.0
        
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
        scores = scores.clone()
        scores[:, ~self.seen_mask] += bias
        if not self.dset.open_world:
            scores[:, ~self.closed_mask] = -1e10
        _, pair_preds = scores.topk(topk, dim=1)
        attr_preds, obj_preds = self.get_attr_obj_from_pairs(pair_preds)
        return pair_preds, attr_preds, obj_preds
    
    def get_overall_metrics(self, features, all_pairs_true, topk_list=[1], progress_bar=True):
        labels = torch.LongTensor(
            [self.dset.pair2idx[pair] for pair in all_pairs_true]
        )
        seen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] in self.seen_pair_set
        ]
        unseen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] not in self.seen_pair_set
        ]

        overall_metrics = {}
        wrong_predictions_all = {}
        
        for topk in topk_list:
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            attr_true, obj_true = self.get_attr_obj_from_pairs(labels)
            
            wrong_preds = self.track_wrong_predictions(pair_preds, labels, all_pairs_true)
            wrong_predictions_all[topk] = wrong_preds
            
            unbiased_pair_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['all_acc']
            attr_acc = self.evaluate(
                attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            obj_acc = self.evaluate(
                obj_preds, obj_true, seen_ids, unseen_ids)['all_acc']

            pair_preds, _, _ = self.predict(features, topk=topk, bias=1e3)
            full_unseen_metrics = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)

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

            all_metrics = []
            for bias in tqdm(bias_list, disable=not progress_bar, desc="Computing AUC"):
                pair_preds, _, _ = self.predict(features, topk=topk, bias=bias)
                metrics = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)
                all_metrics.append(metrics)
            all_metrics.append(full_unseen_metrics)

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
        return overall_metrics, wrong_predictions_all
    
    def get_fast_metrics(self, features, all_pairs_true, topk_list=[1]):
        labels = torch.LongTensor(
            [self.dset.pair2idx[pair] for pair in all_pairs_true]
        )
        attr_true, obj_true = self.get_attr_obj_from_pairs(labels)
        seen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] in self.seen_pair_set
        ]
        unseen_ids = [
            i for i in range(len(all_pairs_true)) if all_pairs_true[i] not in self.seen_pair_set
        ]

        fast_metrics = {}
        wrong_predictions_all = {}
        
        for topk in topk_list:
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            
            wrong_preds = self.track_wrong_predictions(pair_preds, labels, all_pairs_true)
            wrong_predictions_all[topk] = wrong_preds
            
            unbiased_pair_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['all_acc']
            attr_acc = self.evaluate(
                attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            obj_acc = self.evaluate(
                obj_preds, obj_true, seen_ids, unseen_ids)['all_acc']
            
            pair_preds, _, _ = self.predict(features, topk=topk, bias=-1e3)
            best_seen_acc = self.evaluate(
                pair_preds, labels, seen_ids, unseen_ids)['seen_acc']
            
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
        return fast_metrics, wrong_predictions_all


def compute_logits(video_embs, label_embs):
    """Compute similarity logits between video and label embeddings."""
    logit_scale = exp(0.07)
    logit_scale = min(logit_scale, 100.0)
    logits = logit_scale * video_embs @ label_embs.t()
    return logits.to('cpu')


def compute_decomposed_logits(video_embs, factorizer, test_pairs, alpha=0.5):
    """
    Decomposed Scoring: Score verb and object channels SEPARATELY.
    
    Instead of:  score = sim(video, compose(A[verb] + O[obj]))
                 ↑ verb dominates this single similarity
    
    We compute:  score = α * sim(v_tangent, A[verb]) + (1-α) * sim(v_residual, O[obj])
                 ↑ verb score (weighted)      ↑ object score (weighted independently)
    
    Where:
        v_tangent = log_μ(v)                  # video in tangent space
        v_residual = v_tangent - A[verb_pred] # video with verb removed
    
    Key insight: By scoring separately, the weak object signal gets its own
    independent channel and cannot be overwhelmed by the strong verb signal.
    
    Args:
        video_embs: Test video embeddings [N, D] (normalized, on sphere)
        factorizer: Trained TwoStageGDE factorizer
        test_pairs: List of all (verb, obj) pairs to score against
        alpha: Mixing weight. 0.5 = equal weight. Lower = more object emphasis.
    
    Returns:
        logits: [N, num_pairs] decomposed similarity scores
    """
    device = video_embs.device
    
    # Map test videos to tangent space at μ
    video_tangent = logarithmic_map(factorizer.context, video_embs)  # [N, D]
    
    # Get verb and object ideal words
    attr_IW = factorizer.attr_IW.to(device)  # [num_verbs, D]
    obj_IW = factorizer.obj_IW.to(device)    # [num_objs, D]
    
    # Normalize ideal words for cosine similarity
    attr_IW_norm = attr_IW / (attr_IW.norm(dim=1, keepdim=True) + 1e-8)
    obj_IW_norm = obj_IW / (obj_IW.norm(dim=1, keepdim=True) + 1e-8)
    
    # For each test pair, get verb and object indices
    pair_attr_idx = torch.tensor(
        [factorizer.attr2idx[attr] for attr, _ in test_pairs], device=device)
    pair_obj_idx = torch.tensor(
        [factorizer.obj2idx[obj] for _, obj in test_pairs], device=device)
    
    # ---- Verb channel ----
    # sim(v_tangent, A[verb]) for each pair
    video_tangent_norm = video_tangent / (video_tangent.norm(dim=1, keepdim=True) + 1e-8)
    verb_sims_all = video_tangent_norm @ attr_IW_norm.t()  # [N, num_verbs]
    verb_scores = verb_sims_all[:, pair_attr_idx]  # [N, num_pairs]
    
    # ---- Object channel (from residuals) ----
    # For each video, compute residual by removing the BEST verb
    # v_residual = v_tangent - A[best_verb]
    best_verb_idx = verb_sims_all.argmax(dim=1)  # [N]
    best_verb_embs = attr_IW[best_verb_idx]  # [N, D]
    video_residual = video_tangent - best_verb_embs  # [N, D]
    
    video_residual_norm = video_residual / (video_residual.norm(dim=1, keepdim=True) + 1e-8)
    obj_sims_all = video_residual_norm @ obj_IW_norm.t()  # [N, num_objs]
    obj_scores = obj_sims_all[:, pair_obj_idx]  # [N, num_pairs]
    
    # ---- Combined score ----
    logits = alpha * verb_scores + (1.0 - alpha) * obj_scores
    
    return logits.to('cpu')


# ============================================================
# Main
# ============================================================

def main(config: argparse.Namespace, verbose=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    data_path = os.path.join(config.data_dir, 'sth-sth-v2')
    
    test_dataset = VideoCompositionDataset(
        data_path=data_path,
        embeddings_path=config.embeddings_dir,
        phase=config.test_phase,
        open_world=config.open_world
    )
    
    train_dataset = VideoCompositionDataset(
        data_path=data_path,
        embeddings_path=config.embeddings_dir,
        phase='train',
        open_world=False
    )
    
    if verbose:
        scenario = 'open world' if config.open_world else 'closed world'
        print(f"="*60)
        print(f"Two-Stage GDE Evaluation on Something-Something V2")
        print(f"="*60)
        print(f"Method          : TwoStageGDE (Residual Decomposition)")
        print(f"Scoring Mode    : {config.scoring_mode}")
        if config.scoring_mode == 'decomposed':
            print(f"Alpha (verb wt) : {config.alpha}")
        print(f"Scenario        : {scenario}")
        print(f"Test Phase      : {config.test_phase}")
        print(f"Object Weight   : {config.obj_weight}")
        print(f"Refine Iters    : {config.refine_iters}")
        print(f"Device          : {device}")
        print(f"Dataset         : {test_dataset}")
        print(f"="*60)

    all_results = []
    all_wrong_predictions = []
    
    for e in range(config.n_exp):
        if verbose: 
            print(f'\nExperiment {e+1}/{config.n_exp}')
        
        # Step 1: Load training video embeddings
        embs_for_IW, all_pairs_IW = train_dataset.load_all_image_embs()
        embs_for_IW = embs_for_IW.to(device)
        
        # Step 2: Normalize
        embs_for_IW = embs_for_IW / embs_for_IW.norm(dim=1, keepdim=True)
        
        # Step 3: Train Two-Stage GDE factorizer
        if verbose:
            print(f"Training TwoStageGDE on {len(all_pairs_IW)} training videos...")
        factorizer = TwoStageGDE(
            embs_for_IW, all_pairs_IW, weights=None,
            obj_weight=config.obj_weight,
            refine_iters=config.refine_iters,
            verbose=verbose
        )
        
        # Step 4: Compose embeddings for all verb-object pairs
        if verbose:
            print(f"Composing embeddings for {len(test_dataset.pairs)} verb-object pairs...")
        test_pair_embs = factorizer.compute_ideal_words_approximation(
            target_pairs=test_dataset.pairs
        )
        
        # Step 5: Load test embeddings
        image_embs, all_pairs_true = test_dataset.load_all_image_embs()
        image_embs = image_embs.to(device)
        test_pair_embs = test_pair_embs.to(device)
        
        # Step 6: Normalize
        image_embs = image_embs / image_embs.norm(dim=1, keepdim=True)
        test_pair_embs = test_pair_embs / test_pair_embs.norm(dim=1, keepdim=True)
        
        # Step 7: Compute logits
        if verbose:
            print(f"Computing similarities for {len(image_embs)} test videos...")
        
        if config.scoring_mode == 'decomposed':
            # DECOMPOSED SCORING: score verb and object separately
            if verbose:
                print(f"Using decomposed scoring (alpha={config.alpha})")
            logits = compute_decomposed_logits(
                image_embs, factorizer, test_dataset.pairs, alpha=config.alpha)
        else:
            # Standard composed scoring
            logits = compute_logits(image_embs, test_pair_embs)
        
        # Step 8: Evaluate
        evaluator = Evaluator(test_dataset)
        
        if config.compute_auc:
            metrics, wrong_preds = evaluator.get_overall_metrics(
                logits, all_pairs_true, progress_bar=verbose)
            result = metrics[1]
            wrong_predictions = wrong_preds[1]
        else:
            metrics, wrong_preds = evaluator.get_fast_metrics(
                logits, all_pairs_true)
            result = metrics[1]
            wrong_predictions = wrong_preds[1]
        
        all_results.append(result)
        all_wrong_predictions.append(wrong_predictions)

    # Combine results
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
            experiment_details = {
                'config': vars(config),
                'method': 'TwoStageGDE',
                'result': result
            }
            json.dump(experiment_details, fp, indent=4)
        if verbose:
            print(f"\nResults saved to: {config.result_path}")
        
        wrong_preds_path = config.result_path.replace('.json', '_wrong_predictions.json')
        with open(wrong_preds_path, 'w') as fp:
            wrong_preds_data = {
                'config': vars(config),
                'method': 'TwoStageGDE',
                'num_wrong_predictions': len(all_wrong_predictions[0]),
                'wrong_predictions': all_wrong_predictions[0]
            }
            json.dump(wrong_preds_data, fp, indent=4)
        if verbose:
            print(f"Wrong predictions saved to: {wrong_preds_path}")
            print(f"Total wrong predictions: {len(all_wrong_predictions[0])}")

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS (TwoStageGDE)")
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
    parser = argparse.ArgumentParser(description='Two-Stage GDE Video Composition Evaluation')
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings")
    parser.add_argument("--open_world", action="store_true")
    parser.add_argument("--n_exp", default=1, type=int)
    parser.add_argument("--test_phase", default="test", type=str, choices=['test', 'val'])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--compute_auc", action="store_true")
    
    # Two-Stage GDE specific arguments
    parser.add_argument(
        "--obj_weight",
        help="Weight multiplier for object component (higher = more object emphasis)",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--refine_iters",
        help="Number of alternating refinement iterations (0 = two-stage only, no refinement)",
        type=int,
        default=3
    )
    parser.add_argument(
        "--scoring_mode",
        help="Scoring method: 'composed' (standard) or 'decomposed' (separate verb/obj scoring)",
        type=str,
        choices=['composed', 'decomposed'],
        default='decomposed'
    )
    parser.add_argument(
        "--alpha",
        help="Verb weight in decomposed scoring (0.0=object only, 0.5=equal, 1.0=verb only)",
        type=float,
        default=0.5
    )
    
    config = parser.parse_args()
    main(config, verbose=True)
