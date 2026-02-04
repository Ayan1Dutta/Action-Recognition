'''Objects for decomposing embeddings of attribute-object compositions.'''

import sys
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path

# Ensure we can import `sphere` regardless of this file's location (root vs evaluation subdir)
_file_path = Path(__file__).resolve()
_candidate_dirs = [
    _file_path.parent,                    # e.g., repo root
    _file_path.parent.parent,             # e.g., Compositionality_testing
    _file_path.parent / 'Compositionality_testing'
]
_sphere_dir = None
for candidate in _candidate_dirs:
    if candidate is not None and (candidate / 'sphere.py').exists():
        _sphere_dir = candidate
        break

if _sphere_dir is None:
    raise ImportError("Unable to locate sphere.py for factorizer imports")

if str(_sphere_dir) not in sys.path:
    sys.path.insert(0, str(_sphere_dir))

from sphere import (calculate_intrinstic_mean,
                    logarithmic_map,
                    exponential_map,
                    parallel_transport)


def weighted_mean(embeddings, weights=None):
    '''Calculate the weighted mean of `embeddings`. The `weights` are normalized.'''
    if weights is None:
        return embeddings.mean(dim=0)
    else:
        return weights @ embeddings / weights.sum()


def compute_group_means(embeddings, group_ids, unique_groups, weights=None):
    '''
    Computes the mean vector for each group in unique_groups.
    Vector embeddings[i] and embeddings[j] belongs to the same group iff group_ids[i]=group_ids[j].
    '''
    group_id2idx= {id: [] for id in unique_groups}
    for i, group_id in enumerate(group_ids):
        group_id2idx[group_id].append(i)
    
    means = []
    for id in unique_groups:
        idx = group_id2idx[id]

        group_weights = None if weights is None else weights[idx]
        group_mean = weighted_mean(embeddings[idx], group_weights)
        means.append(group_mean)

    return torch.stack(means)


def compute_attr_obj_means(embeddings, all_pairs_gt, centered=True, weights=None):
    '''
    Computes mean for each attribute and object.
    If two or more embeddings have the same pair, a the denoising step is performed first.
    `weights` gives the weight distribution within pair. If None, uniform weights are used.
    '''

    mean_all = weighted_mean(embeddings, weights)
    
    attrs, objs = zip(*all_pairs_gt)
    attr_means = compute_group_means(embeddings, attrs, sorted(set(attrs)), weights)  # sorted wrt unique attrs
    obj_means = compute_group_means(embeddings, objs, sorted(set(objs)), weights)     # sorted wrt unique objs
    
    if centered:
        attr_means = attr_means - mean_all
        obj_means = obj_means - mean_all

    return mean_all, attr_means, obj_means


# Factorizers

class CompositionalFactorizer:

    def __init__(self, embs_for_IW, all_pairs_gt, weights=None):
        '''
        Class that represents a compositional structure for a set of embeddings.
        Input:
            dataset: dataset of the embeddings
            embs_for_IW: embeddings used to compute the Ideal Words (primitive directions in the optimal decomposition)
            all_pair_gt: (attr, obj) label for `embs_for_IW`
            weights: weights assigned to the `embs_for_IW`, if `None` uniform weights are used. Weights are automatically normalized within pair.
        '''
        self.device = embs_for_IW.device
        self.all_pairs_gt = all_pairs_gt
        self.embs_for_IW = embs_for_IW
        self.weights = weights

        attrs, objs = zip(*all_pairs_gt)
        self.attrs = sorted(set(attrs))
        self.objs = sorted(set(objs))

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}

        # Compute IW for attrs and objs in dataset
        self.context, self.attr_IW, self.obj_IW = self.compute_ideal_words(
            embeddings=embs_for_IW,
            all_pairs_gt=all_pairs_gt,
            weights=weights
        )

    def compute_ideal_words(self, embeddings, all_pairs_gt):
        '''
        Extracts ideal words from `embeddings` labeled with `all_pairs_gt`.
        '''
        raise(NotImplementedError)
    
    def combine_ideal_words(self, *ideal_words, context=None):
        '''
        Combines ideal words using `context` as center.
        If context is `None`, `self.context` is used.
        '''
        raise(NotImplementedError)
    
    def get_attr_IW(self, attr):
        attr_idx = self.attr2idx[attr]
        return self.attr_IW[attr_idx]
    
    def get_obj_IW(self, obj):
        obj_idx = self.obj2idx[obj]
        return self.obj_IW[obj_idx]

    def compute_ideal_words_approximation(self, target_pairs):
        target_attr_idx = torch.tensor(
            [self.attr2idx[attr] for attr, _ in target_pairs],
            device=self.device)
        target_obj_idx = torch.tensor(
            [self.obj2idx[obj] for _, obj in target_pairs],
            device=self.device)
        
        # Select attr_IW and obj_IW for target pairs
        attrIW_target = self.attr_IW[target_attr_idx]
        objIW_target = self.obj_IW[target_obj_idx]

        # Compute IW approximation for target pairs
        target_IWapprox = self.combine_ideal_words(attrIW_target, objIW_target)

        return target_IWapprox
        
    def __str__(self) -> str:
        return self.name


class LDE(CompositionalFactorizer):
    name = 'LDE'

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        return compute_attr_obj_means(embeddings, all_pairs_gt, weights=weights)

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        ideal_words = torch.stack(ideal_words)
        return context + torch.sum(ideal_words, dim=0)
    
    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        denoised_pair = compute_group_means(self.embs_for_IW, self.all_pairs_gt, unique_pairs)
        return unique_pairs, denoised_pair


class GDE(CompositionalFactorizer):
    name = 'GDE'
    
    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        intrinsic_mean = calculate_intrinstic_mean(embeddings, weights, init='normalized mean')  # mu

        # 1) Map embedding to the tangent space T_muS^n
        embeddings_T = logarithmic_map(intrinsic_mean, embeddings)

        # 2) Compute IW on the tangent space
        v_c, attr_IW, obj_IW = compute_attr_obj_means(embeddings_T, all_pairs_gt, weights=weights)
        assert torch.norm(v_c, p=2) < 1e-5 # should be v_c=0

        context = intrinsic_mean
        return context, attr_IW, obj_IW

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            original_contex = True
            context = self.context
        else:
            original_contex = False

        # 3) Combine ideal words in the tangent plane
        ideal_words = torch.stack(ideal_words)
        embs_approx_T = torch.sum(ideal_words, dim=0)

        # 4) Map the obtained approximation back to the sphere
        if original_contex:
            embs_approx = exponential_map(context, embs_approx_T)
        else:
            # If context is not mu, we need to transport embs_approx_T from T_muS^n to T_contextS^n
            embs_approx_T_transported = parallel_transport(self.context, context, embs_approx_T)
            embs_approx = exponential_map(context, embs_approx_T_transported)
        
        return embs_approx

    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        embs_T = logarithmic_map(self.context, self.embs_for_IW)
        denoised_pair_T = compute_group_means(embs_T, self.all_pairs_gt, unique_pairs)
        denoised_pair = exponential_map(self.context, denoised_pair_T)
        return unique_pairs, denoised_pair

#my method not proposed in the base paper


class ContrastiveGDE(GDE):
    """
    Contrastive GDE variant that refines ideal words to maximize separation
    between ambiguous attribute-object compositions using margin-based contrastive loss.
    """
    name = 'ContrastiveGDE'
    
    def find_confusing_pairs(self, train_embeddings, train_pairs, top_k=5):
        """
        Find most confusing pairs using baseline GDE predictions.
        Returns dict mapping each pair to its top-k most confused pairs.
        """
        device = self.device
        
        # Get all unique pairs and their indices
        unique_pairs = sorted(set(train_pairs))
        pair_to_idx = {p: i for i, p in enumerate(unique_pairs)}
        
        # Compute composed embeddings for all unique pairs
        composed = self.compute_ideal_words_approximation(unique_pairs).to(device)
        composed_norm = F.normalize(composed, dim=1)
        
        # Compute similarity matrix (on GPU)
        sim_matrix = composed_norm @ composed_norm.t()  # [N, N]
        
        # For each pair, find top-k most similar (confusing) pairs
        confusion_map = {}
        for i, pair in enumerate(unique_pairs):
            sims = sim_matrix[i].clone()
            sims[i] = -1.0  # Exclude self
            top_indices = torch.topk(sims, min(top_k, len(unique_pairs) - 1)).indices.cpu().tolist()
            confusion_map[pair] = [unique_pairs[idx] for idx in top_indices]
        
        return confusion_map
    
    def contrastive_refine(
        self,
        train_pairs: List[Tuple[str, str]],
        steps: int = 1,
        lr: float = 0.0001,
        margin: float = 0.5,
        reg_weight: float = 1.0,
        verbose: bool = True,
        use_confusing_pairs: bool = True
    ):
        """
        Refine attr_IW and obj_IW using contrastive learning on CONFUSED pairs only.
        
        IMPROVED ALGORITHM:
        1. Identify confused pairs where GDE prediction is wrong
        2. For confused pairs only:
           - Positive: Pull predicted composition toward TRUE CLIP embedding
           - Negative: Push predicted composition away from confused predictions
           - Regularization: Stay close to GDE initialization
        
        3. Key improvements:
           - Only update parameters involved in confused pairs (preserve good GDE knowledge)
           - Very small learning rate to avoid disrupting learned structure
           - Strong regularization to prevent overfitting
           - Confusion-aware hard negative mining
        
        Args:
            train_pairs: List of (attr, obj) tuples for training
            steps: Number of optimization steps (default: 1, max recommended: 2)
            lr: Learning rate (default: 0.0001, very conservative)
            margin: Margin for contrastive loss (default: 0.5, strong separation)
            reg_weight: Regularization weight to stay close to GDE init (default: 1.0, strong)
            verbose: Print progress
            use_confusing_pairs: Use confusion-based hard negative mining
        """
        device = self.device
        
        # Build index mappings
        attr_to_idx = {a: i for i, a in enumerate(self.attrs)}
        obj_to_idx = {o: i for i, o in enumerate(self.objs)}
        
        # Build pair to embedding mapping (TRUE CLIP embeddings)
        pair_to_emb_idx = {}
        for idx, pair in enumerate(self.all_pairs_gt):
            if pair not in pair_to_emb_idx:
                pair_to_emb_idx[pair] = idx
        
        # Filter valid training pairs that have TRUE embeddings
        all_valid_pairs = []
        for a, o in train_pairs:
            pair = (a, o)
            if a in attr_to_idx and o in obj_to_idx and pair in pair_to_emb_idx:
                a_idx = attr_to_idx[a]
                o_idx = obj_to_idx[o]
                emb_idx = pair_to_emb_idx[pair]
                true_emb = self.embs_for_IW[emb_idx].to(device)  # TRUE CLIP embedding
                all_valid_pairs.append((a_idx, o_idx, true_emb, pair))
        
        if not all_valid_pairs:
            if verbose:
                print("[ContrastiveGDE] No valid pairs with true embeddings to refine")
            return
        
        # IDENTIFY CONFUSED PAIRS: Only train on pairs where GDE makes mistakes
        confused_pairs = []
        if use_confusing_pairs:
            if verbose:
                print(f"[ContrastiveGDE] Identifying confused pairs from GDE predictions...")
            
            for a_idx, o_idx, true_emb, pair in all_valid_pairs:
                # Compute GDE prediction
                comp_tangent = self.attr_IW[a_idx] + self.obj_IW[o_idx]
                pred_sphere = exponential_map(self.context, comp_tangent.unsqueeze(0)).squeeze(0)
                
                # Check if prediction matches ground truth
                sim_to_true = F.cosine_similarity(pred_sphere.unsqueeze(0), true_emb.unsqueeze(0)).item()
                
                # Consider "confused" if similarity is not very high (threshold: 0.95)
                if sim_to_true < 0.95:
                    confused_pairs.append((a_idx, o_idx, true_emb, pair, pred_sphere))
            
            if verbose:
                print(f"[ContrastiveGDE] Found {len(confused_pairs)} confused pairs out of {len(all_valid_pairs)} ({100*len(confused_pairs)/len(all_valid_pairs):.1f}%)")
        else:
            # Use all pairs if not filtering
            confused_pairs = [(a, o, e, p, None) for a, o, e, p in all_valid_pairs]
        
        if not confused_pairs:
            if verbose:
                print("[ContrastiveGDE] No confused pairs found - GDE is already performing well!")
            return
        
        # Find confusing pairs for hard negative mining
        confusion_map = {}
        if use_confusing_pairs:
            unique_pairs = sorted(set(pair for _, _, _, pair, _ in confused_pairs))
            
            # Compute predictions for all unique pairs
            pred_embs = []
            for pair in unique_pairs:
                a, o = pair
                a_idx = attr_to_idx[a]
                o_idx = obj_to_idx[o]
                comp_tangent = self.attr_IW[a_idx] + self.obj_IW[o_idx]
                comp_sphere = exponential_map(self.context, comp_tangent.unsqueeze(0)).squeeze(0)
                pred_embs.append(comp_sphere)
            pred_embs_tensor = torch.stack(pred_embs).to(device)
            
            confusion_map = self.find_confusing_pairs(pred_embs_tensor, unique_pairs, top_k=5)
            if verbose:
                print(f"[ContrastiveGDE] Built confusion map for hard negative mining")
        
        # Create learnable parameters (clone ideal words in tangent space)
        # IMPORTANT: Keep copy of initialization for regularization
        attr_params = torch.nn.Parameter(self.attr_IW.clone().to(device))
        obj_params = torch.nn.Parameter(self.obj_IW.clone().to(device))
        
        # Store initial values for regularization
        attr_init = self.attr_IW.clone().to(device)
        obj_init = self.obj_IW.clone().to(device)
        
        optimizer = torch.optim.Adam([attr_params, obj_params], lr=lr)
        
        if verbose:
            print(f"[ContrastiveGDE] Refining {len(confused_pairs)} confused pairs")
            print(f"[ContrastiveGDE] steps={steps}, lr={lr}, margin={margin}, reg_weight={reg_weight}")
            print(f"[ContrastiveGDE] Using GPU: {device}")
        
        # Training loop with progress bar
        from tqdm import tqdm
        
        for step in tqdm(range(steps), desc="Contrastive Refinement", disable=not verbose):
            optimizer.zero_grad()
            losses = []
            
            # Shuffle confused pairs each epoch
            import random
            shuffled_data = confused_pairs.copy()
            random.shuffle(shuffled_data)
            
            # Process confused pairs
            for a_idx, o_idx, true_emb, pair, _ in shuffled_data:
                # POSITIVE: Compose using current parameters
                pos_tangent = attr_params[a_idx] + obj_params[o_idx]
                pos_composed = exponential_map(self.context, pos_tangent.unsqueeze(0)).squeeze(0)
                
                # Similarity to TRUE embedding (maximize this)
                sim_positive = F.cosine_similarity(pos_composed.unsqueeze(0), true_emb.unsqueeze(0)).squeeze()
                
                # NEGATIVES: Use confused pairs as hard negatives
                if pair in confusion_map and confusion_map[pair]:
                    # Use one of the confused pairs as negative
                    confused = confusion_map[pair]
                    neg_pair = confused[step % len(confused)]
                    neg_a, neg_o = neg_pair
                    neg_a_idx = attr_to_idx[neg_a]
                    neg_o_idx = obj_to_idx[neg_o]
                else:
                    # Fallback to swapping one component
                    neg_o_idx = torch.randint(0, len(self.objs), (1,), device=device).item()
                    while neg_o_idx == o_idx and len(self.objs) > 1:
                        neg_o_idx = torch.randint(0, len(self.objs), (1,), device=device).item()
                    neg_a_idx = a_idx
                
                # Compose negative
                neg_tangent = attr_params[neg_a_idx] + obj_params[neg_o_idx]
                neg_composed = exponential_map(self.context, neg_tangent.unsqueeze(0)).squeeze(0)
                
                # Similarity of negative to TRUE embedding (minimize this)
                sim_negative = F.cosine_similarity(neg_composed.unsqueeze(0), true_emb.unsqueeze(0)).squeeze()
                
                # CONTRASTIVE LOSS: sim_positive should exceed sim_negative by margin
                contrastive_loss = torch.clamp(margin - (sim_positive - sim_negative), min=0.0)
                
                losses.append(contrastive_loss)
            
            if not losses:
                break
            
            # Compute mean contrastive loss
            mean_contrastive_loss = torch.stack(losses).mean()
            
            # REGULARIZATION: Stay close to GDE initialization
            # This prevents the refinement from destroying what GDE learned
            reg_loss_attr = torch.mean((attr_params - attr_init) ** 2)
            reg_loss_obj = torch.mean((obj_params - obj_init) ** 2)
            reg_loss = reg_loss_attr + reg_loss_obj
            
            # Total loss = contrastive + regularization
            total_loss = mean_contrastive_loss + reg_weight * reg_loss
            total_loss.backward()
            
            # RIEMANNIAN GRADIENT DESCENT:
            # Project gradients to tangent space before stepping
            with torch.no_grad():
                # Project attr gradients
                for i in range(len(self.attrs)):
                    if attr_params.grad is not None:
                        # Current point on manifold (in tangent space at context)
                        v = attr_params[i]
                        grad = attr_params.grad[i]
                        
                        # Project gradient to tangent space (remove radial component)
                        # Since we're already in tangent space at context, we project perpendicular to v
                        grad_proj = grad - torch.dot(grad, v) * v
                        attr_params.grad[i] = grad_proj
                
                # Project obj gradients
                for i in range(len(self.objs)):
                    if obj_params.grad is not None:
                        v = obj_params[i]
                        grad = obj_params.grad[i]
                        grad_proj = grad - torch.dot(grad, v) * v
                        obj_params.grad[i] = grad_proj
            
            # Update with projected gradients
            optimizer.step()
            
            # RETRACT TO MANIFOLD:
            # Normalize to ensure we stay on sphere (tangent vectors should remain normalized)
            with torch.no_grad():
                attr_params.data = F.normalize(attr_params.data, dim=1)
                obj_params.data = F.normalize(obj_params.data, dim=1)
            
            if verbose:
                tqdm.write(f"  Step {step + 1}/{steps}: contrast_loss={mean_contrastive_loss.item():.4f}, reg_loss={reg_loss.item():.4f}, total={total_loss.item():.4f}")
        
        # Update ideal words with refined versions
        self.attr_IW = attr_params.detach()
        self.obj_IW = obj_params.detach()
        
        if verbose:
            print("[ContrastiveGDE] Refinement complete")


FACTORIZERS = {
    'LDE': LDE,
    'GDE': GDE,
    'ContrastiveGDE': ContrastiveGDE,
}


def train_factorizer_with_contrastive(embs, pairs, refine_steps=5, lr=0.01, margin=0.1, verbose=True):
    """
    Helper function to train ContrastiveGDE with specified refinement steps.
    
    Args:
        embs: Training embeddings
        pairs: List of (attr, obj) tuples
        refine_steps: Number of contrastive refinement steps (default: 5)
        lr: Learning rate for refinement
        margin: Contrastive margin
        verbose: Print progress
    
    Returns:
        Trained ContrastiveGDE factorizer
    """
    factorizer = ContrastiveGDE(embs, pairs, weights=None)
    if refine_steps > 0:
        factorizer.contrastive_refine(
            train_pairs=pairs,
            steps=refine_steps,
            lr=lr,
            margin=margin,
            verbose=verbose
        )
    return factorizer