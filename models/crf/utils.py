import torch


UNLABELED_INDEX = -1  # TODO Make UNLABELED_INDEX a configurable value
IMPOSSIBLE_SCORE = -10000000.0


def create_possible_tag_masks(num_tags: int, tags: torch.Tensor, unlabeled_index: int = None) -> torch.Tensor:
    if unlabeled_index is None:
        unlabeled_index = UNLABELED_INDEX
    copy_tags = tags.clone()
    no_annotation_idx = copy_tags == unlabeled_index
    copy_tags[copy_tags == unlabeled_index] = 0

    tags_ = torch.unsqueeze(copy_tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags, dtype=torch.uint8, device=tags.device)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks


def log_sum_exp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()
