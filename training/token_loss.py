"""Assistant-only causal targets, with equal weight per sequence."""
import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def make_labels(input_ids, assistant_mask, attention_mask=None):
    """The caller supplies the token-aligned supervision mask, not role guesses."""
    if (input_ids.ndim != 2 or input_ids.dtype != torch.long
            or input_ids.shape[0] == 0 or input_ids.shape[1] < 2):
        raise ValueError("input_ids must be an int64 [batch, sequence] tensor")
    if assistant_mask.shape != input_ids.shape or assistant_mask.dtype != torch.bool:
        raise ValueError("assistant_mask must be a boolean tensor aligned with input_ids")
    mask = assistant_mask.clone()
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must be binary")
        mask &= attention_mask.bool()
    if mask[:, 0].any():
        raise ValueError("The first token has no causal prediction position")
    if (mask.sum(dim=-1) == 0).any():
        raise ValueError("Every sequence must contain supervised assistant tokens")
    return input_ids.masked_fill(~mask, IGNORE_INDEX)


def supervised_token_loss(logits, targets, chunk_size=128):
    """Mean of each sequence's supervised-token mean, not a pooled token mean."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("Expected [B, L, V] logits and int64 [B, L] targets")
    if chunk_size <= 0 or targets.shape[0] == 0:
        raise ValueError("Positive chunk size and nonempty batch required")
    sequence_losses = []
    for row_logits, row_targets in zip(logits, targets, strict=True):
        positions = row_targets.ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("Cannot train on an unsupervised sequence")
        total = row_logits.new_zeros((), dtype=torch.float32)
        for index in positions.split(chunk_size):
            total = total + F.cross_entropy(row_logits[index].float(), row_targets[index], reduction="sum")
        sequence_losses.append(total / positions.numel())
    return torch.stack(sequence_losses).mean()


def causal_loss(logits, labels, chunk_size=128):
    """Dense reference implementation of the same next-token objective."""
    if labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Logits and unshifted labels must be aligned")
    if labels[:, 0].ne(IGNORE_INDEX).any():
        raise ValueError("The first label must be masked")
    return supervised_token_loss(logits[:, :-1], labels[:, 1:], chunk_size)


def sparse_targets(labels):
    """Select causal logit positions without computing unused vocabulary logits."""
    if labels.ndim != 2 or labels.shape[0] != 1 or labels[0, 0] != IGNORE_INDEX:
        raise ValueError("Sparse training expects one sequence with its first label masked")
    positions = labels[0, 1:].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("No assistant targets")
    return positions, labels[:, positions + 1]
