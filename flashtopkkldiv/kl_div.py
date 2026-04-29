import torch
import torch.nn.functional as F
from .sparse_index_matmul import sparse_index_matmul_lib


def original(
    embedding: torch.Tensor,
    hidden_state: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence computed via explicit topk embedding lookup (with intermediate tensors).

    Args:
        embedding: Teacher embedding matrix [V, D]
        hidden_state: Student hidden states [B*S, D] or [B, S, D]
        teacher_indices: Top-K indices from teacher logits [B*S, K]
        teacher_probs: Softmax probabilities for top-K entries [B*S, K]
        temperature: Temperature scaling for softmax
        reduction: One of "none", "mean", "sum"

    Returns:
        KL divergence loss
    """
    total_elements = hidden_state.view(-1, hidden_state.shape[-1]).shape[0]

    # Project student hidden states via embedding matrix multiplication
    s_flat = hidden_state.view(total_elements, -1)
    s_transT = s_flat @ embedding.T  # [total_elements, V] — intermediate tensor

    # Lookup top-k logits via torch.gather (memory heavy — materializes intermediate tensors)
    s_topk_logits = torch.gather(s_transT, dim=1, index=teacher_indices)  # [total_elements, K]

    s_topk_log_probs = F.log_softmax(s_topk_logits / temperature, dim=-1)
    kl_loss = F.kl_div(s_topk_log_probs, teacher_probs, reduction="none")

    return _reduce_kl(kl_loss, reduction)


def fast(
    embedding: torch.Tensor,
    hidden_state: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence computed via sparse index matrix multiplication (memory efficient).

    Uses Triton kernel to compute dot(hidden_state, E[idx]) without intermediate tensors.

    Args:
        embedding: Teacher embedding matrix [V, D]
        hidden_state: Student hidden states [B*S, D] or [B, S, D]
        teacher_indices: Top-K indices from teacher logits [B*S, K]
        teacher_probs: Softmax probabilities for top-K entries [B*S, K]
        temperature: Temperature scaling for softmax
        reduction: One of "none", "mean", "sum"

    Returns:
        KL divergence loss
    """
    total_elements = hidden_state.view(-1, hidden_state.shape[-1]).shape[0]

    # Direct sparse matmul with hidden state — no intermediate tensors
    s_flat = hidden_state.view(total_elements, -1)
    s_topk_logits = sparse_index_matmul_lib(s_flat, embedding, teacher_indices)  # [total_elements, K]

    s_topk_log_probs = F.log_softmax(s_topk_logits / temperature, dim=-1)
    kl_loss = F.kl_div(s_topk_log_probs, teacher_probs, reduction="none")

    return _reduce_kl(kl_loss, reduction)


def _reduce_kl(kl_loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Helper to apply reduction to KL loss."""
    if reduction == "mean":
        return kl_loss.mean()
    elif reduction == "sum":
        return kl_loss.sum()
    else:  # "none"
        return kl_loss.detach()
