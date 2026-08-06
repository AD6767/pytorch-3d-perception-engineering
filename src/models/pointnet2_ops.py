import torch


def pairwise_squared_distance(source: torch.Tensor, 
                              target: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.
    Args:
        source: Tensor with shape [B, N, C].
        target: Tensor with shape [B, M, C].
    Returns:
        Tensor with shape [B, N, M].
    """
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source and target must both have shape [B, N, C]")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same batch size")
    if source.shape[2] != target.shape[2]:
        raise ValueError("source and target must have the same feature dimension")

    B, N, C = source.shape
    M = target.shape[1]
    source_t = source[:, :, None, :] # (B, N, 1, C)
    target_t = target[:, None, :, :] # (B, 1, M, C)
    distances = (source_t - target_t)**2 # (B, N, M, C)
    distances = torch.sum(distances, dim=-1) # (B, N, M)
    return distances

def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather points independently for each batch.
    Args:
        points: Tensor with shape [B, N, C].
        indices: Long tensor with shape [B, ...].
    Returns:
        Indexed points with shape [B, ..., C].
    """
    if points.ndim != 3:
        raise ValueError(f"points must have shape [B, N, C]. Current shape={points.shape}")
    if indices.ndim < 2:
        raise ValueError(f"indices must include a batch dimension. Current shape={indices.shape}")
    if points.shape[0] != indices.shape[0]:
        raise ValueError(
            f"points and indices must have the same batch size {points.shape[0]}!={indices.shape[0]}")
    if indices.dtype != torch.long:
        raise ValueError(f"indices must have dtype torch.long. Current={indices.dtype}")

    B = points.shape[0]
    C = points.shape[-1]
    original_index_shape = indices.shape[1:]
    flat_indices = indices.reshape(B, -1)  # [B, S*K]
    flat_indices = flat_indices.unsqueeze(dim=-1) # (B, S*K, 1)
    expanded_indices = flat_indices.expand(size=(-1, -1, C)) # (B, S*K, C)
    selected = torch.gather(points, dim=1, index=expanded_indices) # (B, S*K, C)
    # `*`: Take the items inside this sequence and insert them here as separate values
    return torch.reshape(selected, shape=(B, *original_index_shape, C))
