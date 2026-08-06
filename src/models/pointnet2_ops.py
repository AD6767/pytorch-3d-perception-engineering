from numpy import indices
from sklearn import neighbors
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

def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Select well-spaced points using farthest-point sampling.
    Args:
        points: Point coordinates with shape [B, N, C].
        num_samples: Number of points to select.
    Returns:
        Selected point indices with shape [B, num_samples].
    """
    if points.ndim != 3:
        raise ValueError(f"points must have shape [B, N, C] {points.shape}")
    B, N, C = points.shape
    if not 1 <= num_samples <= N:
        raise ValueError(f"num_samples must be between 1 and {N}")

    indices = torch.zeros(size=(B, num_samples), dtype=torch.long, device=points.device) # (B, num_samples)
    # Min Squared distance from any point to any point that has been selected
    min_distances = torch.full(size=(B, N), fill_value=float("inf"), device=points.device) # (B, N)
    # Use point 0 as deterministic initial point
    farthest_indices = torch.zeros(size=(B,), dtype=torch.long, device=points.device) # (B,)
    batch_indices = torch.arange(start=0, end=B, step=1, dtype=torch.long, device=points.device)
    for idx in range(num_samples):
        indices[:, idx] = farthest_indices
        selected_points = points[batch_indices, farthest_indices] # (B, C)
        selected_points = torch.unsqueeze(selected_points, dim=1) # (B, 1, C)
        squared_distances = torch.sum((points - selected_points)**2, dim=-1) # (B, N)
        min_distances = torch.minimum(min_distances, squared_distances) # (B, N,)
        farthest_indices = torch.max(min_distances, dim=-1).indices # (B,)
    return indices

def get_top_k_indices(masked_distances: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # masked_distances (B, S, N): contain infinity for invalid distances
    sorted_distances, indices = torch.sort(masked_distances, dim=-1) # along the point dim (B, S, N)
    # keep only k=max_neighbors
    sorted_distances = sorted_distances[:, :, :k] # (B, S, k)
    indices = indices[:, :, :k] # (B, S, k)
    return sorted_distances, indices

def query_ball_point(points: torch.Tensor, 
                     centers: torch.Tensor, 
                     radius: float,
                     max_neighbors: int) -> torch.Tensor:
    """Find local point neighborhoods around center points.
    Args:
        points: All point coordinates with shape [B, N, C].
        centers: Center coordinates with shape [B, S, C].
        radius: Maximum neighborhood radius.
        max_neighbors: Maximum number of neighbors per center.
    Returns:
        Neighbor indices with shape [B, S, max_neighbors].
    """
    if points.ndim != 3 or centers.ndim != 3:
        raise ValueError("points and centers must have shape [B, N, C]")
    if points.shape[0] != centers.shape[0]:
        raise ValueError("points and centers must have the same batch size")
    if points.shape[2] != centers.shape[2]:
        raise ValueError("points and centers must have the same coordinate dimension")
    if radius <= 0:
        raise ValueError("radius must be positive")
    N = points.shape[1]
    if not 1 <= max_neighbors <= N:
        raise ValueError("max_neighbors must be between 1 and num_points")
    # distance from every center to all points
    # distances = torch.sum((centers[:, :, None, :] - points[:, None, :, :])**2, dim=-1) # [B, S, N]
    distances = pairwise_squared_distance(centers, points)
    # keep distances <= radius
    within_radius = distances <= radius**2 # (B, S, N)
    # fill inf value to add distances > radius
    masked_distances = distances.masked_fill(~within_radius, float("inf")) # (B, S, N)
    # (B, S, max_neighbors)
    neighbor_distances, neighbor_indices = get_top_k_indices(masked_distances=masked_distances,
                                                             k=max_neighbors)
    nearest_indices = distances.argmin(dim=-1, keepdim=True) # (B, S, 1)
    nearest_indices = nearest_indices.expand(-1, -1, max_neighbors) # (B, S, max_neighbors)
    invalid_neighbors = neighbor_distances == float("inf") # (B, S, max_neighbors)
    neighbor_indices = torch.where(invalid_neighbors, nearest_indices, neighbor_indices) # (B, S, max_neighbors)
    return neighbor_indices
