import torch


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    Center a point cloud and scale its farthest point to radius 1.
    Args:
        points: Tensor with shape [N, 3].
    Returns:
        Normalized tensor with shape [N, 3].
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    centroid = torch.mean(points, dim=0, keepdim=True) # (1, 3)
    points = points - centroid # (N, 3)

    max_dist = torch.linalg.norm(points, dim=1).max() # (1,)
    return points / torch.clamp(max_dist, min=1e-8)

def normalize_and_rotate_z_and_jitter(points: torch.Tensor) -> torch.Tensor:
    points = normalize_point_cloud(points)
    points = random_rotate_z(points)
    points = jitter_point_cloud(points, sigma=0.10)
    return points

def normalize_and_rotate_z(points: torch.Tensor) -> torch.Tensor:
    points = normalize_point_cloud(points)
    points = random_rotate_z(points)
    return points

def normalize_and_rotate_3d(points: torch.Tensor) -> torch.Tensor:
    points = normalize_point_cloud(points)
    points = random_rotate_3d(points)
    return points

def normalize_and_jitter(points):
    points = normalize_point_cloud(points)
    return jitter_point_cloud(points, sigma=0.10)

def normalize_and_subsample(points):
    points = normalize_point_cloud(points)
    return subsample_point_cloud(points, keep_fraction=0.10)

def normalize_and_add_outliers(points):
    points = normalize_point_cloud(points)
    return add_outlier_points(points, outlier_fraction=0.20, outlier_bound=2.0)

def random_rotate_z(points: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Rotate a point cloud around the z-axis.
    Args:
        points: [N, 3]
    Returns:
        Rotated points: [N, 3]
    """
    random_num = torch.rand(size=(), generator=generator) # range [0, 1)
    angle = random_num * 2.0 * torch.pi # [0, 360)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation_matrix = torch.tensor([ # [3x3]
        [cosine, -sine, 0.],
        [sine, cosine, 0.],
        [0., 0., 1.]
    ], dtype=points.dtype)
    return points @ rotation_matrix.T # [N, 3]

def random_rotate_3d(points: torch.Tensor,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Apply random rotations around the x, y, and z axes.
    This provides broad rotational variation, although it is not
    a mathematically uniform sample over all possible 3D rotations.
    """
    random_nums = torch.rand(size=(3,), generator=generator) # (3,)
    angles = random_nums * 2.0 * torch.pi
    x_angle, y_angle, z_angle = angles
    cos_x, sin_x = torch.cos(x_angle), torch.sin(x_angle)
    cos_y, sin_y = torch.cos(y_angle), torch.sin(y_angle)
    cos_z, sin_z = torch.cos(z_angle), torch.sin(z_angle)
    rotation_x = torch.tensor([
        [cos_x, -sin_x, 0.],
        [sin_x, cos_x, 0.],
        [0., 0., 1.]
    ], dtype=points.dtype)
    rotation_y = torch.tensor([
        [cos_y, -sin_y, 0.],
        [sin_y, cos_y, 0.],
        [0., 0., 1.]
    ], dtype=points.dtype)
    rotation_z = torch.tensor([
        [cos_z, -sin_z, 0.],
        [sin_z, cos_z, 0.],
        [0., 0., 1.]
    ], dtype=points.dtype)
    points = points @ rotation_x.T # (N, 3)
    points = points @ rotation_y.T # (N, 3)
    points = points @ rotation_z.T # (N, 3)
    return points

def jitter_point_cloud(points: torch.Tensor,
                       sigma: float = 0.01,
                       clip: float = 0.05,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Add clipped Gaussian noise to point coordinates.
    Args:
        points: [N, 3]
        sigma: Standard deviation of the noise.
        clip: Maximum absolute noise value.
    """
    noise = torch.randn(size=(points.shape), generator=generator)
    noise = noise * sigma
    noise = torch.clamp(noise, min=-clip, max=clip)
    return points + noise

def subsample_point_cloud(points: torch.Tensor,
                          keep_fraction: float,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Randomly retain a fraction of points.
    Input:
        [N, 3]
    Output:
        [num_kept, 3]
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in the interval (0, 1]")
    N = points.shape[0]
    M = round(keep_fraction * N)
    random_indices = torch.randperm(n=N, generator=generator)[:M] # length M
    points = points[random_indices] # (M, 3)
    return points

def add_outlier_points(points: torch.Tensor, 
                       outlier_fraction: float = 0.05,
                       outlier_bound: float = 1.5,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Replace a fraction of points with uniformly sampled outliers.
    """
    if not 0.0 <= outlier_fraction <= 1.0:
        raise ValueError("outlier_fraction must be between 0 and 1")
    corrupted_points = points.clone()
    num_points = points.shape[0]
    num_outliers = round(num_points * outlier_fraction)
    if num_outliers == 0:
        return corrupted_points
    outlier_indices = torch.randperm(num_points, generator=generator)[:num_outliers]
    outliers = (torch.rand(size=(num_outliers, 3),generator=generator,dtype=points.dtype,) * 2.0 * outlier_bound - outlier_bound)
    corrupted_points[outlier_indices] = outliers
    return corrupted_points