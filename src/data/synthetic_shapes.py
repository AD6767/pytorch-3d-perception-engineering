import torch

from torch.utils.data import Dataset

CLASSES = {"sphere": 0, "cube": 1, "cylinder": 2}
CLASS_IDX_TO_LABEL = {0: "sphere", 1: "cube", 2: "cylinder"}

def generate_cube(num_points: int, 
                  side_length: float = 1., 
                  center: torch.Tensor = torch.tensor([0.,0.,0.]), 
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate a solid cube point cloud"""
    points = torch.rand(size=(num_points, 3), 
                        generator=generator, 
                        dtype=torch.float32) # [N, 3] generates between 0., 1.
    # scale it as per the side length
    points = points * side_length
    # shift as per center
    points = points + center - (side_length / 2)
    return points # [N, 3]

def generate_sphere(num_points: int,
                    radius: float = 1.,
                    center: torch.Tensor = torch.tensor([0., 0., 0.]),
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate a solid sphere as per the radius"""
    directions = torch.randn(size=(num_points, 3), generator=generator)
    norm = torch.linalg.norm(directions, dim=1, keepdim=True) # [N,] :along x, y, z direction (for each point).
    unit_directions = directions / (norm + 1e-8)
    # Uniform volumetric radius distribution (cube root handles density cluster fixes)
    # we do not see a a lot of points squeezed around the center but uniformly distributed outside as well.
    r = radius * (torch.rand(size=(num_points, 1), generator=generator) ** (1.0 / 3.0)) # cube root
    return (unit_directions * r) + center

def generate_cylinder(num_points: int,
                    radius: float = 1.,
                    height: float = 1.5,
                    center: torch.Tensor = torch.tensor([0., 0., 0.]),
                    generator: torch.Generator | None = None) -> torch.Tensor:
    # Sample random angles and radius parameters for the base circular cross-section
    theta = torch.rand(size=(num_points,), generator=generator) * 2.0 * torch.pi # [0., 360.] degrees
    r = radius * (torch.rand(size=(num_points,), generator=generator) ** (1.0 / 2.0)) # sqrt
    # Compute base coordinates
    x = r * torch.cos(theta) # [N,]
    y = r * torch.sin(theta) # [N,]
    # Uniform bounding range distribution along height (Z axis)
    z = (torch.rand(size=(num_points,), generator=generator) * height) - (height / 2.0) # [N,]
    points = torch.stack([x, y, z], dim=1) # (N, 3)
    return points + center


class SyntheticPointCloudDataset(Dataset):
    def __init__(self,
                 num_samples: int = 300,
                 points_per_shape: int = 2048,
                 seed: int | None = None,
                 transform=None):
        super().__init__()
        self.num_samples = num_samples
        self.points_per_shape = points_per_shape
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def get_label(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} outside dataset of size {len(self)}")
        return index % 3

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # randomness
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)
        # randomly pick between sphere, cube, cylinder to be generated --> might create class imbalance
        # idx = torch.randint(0, 3, size=(), generator=self.generator).item()
        class_label = index % 3
        # random center
        center = (torch.rand(3, generator=generator) * 4.0) - 2.0 # range [-2.0, 2.0]

        # sphere
        if class_label == 0:
            r = (torch.rand(1, generator=generator).item() * 0.4) + 0.8  # Radius: [0.8, 1.2]
            points = generate_sphere(num_points=self.points_per_shape,
                                     radius=r,
                                     center=center,
                                     generator=generator)
        # cube
        elif class_label == 1:
            side_length = (torch.rand(1, generator=generator).item() * 0.8) + 1.6 # range [1.6, 2.4]
            points = generate_cube(num_points=self.points_per_shape, 
                                   side_length=side_length, 
                                   center=center,
                                   generator=generator)
        # sphere
        elif class_label == 2:
            r = (torch.rand(1, generator=generator).item() * 0.4) + 0.8  # Radius: [0.8, 1.2]
            h = (torch.rand(1, generator=generator).item() * 0.8) + 1.6  # Height: [1.6, 2.4]
            points = generate_cylinder(num_points=self.points_per_shape,
                                       radius=r,
                                       height=h,
                                       center=center,
                                       generator=generator)
        else:
            raise ValueError(f"Invalid Class Label {class_label}")

        # Apply optional preprocessing or augmentation.
        if self.transform is not None:
            points = self.transform(points)

        return points, torch.tensor(class_label, dtype=torch.int64)
        
