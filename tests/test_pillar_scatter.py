import torch

from src.models.pillar_scatter import PillarScatter


def test_pillar_scatter():
    features = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    indices = torch.tensor([
        [1, 2],
        [3, 0],
    ])
    scatter = PillarScatter(num_x=4, num_y=3)
    bev = scatter(features, indices)
    assert bev.shape == (2, 3, 4)
    assert torch.equal(bev[:, 2, 1], torch.tensor([1.0, 2.0]))
    assert torch.equal(bev[:, 0, 3], torch.tensor([3.0, 4.0]))

if __name__ == '__main__':
    test_pillar_scatter()
    print(f"All tests for Pillar Scatter passed")
