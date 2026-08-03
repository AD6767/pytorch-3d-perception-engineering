# PyTorch 3D Perception Engineering

A practical learning project for building and debugging 3D perception
pipelines in PyTorch.

## Current scope

- Synthetic point-cloud generation
- Dataset splitting
- Point-cloud normalization
- PointNet classification
- Training and evaluation
- Robustness experiments

## Planned progression

1. ModelNet10 classification
2. PointNet++ and local neighborhoods
3. Point-cloud segmentation
4. LiDAR coordinate geometry
5. PointPillars detection
6. Camera–LiDAR fusion

## Initial synthetic baseline

The minimal PointNet reached 100% test accuracy on the balanced sphere/cube/cylinder classification task.

## Robustness results

| Model | Z rotation | 3D rotation | Jitter |
|---|---:|---:|---:|
| Baseline | 73.33% | 71.11% | 100.00% |
| Rotation augmented | 93.33% | 93.33% | 91.11% |
