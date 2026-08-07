# PyTorch 3D Perception Engineering

A practical learning project for building, debugging, and evaluating 3D perception pipelines in PyTorch.

## Current scope

* Synthetic and real point-cloud datasets
* Point-cloud normalization and augmentation
* Stratified train/validation splitting
* PointNet classification
* PointNet++ local neighborhood learning
* Shared PyTorch training and evaluation pipeline
* Per-class accuracy and confusion-matrix analysis
* Rotation and jitter robustness experiments

## Completed milestones

1. Synthetic shape classification with PointNet
2. ModelNet10 classification with 512 sampled surface points
3. PointNet robustness evaluation and augmentation
4. PointNet++ fundamentals
   * Batched pairwise distances
   * Batched point indexing
   * Farthest-point sampling
   * Radius-based neighborhood grouping
   * Local coordinate construction
5. Minimal hierarchical PointNet++ classifier
6. PointNet vs. PointNet++ evaluation on ModelNet10

## Planned progression

1. Point-cloud segmentation
2. LiDAR coordinate geometry
3. PointPillars detection
4. Camera–LiDAR fusion

## Synthetic point-cloud baseline

A minimal PointNet classifier was trained on a balanced synthetic dataset containing spheres, cubes, and cylinders.

The model reached **100% clean test accuracy**.

### Robustness

| Model                       | Z rotation | 3D rotation |  Jitter |
| --------------------------- | ---------: | ----------: | ------: |
| Baseline PointNet           |     73.33% |      71.11% | 100.00% |
| Rotation-augmented PointNet |     93.33% |      93.33% |  91.11% |

The experiment demonstrates that PointNet is permutation-invariant but not inherently rotation-invariant. Training-time rotation augmentation substantially improved robustness to unseen orientations.

## ModelNet10 classification

The synthetic pipeline was extended to ModelNet10 using 512 surface points sampled from each CAD model.

The training pipeline uses:

* Official ModelNet10 training and test splits
* Stratified training and validation split
* Unit-radius point-cloud normalization
* Shared training and evaluation utilities
* Best-validation checkpointing
* Per-class accuracy and confusion matrices

### PointNet robustness

| Model                                  | Clean test accuracy | Rotate + jitter accuracy |
| -------------------------------------- | ------------------: | -----------------------: |
| Baseline PointNet                      |          **87.67%** |                   40.64% |
| Rotation-and-jitter augmented PointNet |              80.18% |               **79.30%** |

The baseline performs well on canonically aligned ModelNet10 objects but degrades substantially under rotation and coordinate jitter.

Training-time augmentation improves corrupted-test accuracy by **38.66 percentage points**, while reducing clean aligned accuracy by **7.49 percentage points**.

Common class confusions include:

* Desk and table
* Dresser and nightstand
* Bathtub and bed

These categories have similar normalized geometry, illustrating both the limitations of global point features and the effect of removing absolute scale during normalization.

## PointNet++ fundamentals

A compact PointNet++ implementation was built from basic PyTorch tensor operations rather than relying on optimized point-cloud libraries.

The implementation includes:
* Pairwise squared Euclidean distances
* Batched point indexing
* Farthest-point sampling
* Radius-based neighborhood queries
* Center-relative local coordinates
* Shared local MLPs
* Neighborhood max pooling
* Hierarchical feature aggregation

The minimal hierarchy reduces:
```text
512 input points
→ 128 local centers
→ 32 higher-level centers
→ global feature
→ classification
```

### PointNet vs. PointNet++

| Model              | Representation              | ModelNet10 clean test accuracy |
| ------------------ | --------------------------- | -----------------------------: |
| PointNet           | Global point features       |                     **87.67%** |
| Minimal PointNet++ | Hierarchical local features |                         75.33% |

The minimal PointNet++ implementation did not outperform the PointNet baseline.

This does not imply that PointNet++ is generally inferior. Instead, it highlights that hierarchical point-cloud models depend strongly on neighborhood radius, sampling strategy, architecture configuration, preprocessing, and training setup.

The implementation in this repository is intentionally compact and focuses on understanding the core mechanics of local neighborhood learning rather than reproducing the full optimized PointNet++ architecture.

### PointNet++ robustness observation

The unaugmented PointNet++ model dropped from **75.33%** clean accuracy to **30.84%** under rotation and jitter.

Training with rotation and jitter improved corrupted-test accuracy to **47.91%**, but further PointNet++ tuning is intentionally left outside the scope of this project.

## Next milestone

The next phase moves from object-level classification to **point-level segmentation**:

```text
classification:
[B, N, 3] → [B, num_classes]

segmentation:
[B, N, 3] → [B, N, num_classes]
```

The goal is to understand per-point prediction and local/global feature fusion before moving into LiDAR geometry and 3D detection.
