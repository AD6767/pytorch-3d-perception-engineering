# PyTorch 3D Perception Engineering

A practical PyTorch project for understanding core 3D perception concepts through implementation, debugging, and controlled experiments.

The focus is on building intuition for point-cloud representations, robustness, local geometry, LiDAR processing, BEV representations, and PointPillars-style perception.

## Key concepts implemented

### Point-cloud classification

Implemented a minimal PointNet pipeline for:

* Synthetic point-cloud generation
* ModelNet10 surface-point sampling
* Point-cloud normalization
* Stratified train/validation splitting
* Shared PyTorch training and evaluation
* Per-class accuracy and confusion matrices
* Rotation and jitter robustness experiments

PointNet processes each point independently with shared MLPs and aggregates the complete point cloud using symmetric max pooling:

```text
[B, N, 3]
    ↓ shared point features
[B, N, C]
    ↓ max over points
[B, C]
    ↓ classifier
[B, num_classes]
```

The symmetric pooling operation makes PointNet permutation-invariant, but not inherently rotation-invariant.

---

### PointNet++ and local geometry

Implemented the fundamental PointNet++ operations directly in PyTorch:

* Batched pairwise squared distances
* Batched point indexing
* Farthest-point sampling
* Radius-based neighborhood grouping
* Center-relative local coordinates
* Shared local MLPs
* Neighborhood max pooling
* Hierarchical set abstraction

The core idea is to progressively learn features from local geometric neighborhoods:

```text
512 input points
    ↓
128 local centers
    ↓
32 higher-level centers
    ↓
global feature
    ↓
classification
```

The implementation is intentionally minimal and focuses on understanding the mechanics of local point-cloud learning rather than reproducing an optimized PointNet++ implementation.

---

## Classification and robustness findings

### Synthetic shapes

A minimal PointNet reached **100% clean test accuracy** on the balanced synthetic sphere/cube/cylinder classification task.

| Model                       | Z rotation | 3D rotation |  Jitter |
| --------------------------- | ---------: | ----------: | ------: |
| Baseline PointNet           |     73.33% |      71.11% | 100.00% |
| Rotation-augmented PointNet |     93.33% |      93.33% |  91.11% |

The experiment demonstrated that permutation invariance does not imply rotation invariance. Training-time rotation augmentation substantially improved orientation robustness.

### ModelNet10

All models use 512 sampled surface points and the official ModelNet10 test split.

| Model                                    | Clean test accuracy | Rotate + jitter accuracy |
| ---------------------------------------- | ------------------: | -----------------------: |
| Baseline PointNet                        |          **87.67%** |                   40.64% |
| Rotation-and-jitter augmented PointNet   |              80.18% |               **79.30%** |
| Minimal PointNet++                       |              75.33% |                   30.84% |
| Rotation-and-jitter augmented PointNet++ |              59.80% |                   47.91% |

The strongest robustness result came from the augmented PointNet:

* Robust accuracy improved from **40.64% → 79.30%**
* Clean accuracy decreased from **87.67% → 80.18%**
* The clean-to-corrupted gap was reduced to less than one percentage point

The experiments show a clear robustness trade-off: augmentation can reduce performance on canonically aligned data while substantially improving invariance to expected corruptions.

The minimal PointNet++ implementation did not outperform PointNet. This highlights that hierarchical point-cloud models are sensitive to sampling strategy, neighborhood radius, architectural choices, and training configuration.

Common ModelNet10 confusions included:

* Desk ↔ table
* Dresser ↔ nightstand
* Bathtub ↔ bed

These classes have similar normalized geometry, illustrating both feature ambiguity and the effect of removing absolute scale during preprocessing.

---

## LiDAR geometry

The project then moves from isolated object point clouds to scene-level LiDAR perception.

A LiDAR point is represented as:

```text
(x, y, z, intensity)
```

Unlike ModelNet object classification, LiDAR scenes preserve metric geometry. Absolute position and distance are meaningful and should not be normalized away.

The working physical coordinate convention is:

```text
+x → forward
+y → left
+z → up
```

Dataset-specific coordinate conventions must always be verified explicitly.

### Region-of-interest filtering

Implemented 3D spatial filtering using configured ranges for `x`, `y`, and `z`.

This reduces the raw LiDAR frame to the region relevant for downstream perception.

```text
raw LiDAR frame
    ↓
range filtering
    ↓
relevant scene points
```

---

## BEV and pillar representation

Continuous LiDAR coordinates are discretized into a 2D bird's-eye-view grid.

For a point `(x, y)`:

```text
pillar_x = floor((x - x_min) / pillar_size_x)
pillar_y = floor((y - y_min) / pillar_size_y)
```

This converts continuous metric positions into discrete pillar coordinates.

Example:

```text
x = 12.3 m
y = -4.7 m
pillar size = 0.5 m

→ pillar index = (24, 30)
```

### BEV indexing convention

Pillar indices use:

```text
[pillar_x, pillar_y]
```

while a PyTorch BEV tensor uses:

```text
bev[channel, row, column]

row    = pillar_y
column = pillar_x
```

Tensor indexing and physical coordinates therefore represent different conventions and should not be conflated.

---

## PointPillars-style feature generation

Points belonging to the same BEV cell are grouped into pillars.

Each point is represented using:

```text
[x, y, z, intensity]             # raw point features

[x - x_mean,
 y - y_mean,
 z - z_mean]                     # offset from pillar point cluster

[x - pillar_center_x,
 y - pillar_center_y]            # offset from geometric pillar center
```

giving **9 features per point**.

The two offsets encode complementary information:

* Cluster offsets describe a point relative to the other points in its pillar.
* Pillar-center offsets describe where the point lies inside the discrete BEV cell.

Variable-sized pillars are padded to a fixed number of points:

```text
[P, T, 9]
```

where:

* `P` = occupied pillars
* `T` = maximum points per pillar

A boolean mask distinguishes real points from padding.

---

## Pillar feature encoder

Implemented a minimal PointPillars-style pillar encoder:

```text
[P, T, 9]
    ↓ shared MLP
[P, T, C]
    ↓ masked max pooling
[P, C]
```

Conceptually, this behaves like a small PointNet applied independently to every occupied pillar.

Padding is masked before max pooling so padded points cannot influence the learned pillar representation.

---

## BEV pseudo-image

Each learned pillar feature is scattered back into its corresponding grid location:

```text
pillar features
[P, C]

      +

pillar coordinates
[P, 2]

      ↓ scatter

BEV feature map
[C, H, W]
```

For a pillar at `(pillar_x, pillar_y)`:

```text
bev[:, pillar_y, pillar_x] = pillar_feature
```

Empty BEV locations remain zero.

This is the key transition in PointPillars:

```text
irregular 3D point cloud
    ↓
pillars
    ↓
learned pillar features
    ↓
structured BEV pseudo-image
    ↓
standard 2D CNN processing
```

Once the representation reaches `[C, H, W]`, conventional `Conv2d` backbones can be used efficiently for scene-level perception.

---

## Current direction

The next components are:

* BEV convolutional backbone
* PointPillars-style detection head
* 3D bounding-box prediction concepts
* Camera–LiDAR projection
* Camera–LiDAR fusion

