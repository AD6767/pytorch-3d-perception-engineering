## Current scope

* Synthetic and real point-cloud datasets
* Point-cloud normalization and augmentation
* PointNet classification
* PointNet++ local neighborhood learning
* Shared PyTorch training and evaluation pipeline
* Per-class accuracy and confusion-matrix analysis
* Rotation and jitter robustness experiments
* Point-cloud sampling and neighborhood operations

## Completed milestones

1. Synthetic shape classification with PointNet
2. ModelNet10 classification with 512 sampled surface points
3. PointNet robustness evaluation and augmentation
4. PointNet++ fundamentals

   * Batched pairwise distances
   * Batched point indexing
   * Farthest-point sampling
   * Radius-based neighborhood grouping
   * Center-relative local coordinates
5. Minimal hierarchical PointNet++ classifier
6. PointNet vs. PointNet++ evaluation on ModelNet10

## Planned progression

1. LiDAR coordinate geometry and point-cloud processing
2. Bird's-eye-view (BEV) representations
3. PointPillars detection
4. Camera–LiDAR geometry and projection
5. Camera–LiDAR fusion

## Next milestone: LiDAR perception

The next phase moves from object-centric point clouds to scene-level LiDAR perception.

The focus will be on:

* LiDAR point representation `(x, y, z, intensity)`
* Sensor and vehicle coordinate systems
* Rigid transformations between coordinate frames
* Spatial filtering and range constraints
* Voxel and pillar indexing
* Bird's-eye-view representations
* PointPillars-style feature encoding and detection
* Camera projection and multimodal fusion

Point-cloud segmentation is intentionally deferred. Per-point and dense-prediction concepts will be revisited where they naturally appear in later LiDAR perception tasks.
