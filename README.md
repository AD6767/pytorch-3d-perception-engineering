# PyTorch 3D Perception Engineering

A practical learning project for building and debugging 3D perception
pipelines in PyTorch.

## Current scope

* Synthetic point-cloud generation
* ModelNet10 surface-point sampling
* Stratified dataset splitting
* Point-cloud normalization
* PointNet classification
* Shared PyTorch training and evaluation pipeline
* Per-class accuracy and confusion-matrix analysis
* Rotation and jitter robustness experiments

## Completed milestones

1. Synthetic shape classification with PointNet
2. ModelNet10 classification with 512 sampled surface points
3. Baseline versus rotation-and-jitter augmented training
4. Robustness and class-wise evaluation

## Planned progression

1. PointNet++ fundamentals and local neighborhoods
2. Point-cloud segmentation
3. LiDAR coordinate geometry
4. PointPillars detection
5. Camera–LiDAR fusion

## Initial synthetic baseline

The minimal PointNet reached 100% test accuracy on the balanced sphere/cube/cylinder classification task.

## Robustness results

| Model | Z rotation | 3D rotation | Jitter |
|---|---:|---:|---:|
| Baseline | 73.33% | 71.11% | 100.00% |
| Rotation augmented | 93.33% | 93.33% | 91.11% |

## ModelNet10 PointNet baseline

A minimal PointNet classifier was trained on ModelNet10 using 512 surface points per object. The pipeline uses the official training and test splits, a stratified validation split, unit-radius normalization, best-validation checkpointing, and class-wise evaluation.

### Results

| Model                                  | Clean test accuracy | Rotate + jitter accuracy |
| -------------------------------------- | ------------------: | -----------------------: |
| Baseline PointNet                      |              87.67% |                   40.64% |
| Rotation-and-jitter augmented PointNet |              80.18% |                   79.30% |

The baseline performs well on canonically aligned ModelNet10 objects but degrades substantially under random Z-axis rotation and coordinate jitter. Training-time augmentation improves corrupted-test accuracy by 38.66 percentage points and reduces the clean-to-corrupted robustness gap to less than one percentage point.

The result also demonstrates a trade-off: robustness augmentation reduces clean aligned-test accuracy by 7.49 percentage points.

Common class confusions include:

* Desk and table
* Dresser and nightstand
* Bathtub and bed

These categories have similar normalized geometry, highlighting the limitations of global PointNet features and the removal of absolute scale during preprocessing.

