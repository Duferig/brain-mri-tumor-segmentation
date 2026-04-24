# Experiment Notes For Diploma

## Current Hardware And Constraints

- OS: Windows
- GPU: NVIDIA GeForce RTX 3060 12 GB
- RAM: 16 GB
- Storage: SSD/NVMe

The experiments were run in a constrained local environment. This affected:

- the maximum feasible `batch_size`
- the choice of `roi_size`
- the number of long training runs that could be completed
- the ability to fully tune heavier architectures

Because of these limits, the results below should be treated as practical research results obtained in a limited compute setup, not as the maximum achievable quality for the chosen architectures.

## Baseline Result

Model:

- `Baseline 3D U-Net`

Best observed result:

- `mean_dice = 0.5362`
- `best_epoch = 108`
- `dice_tc = 0.7382`
- `dice_wt = 0.8704`
- `dice_et = 0.0000`

Checkpoint:

- `artifacts/models/baseline_3060_12gb_quality_best.pt`

Interpretation:

- The baseline model produced a usable and stable result.
- The model segmented `WT` and `TC` relatively well.
- The weakest part was `ET`, which remained effectively unlearned in the current setup.

## Improved Result

Model:

- `Improved SwinUNETR`

Observed result during the current run:

- by epoch 45, the best observed `mean_dice` was `0.5202`

Interpretation:

- The improved architecture was heavier and more demanding.
- In the current compute setup it did not surpass the baseline result.
- The training curve suggested slow improvement, but not enough evidence of a future breakthrough over the baseline.

Practical conclusion:

- In the current configuration, `improved` did not justify its higher training cost.
- For the diploma, this can be described as an experiment where a more complex architecture did not outperform the simpler baseline under available computational constraints.

## Why The Results Are Moderate

The relatively modest results can be explained by a combination of factors:

- limited computational resources: one RTX 3060 with 12 GB VRAM and 16 GB RAM
- Windows-specific training overheads and data loading complexity
- the need to use conservative training settings for stability
- limited number of full experimental runs
- training from scratch instead of starting from strong pretrained weights

Important nuance:

- It is more accurate to say that the results were limited by the available compute budget and experiment scope, not that the models are inherently weak.

## Research Conclusion At This Stage

At the current stage of the project:

- `baseline` is the best trained model
- `improved` is useful as a comparison experiment
- the next rational step is not to keep training from scratch, but to fine-tune suitable pretrained weights

## Decision For The Next Stage

Next stage:

- fine-tuning pretrained weights

Why this decision is reasonable:

- faster convergence
- better starting representation quality
- higher probability of improving `ET`
- lower dependence on long training from scratch

## Concrete Transfer Learning Plan

Recommended starting point for the next experiments:

- MONAI bundle `brats_mri_segmentation`
- `SegResNet` as the fine-tuning architecture
- config `configs/train_transfer_monai_segresnet_3060_12gb.toml`

Suggested bundle download command:

- `.venv\Scripts\python.exe -c "from monai.bundle import download; download(name='brats_mri_segmentation', bundle_dir='./models')"`

Expected local weights path:

- `models/brats_mri_segmentation/models/model.pt`

Recommended fine-tuning defaults in the current hardware setup:

- `learning_rate = 5e-5`
- `epochs = 12`
- `patience = 4`
- validate every epoch
- keep `DiceCELoss(sigmoid=True)` because the BraTS target regions overlap

Practical note:

- This keeps the existing MONAI-based pipeline intact while replacing only the architecture choice and initialization strategy.

## Follow-Up Transfer Run

Planned follow-up experiment after the first transfer run:

- config `configs/train_transfer_monai_segresnet_3060_12gb_et_refine_v2.toml`
- initialization from `artifacts/models/transfer_segresnet_monai_3060_12gb_best.pt`
- lower learning rate for refinement
- `DiceFocalLoss` with higher weight for `ET`

Important artifact management note:

- the new run writes a separate checkpoint file
- old checkpoints are not deleted or overwritten
- the new weights are intended to be treated as a separate experimental version

## Suggested Diploma Wording

Possible neutral wording for the thesis:

> In the course of the experiments, two architectures were evaluated: a baseline 3D U-Net model and an improved SwinUNETR model. The baseline model demonstrated the best practical result in the available computational environment, achieving a mean Dice score of 0.5362. The improved architecture, despite its higher representational capacity, did not outperform the baseline under the given constraints.

Another version:

> The obtained segmentation quality should be considered moderate. The main limiting factors were the restricted computational resources of the experimental platform, including a single RTX 3060 GPU with 12 GB VRAM, 16 GB RAM, and the necessity to use conservative training settings. Under such conditions, the baseline architecture proved more efficient than the more computationally intensive improved model.

Transition to the next section:

> Taking into account the achieved results and the limitations of training from scratch, the subsequent stage of the study focuses on fine-tuning pretrained weights as a more promising strategy for improving segmentation quality.

## Notes For Final Writing

When writing the diploma text, it is better to state:

- the baseline result was the strongest among the completed runs
- the improved model was tested and analyzed, but did not exceed the baseline
- the current results are useful as a research baseline
- the next development direction is transfer learning / fine-tuning

It is better to avoid overly absolute wording such as:

- "the improved model is bad"
- "the method does not work"
- "the results are poor only because the GPU is weak"

It is better to write:

- "the achieved quality was limited by the available computational resources"
- "the experimental setup constrained the size and number of training runs"
- "the use of pretrained weights is justified as the next step of the study"
