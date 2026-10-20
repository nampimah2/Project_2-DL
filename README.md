# CIFAR-100 Classification: Compact vs. Deep CNNs

A comparison of two convolutional neural networks trained from scratch on CIFAR-100. The project evaluates classification performance, uncertainty, training behavior, class confusion, and Integrated Gradients attributions.

**Authors:** Yin Po Po Aung, Jane Ohia, and Nicholas Ampimah  
**Affiliation:** University of Nebraska–Lincoln

## Dataset and task

- Input: 32 × 32 RGB images, scaled to [0, 1].
- Output: probabilities over 100 fine-grained classes.
- Training/validation/test sizes: 45,000 / 5,000 / 10,000 images.
- The training/validation split is stratified with seed 42.
- Training augmentation: zero-padding to 40 × 40, random cropping to 32 × 32, and horizontal flipping.
- Validation and test images receive scaling only.

## Architectures

Each convolution uses a 3 × 3 kernel, stride 1, same padding, no bias, batch normalization, and ReLU. Max pooling uses a 2 × 2 window and stride 2. Shapes omit the batch dimension.

| Stage | Compact CNN | Deep CNN |
|---|---|---|
| Input | 32 × 32 × 3 | 32 × 32 × 3 |
| Stage 1 | Conv32 → Conv32 → Pool → Dropout(d/2) | Conv32 → Conv32 → Pool → Dropout(d/2) |
| Stage 1 output | 16 × 16 × 32 | 16 × 16 × 32 |
| Stage 2 | Conv64 → Pool → Dropout(d) | Conv64 → Conv64 → Pool → Dropout(d/2) |
| Stage 2 output | 8 × 8 × 64 | 8 × 8 × 64 |
| Stage 3 | — | Conv128 → Conv128 → Pool → Dropout(d/2) |
| Feature map before aggregation | 8 × 8 × 64 | 4 × 4 × 128 |
| Aggregation | Global average pooling → 64 features | Global average pooling → 128 features |
| Hidden layer | Dense128 → ReLU → Dropout(d) | Dense256 → ReLU → Dropout(d) |
| Output | Dense100 → Softmax | Dense100 → Softmax |
| Total parameters | 50,244 | 347,076 |
| Trainable parameters | 49,988 | 346,180 |

## Experimental settings

Each architecture is trained under two joint hyperparameter settings:

| Setting | Initial learning rate | Dropout d | L2 strength |
|---|---:|---:|---:|
| A | 0.001 | 0.30 | 0.0001 |
| B | 0.0003 | 0.50 | 0.0005 |

Training uses Adam, categorical cross-entropy, batch size 128, and a maximum of 100 epochs. Early stopping monitors validation loss with patience 8 and minimum improvement 0.0001. A plateau scheduler halves the learning rate after four epochs without improvement, down to 0.000001. The best checkpoint from each run is evaluated; the overall model is selected by minimum validation loss.

L2 applies to convolutional and hidden dense kernels. The final classifier kernel has no explicit L2 penalty. Reported loss includes regularization.

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs the four experiments, saves checkpoints and histories, and selects the best model |
| `model.py` | Defines and compiles the compact and deep CNNs |
| `util.py` | Loads data, applies augmentation, evaluates predictions, computes confidence intervals, and generates figures |
| `CIFAR100_IEEE_Report.pdf` | Three-page report |
| `CIFAR100_IEEE_Report.tex` | Editable LaTeX report source |
| `IEEEtran.cls` | LaTeX document class included with the editable report |
| `figures/` | Figure assets used by the LaTeX report |

Keep the three Python files in the same directory. Add the report and figure assets if you want them included in the repository.

## Installation

Use a Python environment with the following packages:

```bash
python -m pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn
```

Package versions are not pinned in this project. Exact numerical results can vary with software versions, hardware, and training randomness. TensorFlow Datasets downloads CIFAR-100 on the first run, requiring internet access and available disk space.

## Run the full experiment

From the directory containing the three Python files:

```bash
python main.py --output-dir results --epochs 100 --patience 8 --batch-size 128
```

For a shared compute environment, the code defaults to two TensorFlow/data-pipeline threads:

```bash
python main.py --output-dir results --epochs 100 --patience 8 --batch-size 128 --threads 2
```

The current script starts training from scratch on each invocation; it does not implement a `--resume` option. Use a different output directory to retain an earlier run's files.

## Generated outputs

The output directory contains:

- `results.csv`: test metrics and 95% Wilson confidence intervals for all runs.
- `run_config.json`: command-line settings.
- `*_epochs.csv`: per-epoch training and validation histories.
- `*.keras`: best-validation-loss model checkpoints.
- `training_curves.png`: training/validation loss and accuracy versus epoch.
- `normalized_confusion_matrix.png`: row-normalized test confusion matrix for the selected model.
- `integrated_gradients_examples.png`: original images, attribution maps, and overlays for the selected model.
- `ig_example_selection.json`: selected model and attribution-example details.

## Reported results

Results below come from the supplied experiment outputs, using 10,000 held-out test images.

| Model | Epochs / best epoch | Top-1 accuracy | 95% accuracy CI | Top-5 accuracy |
|---|---:|---:|---:|---:|
| Compact A | 66 / 58 | 37.24% | 36.30–38.19% | 70.43% |
| Compact B | 100 / 96 | 29.05% | 28.17–29.95% | 61.46% |
| **Deep A** | **81 / 73** | **61.24%** | **60.28–62.19%** | **87.47%** |
| Deep B | 99 / 91 | 55.92% | 54.94–56.89% | 84.61% |

Deep A was selected by its lowest validation loss (1.5585). It improves top-1 accuracy over Compact A by 24.00 percentage points. Its test error is 38.76%, with a 95% Wilson interval of 37.81–39.72%.

## Integrated Gradients

Attributions target the predicted-class softmax probability, using a black baseline and 64 trapezoidal integration intervals. Selection chooses the first class by ID with at least two correct predictions and one error, then selects:

1. The highest-confidence correct prediction.
2. The lowest-confidence correct prediction.
3. The highest-confidence incorrect prediction.

In the supplied results, the shared true class is apple. The incorrect example is predicted as pear with 91.38% confidence. Displayed maps sum absolute attributions across RGB channels and normalize each image separately. They show relative sensitivity within each image; they do not retain attribution sign or support comparisons of absolute magnitude between examples.

## Limitations

- Results use one seed; confidence intervals describe test-sample uncertainty, not variability from retraining.
- Learning rate, dropout, and L2 change together, so their individual effects cannot be separated.
- The architectures differ in depth, width, hidden-layer size, and dropout placement; the performance gap cannot be attributed to depth alone.
- Compact B reaches the 100-epoch cap four epochs after its best checkpoint, before completing the full eight-epoch stopping window.
- A normalized confusion matrix is generated only for the selected model.
- Runtime and memory usage were not systematically measured.

## Compile the report

Keep the LaTeX source, `IEEEtran.cls`, and `figures/` together. Upload the complete editable ZIP to Overleaf, or compile locally:

```bash
pdflatex CIFAR100_IEEE_Report.tex
pdflatex CIFAR100_IEEE_Report.tex
```

## References

- A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. University of Toronto, 2009. [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
- M. Sundararajan, A. Taly, and Q. Yan. *Axiomatic Attribution for Deep Networks*. ICML, 2017.
