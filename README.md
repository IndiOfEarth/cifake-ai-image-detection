# What Do Neural Networks Actually Detect in AI-Generated Images?

**An experimental study of AI-image detection, model architecture, and robustness to image transformations.**

A neural network can learn to distinguish a photograph from an AI-generated image. But *what is it actually detecting?* The contents of the image, subtle artefacts left by the generator, or something simpler?

I explored that question by building a **dense-only neural network** to classify real and synthetic images from the CIFAKE dataset, then deliberately changing the test images to see which changes caused its predictions to fail.

**[Read the full illustrated report](cifake_report.html) · [Explore the Jupyter notebook](cifake_report.ipynb)**

## Results at a glance

| Metric | Result |
| --- | ---: |
| **Test accuracy** | **81.63% ± 0.82 percentage points** |
| **Test ROC-AUC** | **0.9117 ± 0.0010** |
| Dataset | 120,000 images (60,000 real / 60,000 AI-generated) |
| Final model | 2 × 512-unit Dense hidden layers |
| Model parameters | 1,836,545 |
| Evaluation | 20,000-image held-out test set; 3 random seeds |

The test results are the **mean ± sample standard deviation across three training runs**, not confidence intervals. For context, Bird and Lotfi (2024) reported **92.98% accuracy** using a convolutional approach; that is a reference point, not a like-for-like architectural comparison.

## The research question

> **Do AI-image detectors learn general properties of synthetic imagery, or are they relying on generator-specific artefacts?**

The experiment tested three ideas:

1. **Model capacity:** A multilayer perceptron should outperform linear logistic regression at separating real and synthetic images.
2. **Fine-detail sensitivity:** If a detector depends on fragile generator fingerprints, transformations that destroy fine pixel detail should substantially damage performance.
3. **Colour dependence:** If texture matters more than colour, greyscale conversion should have comparatively little impact.

The last two were the most revealing — and neither was supported in the way I originally expected.

## How the experiment worked

### 1. Build a detector without convolutional layers

The [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) contains **32 × 32 RGB** images: real photographs from CIFAR-10 and synthetic images produced using Stable Diffusion 1.4.

Each image was flattened into **3,072 normalised pixel values**, then passed into a fully connected neural network. The project deliberately excludes convolutional layers so the model has to learn useful patterns without built-in spatial feature extraction.

I compared the network with chance and logistic-regression baselines, then explored model depth, width, dropout, L2 regularisation, optimiser settings and learning rates. Experiments used a stratified **80,000 / 20,000 training-validation split** from CIFAKE's original training partition. The separate **20,000-image test set** was reserved for final evaluation.

The selected final configuration was:

| Setting | Value |
| --- | --- |
| Input | 3,072 flattened, normalised pixel values |
| Hidden layers | 512 ReLU units × 2 |
| Output | 1 sigmoid unit (real vs AI-generated) |
| Optimiser | Adam, learning rate `0.0001` |
| Loss | Binary cross-entropy |
| Dropout / L2 | None |
| Training | 23 epochs on all 100,000 training images |

### 2. Test what happens when images change

After training the final models on clean images, I evaluated them on **13 transformed conditions plus the clean baseline** — without retraining. These included JPEG recompression at different quality levels, downsampling, blur, noise, horizontal flip, brightness, contrast, greyscale, and a composite transformation intended to simulate screenshot/reposting effects.

| Transformation | Effect on accuracy compared with clean images |
| --- | ---: |
| Downsampling (×4) | **−10.4 percentage points** |
| Greyscale | **−7.4 points** |
| Contrast adjustment | **−4.9 points** |
| Gaussian blur | **−4.7 points** |
| Horizontal flip | −0.85 points |
| Brightness adjustment | −0.3 points |
| JPEG recompression and additive noise | Under 1 point of loss in the reported conditions |

These are experimental results from this model and dataset, **not general robustness guarantees** for AI-image detectors.

## What I found

The surprising part was that **JPEG compression and added noise barely affected the detector**, even though I expected these changes to disrupt subtle pixel-level fingerprints. Meanwhile, **downsampling and greyscale conversion caused some of the largest drops in accuracy**.

The pattern was less about *fine detail versus global appearance* than about **removing useful information versus perturbing it**. For this network, colour and coarse spatial structure appear to matter more than I originally predicted.

That does **not** demonstrate that generator fingerprints do not exist. It suggests that **this particular dense-only model, trained on this particular dataset, was not relying primarily on the fragile fingerprints I had hypothesised**.

I also examined the **confusion matrix, ROC curve, threshold trade-offs, and the model's most confident mistakes**. Raising the decision threshold reduced the number of real photographs wrongly labelled as AI-generated, but also meant missing more synthetic images — a meaningful trade-off when considering how such a detector might be used.

## Limitations and future work

The results need to be interpreted within the experiment's constraints:

- **One generator:** The synthetic half of CIFAKE comes from Stable Diffusion 1.4. This study does not establish generalisation to newer or unseen generators.
- **Different data histories:** CIFAR-10 photographs and generated images do not share the same original production and processing pipeline. The model may be exploiting dataset-specific differences.
- **Small images, dense-only architecture:** Flattened 32 × 32 inputs may hide or discard distinctions that a higher-resolution or convolutional/frequency-domain model could use.
- **Test-time perturbations only:** The study measures how a trained detector behaves under transformations, not whether retraining or augmentation could recover performance.

A natural follow-up would be to **train across multiple generators and evaluate on an unseen one**, alongside datasets that control for compression and preprocessing differences.

## Read or reproduce the work

This repository is intended to hold both the **executed notebook** and its **HTML research report**:

```text
.
├── README.md
├── cifake_report.ipynb   # Code, experiments, saved outputs, discussion
└── cifake_report.html    # Full report for reading in a browser
```

**To run the notebook locally:**

1. Install Python with Jupyter and the main dependencies:
   ```bash
   python -m pip install jupyter tensorflow numpy pillow matplotlib
   ```
2. Download CIFAKE from [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and arrange the JPEGs like this:
   ```text
   data/
   ├── train/
   │   ├── REAL/
   │   └── FAKE/
   └── test/
       ├── REAL/
       └── FAKE/
   ```
3. Open the notebook with `jupyter notebook cifake_report.ipynb` and run its cells in order.

The notebook generates intermediate files including `cache/*.npz`, `grid_results.json`, trained `.keras` models, and prediction arrays. These are **outputs of the experiment**, not files you need to commit. Reproducing the full architecture search and repeated training runs requires substantial compute and memory; exact behaviour can vary with library versions and hardware.

## References

- **Bird, J. J. & Lotfi, A. (2024).** *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access. [DOI: 10.1109/ACCESS.2024.3356122](https://doi.org/10.1109/ACCESS.2024.3356122)
- **Frank, J. et al. (2020).** *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML. [Paper](https://proceedings.mlr.press/v119/frank20a.html)
- **Chollet, F. & Watson, M. (2026).** *Deep Learning with Python*, 3rd edition. Manning.

The full report contains additional references, experiment-level methodology, and a **code-provenance table** distinguishing original implementations from adapted patterns.

---

**Indi Caburian** · BSc Computer Science, University of London · 2026
