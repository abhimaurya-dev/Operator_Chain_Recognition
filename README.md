
# Operator_Chain_Recognition

**Efficient Operator Chain Recognition using ViT and LoRA**

## Overview

This project focuses on recognizing image manipulation operations and their combinations using Deep Learning. It explores **Vision Transformers (ViT)** combined with **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning.

The objective is to classify images based on the image processing operations applied to them and identify different operator combinations.

Instead of fine-tuning the entire Vision Transformer, this approach uses LoRA to adapt selected attention layers of a pretrained model, significantly reducing the number of trainable parameters while leveraging pretrained visual representations.

## Model Architecture

The project uses a pretrained Vision Transformer as the backbone, adapted for multi-class image classification.

- **Base Model:** `google/vit-base-patch16-224-in21k`
- **Architecture:** Vision Transformer (ViT-Base)
- **Fine-tuning Strategy:** Low-Rank Adaptation (LoRA)
- **Framework:** PyTorch
- **Libraries:** Hugging Face Transformers, PEFT

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| LoRA Alpha | 8 |
| LoRA Dropout | 0.1 |
| Target Modules | Key, Value |
| Modules to Save | Classifier |

LoRA introduces trainable low-rank matrices into selected attention projections, allowing efficient adaptation of the pretrained model.

## Dataset

The dataset is organized into training, validation, and testing subsets.

```text
imd_temp/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each class represents a different image manipulation category or operator combination.

The dataset is loaded using PyTorch's `ImageFolder`, which automatically assigns labels based on directory names.

### Preprocessing

- Resize images to the pretrained ViT input resolution (224 × 224).
- Convert images into PyTorch tensors.
- Normalize images using the pretrained ViT image processor's mean and standard deviation.

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 2 |
| Maximum Epochs | 20 |
| Early Stopping Patience | 3 |
| Mixed Precision | Enabled |
| Hardware Acceleration | CUDA (if available) |

The training pipeline includes:

- Dataset verification.
- Pretrained ViT initialization.
- LoRA adaptation.
- Mixed precision training.
- Gradient accumulation.
- Validation after each epoch.
- Early stopping based on validation loss.
- Saving the best-performing checkpoint.

## Evaluation

The trained model is evaluated on a separate test dataset.

The following metrics are used:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score

The best model checkpoint is restored based on validation loss before final testing.

## Project Structure

```text
Operator_Chain_Recognition/
│
├── imd_temp/
│   ├── train/
│   ├── val/
│   └── test/
│
├── vit_lora_r8.ipynb
├── vit_lora_r8.pth
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/Operator_Chain_Recognition.git
cd Operator_Chain_Recognition
```

Install the required dependencies:

```bash
pip install torch torchvision transformers peft torchmetrics tqdm
```

## Usage

1. Prepare the dataset using the directory structure described above.
2. Open `vit_lora_r8.ipynb`.
3. Update dataset paths if required.
4. Execute the notebook to train the model.
5. The best checkpoint will be saved as `vit_lora_r8.pth`.
6. Evaluate the trained model on the test dataset.

## Technologies

- Python
- PyTorch
- Vision Transformers (ViT)
- Low-Rank Adaptation (LoRA)
- Hugging Face Transformers
- Hugging Face PEFT
- TorchMetrics
- CUDA

## License

This project is intended for research and educational purposes.
