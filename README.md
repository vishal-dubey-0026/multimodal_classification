# Multimodal Classification Project

This repository contains the implementation of a **multimodal classification** system designed to process and classify data containing multiple modalities such as images and text. The model is trained on roughly 1.3 million image-text pairs and can be used for the task of multimodal content moderation. The project supports training and inference for classification tasks using state-of-the-art deep learning techniques.

---

## Features

- **Multimodal Input Handling**: Processes both image and text inputs for classification.
- **LoRA Integration**: Implements Low-Rank Adaptation (LoRA) on output projection matrices in the attention layers of both vision and text towers.
- **CLIP Architecture**: Utilizes ViT-B/32 as the vision tower and a Causal LM for the text tower.
- **Efficient Data Loading**: Supports large datasets using WebDataset format with sharded `.tar` files.
- **Mask for Valid Pairs**: Computes contrastive loss only on valid positive pairs defined by a binary mask.
- **Transformer-Based Classification Head**: Combines image and text embeddings with a classification head consisting of two transformer layers and a linear projection to logits.
- **Seven-Class Classification**: Combines MMHS150K hate classes into a single "Hate" class, resulting in seven overall classes.
- **WebDataset Preparation**: Provides tools and notebooks for preparing WebDataset `.tar` files containing images, captions, and classification labels.
- **Top-k Accuracy Evaluation**: Evaluates the model using top-k accuracy metrics.
- **Training Pipeline**: Includes training, validation, and testing loops with configurable hyperparameters.

---

## Datasets

### Supported Datasets

- **MMHS150K** ([Paper](https://arxiv.org/pdf/1910.03814)): Contains 150K image-text pairs from Twitter, divided into six classes: NotHate, Racist, Sexist, Homophobe, Religion, and OtherHate. Racist, Sexist, Homophobe, Religion, and OtherHate classes are merged into the "Hate" class.
- **Fakeedit** ([Paper](https://aclanthology.org/2020.lrec-1.755.pdf)): Contains 600K+ image-text pairs from Reddit, classified into six categories: True, Satire/Parody, Misleading Content, Imposter Content, False Connection, and Manipulated Content.
- **MSCOCO** ([Dataset](https://cocodataset.org/#download)): Contains 600K+ image-text pairs.

### Dataset Specification

Data from MMHS150K and Fakeedit is combined into the following seven classes:
1. True
2. Satire/Parody
3. Misleading Content
4. Imposter Content
5. False Connection
6. Manipulated Content
7. Hate

MSCOCO, True (Fakeedit), and NotHate (MMHS150K) samples are mapped to the "True/Neutral" class.

---

## Training Objectives

The model is trained using two loss functions with equal weight:

1. **CLIP-Style Contrastive Loss**: Used to train the vision-text towers. Contrastive loss is computed only on valid image-text pairs (True/Neutral class), excluding other classes due to potential misalignment or noise.

2. **Cross-Entropy Loss**: Used to train the classifier head.

---

## Requirements

### Install Dependencies
To install the required Python packages, run:
```bash
pip install -r requirements-training.txt
```

---

## Dataset Preparation

The project supports datasets stored in **WebDataset format**. Ensure the dataset is prepared with:
- Resized images.
- Text captions.
- Metadata including labels and additional information.

### Example Dataset Structure

```
data/
├── class_0/
│   ├── 00001.tar
│   ├── 00002.tar
├── class_1/
│   ├── 00001.tar
│   ├── 00002.tar
...
```

Each `.tar` file contains:
- Resized images (`id1.jpg`) to 256x256 resolution.
- Captions (`id1.txt`).
- Metadata (`id1.json`).

To prepare WebDataset `.tar` files, use the provided notebook `webdataset_prepare.ipynb`. Training dataset shards have a size of 10k, while evaluation/testing dataset shards have a size of 1k.

---

## Training Configuration

The following are the key training parameters used for multimodal classification:

- **Save Frequency**: Save checkpoints every 5 epochs.
- **Warmup**: 10,000 warmup steps for learning rate scheduling.
- **Batch Size**: 2048.
- **Learning Rate**: 1e-3.
- **Weight Decay**: 0.1.
- **Epochs**: 600.
- **Workers**: 8 (number of data loading workers).
- **Model**: ViT-B-32.
- **Pretrained Weights**: `laion2b_s34b_b79k`.
- **Training Samples**: 128,000 samples.
- **Resampled Dataset**: Enabled.
- **Dataset Type**: WebDataset.
- **Precision**: AMP (Automatic Mixed Precision).
- **Training Data**: Path specified in `{train_data}`.
- **Training Data Upsampling Factors**: Defined by `{train_data_sampling}`.
- **Gradient Checkpointing**: Enabled to save memory.
- **Gradient Accumulation**: Frequency of 4.
- **Cache Directory**: `../../pretrained_models_weights`.
- **LoRA**: Enabled.
- **Validation Data**: Path specified in `{val_data}`.

### Example Training Command
```bash
python -m open_clip_train.main \
    --save-frequency 5 \
    --warmup 10000 \
    --batch-size=2048 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=600 \
    --workers=8 \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --train-num-samples 128000 \
    --dataset-resampled \
    --dataset-type webdataset \
    --precision amp \
    --train-data "{train_data}" \
    --train-data-upsampling-factors "{train_data_sampling}" \
    --grad-checkpointing \
    --accum-freq 4 \
    --cache-dir ../../pretrained_models_weights \
    --enable-lora \
    --val-data "{val_data}"
```

---

## Inference

Inference capabilities are implemented to evaluate the model on unseen data. Additional documentation for the inference pipeline will be provided in subsequent updates.

---

## Results

- **Testset**: Consists of 60k image-text pairs from the validation splits of MMHS150K and Fakeedit.
- **Vision Tower**: ViT-B/32 model with 88 million parameters.
- **Text Tower**: Causal LM with 63 million parameters.
- **Classifier Head**: Contains 6.6 million parameters.
- **Baseline Model (model_baseline)**: Trains all parameters in the text and vision towers, resulting in 158 million trainable parameters.
- **LoRA Model (model_lora)**: Adds LoRA to both text and vision towers, introducing 120k parameters, with a total of 6.7 million trainable parameters.

### Accuracy Results

| Model          | Top-1 Accuracy | Top-2 Accuracy |
|----------------|----------------|----------------|
| Baseline       | 92.18%         | 95.30%         |
| LoRA Model     | 94.87%         | 98.85%         |

- **Training Details**: All experiments were conducted on Google Colab's Tesla T4 GPU (15 GB VRAM). The batch size was chosen to utilize the same amount of GPU memory for both models, ensuring a fair comparison. Training samples were distributed evenly across classes.

---

## Project Structure

```
.
├── src/open_clip_train/main.py       # Script for training the model
├── requirements-training.txt         # Python dependencies
├── eval.ipynb                        # Notebook for evaluating the model
├── train.ipynb                       # Notebook for training the model
├── webdataset_prepare.ipynb          # Notebook for preparing WebDataset tar files
└── README.md                         # Project documentation
```

---

## Acknowledgments

This codebase is based on [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main). Special thanks to the open-source community for their contributions to this project.
