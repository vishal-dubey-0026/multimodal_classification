# Multimodal-Classification
# Multimodal Classification Project

This repository contains the implementation of a **multimodal classification** system designed to process and classify data containing multiple modalities such as images and text. The project supports training and inference for classification tasks using deep learning techniques.

---

## Features
- **Multimodal Input Handling**: Processes both image and text inputs for classification.
- **LoRA Integration**: Added LoRA (Low-Rank Adaptation) on output projection matrices in the attention layers of both the vision and text towers.
- **Architecture**: ViT-B/32 vision tower and BERT style text tower.
- **Efficient Data Loading**: Handles large datasets with WebDataset support and sharded `.tar` files.
- **Mask for Valid Pairs**: Contrastive loss is computed only on valid positive pairs, defined by a binary mask.
- **Transformer-Based Classifier**: Combines image and text embeddings for final classification.
- **Seven-Class Classification**: Merges MMHS150K hate classes into a single "Hate" class, resulting in seven overall classes.
- **WebDataset Preparation**: Includes tools and notebooks for preparing WebDataset `.tar` files containing images, captions, and classification labels.
- **Top-k Accuracy Evaluation**: Implements top-k accuracy metrics for model evaluation.
- **Training Pipeline**: Implements training, validation, and testing loops with configurable hyperparameters.

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
│   ├── shard_00001.tar
│   ├── shard_00002.tar
├── class_1/
│   ├── shard_00001.tar
│   ├── shard_00002.tar
...
```
Each `.tar` file contains:
- Resized images (`image_<idx>.jpg`)
- Captions (`caption_<idx>.txt`)
- Metadata (`metadata_<idx>.json`)

To prepare WebDataset `.tar` files, use the provided notebook `webdataset_prepare.ipynb`.

---

## Training Configuration

The following are the key training parameters used for the multimodal classification task:

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

### Example Command
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





## Results



Results are saved in the output directory specified in the configuration file.

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

---

## Acknowledgments
This codebase is based on [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main)
