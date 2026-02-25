# TAP-SLF: Parameter-Efficient Adaptation of Vision Foundation Models for Multi-Task Ultrasound Image Analysis

Welcome to the TAP-SLF project, a specialized framework for fine-tuning and deploying Microsoft's Florence-2 vision foundation model for medical image analysis tasks. This repository provides tools and scripts for training, evaluation, and deployment of Florence-2 models on medical imaging datasets.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Download](#model-download)
- [Dataset Preparation](#-dataset-preparation)
- [Model Training](#-model-training)
  - [Main Training Script](#main-training-script)
  - [Ablation Studies](#ablation-studies)
- [Inference and Evaluation](#-inference-and-evaluation)
- [Docker Support](#-docker-support)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 📌 Project Overview

The TAP-SLF project is designed to leverage the power of Microsoft's Florence-2 vision foundation model for medical image analysis tasks. This framework enables:

- Fine-tuning Florence-2 on medical imaging datasets
- Multi-task learning across segmentation, classification, detection, and regression tasks
- Flexible adapter configurations with LoRA and prompt tuning
- Docker-based deployment for production and competition submissions

The project is optimized for ultrasound image analysis but can be extended to other medical imaging modalities.

---

## ✨ Key Features

- **Multi-task Learning**: Supports segmentation, classification, detection, and regression tasks in a single model
- **Adapter Configurations**: Multiple adapter variants including full LoRA, no LoRA, no prompt, and freeze-all versions
- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for efficient parameter fine-tuning
- **Prompt Tuning**: Supports task-specific soft prompts for better performance
- **Docker Support**: Complete Docker integration for consistent deployment
- **Robust Error Handling**: Clear error messages for model loading and data preparation issues
- **Comprehensive Ablation Studies**: Pre-configured ablation scripts for model analysis

---

## 📁 Project Structure

```
Florence-2-adapter/
├── ablation_study/          # Ablation study scripts and adapters
│   ├── florence_adapter_freeze_all.py    # Adapter with all parameters frozen
│   ├── florence_adapter_full_lora.py     # Adapter with LoRA on all layers
│   ├── florence_adapter_no_lora.py      # Adapter without LoRA
│   ├── florence_adapter_no_prompt.py    # Adapter without prompt tuning
│   ├── train_florence_freeze_all.py     # Training script for freeze-all 
│   ├── train_florence_full_lora.py      # Training script for full LoRA
│   ├── train_florence_no_lora.py       # Training script for no LoRA
│   └── train_florence_no_prompt.py     # Training script for no prompt
├── docker/                  # Docker-related files
│   ├── model/               # Florence-2 model directory
│   ├── Dockerfile           # Dockerfile for building container
│   ├── README.md            # Docker documentation
│   ├── QUICKSTART.md        # Docker quick start guide
│   ├── build.sh             # Docker build script
│   ├── model_inference.py   # Docker model inference script
│   ├── run_test.sh          # Docker test script
│   └── florence_adapter.py  # Docker-specific adapter implementation
├── model/                   # Florence-2 model directory
│   └── florence-2-base/     # Florence-2 base model files
├── train/                   # Training data directory
│   ├── Segmentation/        # Segmentation task data
│   ├── Classification/      # Classification task data
│   ├── Detection/           # Detection task data
│   ├── Regression/          # Regression task data
│   ├── csv_files/           # CSV index files
│   ├── train_8_csv/          # 80% training data
│   └── valdataset_2_csv/     # 20% validation data
├── config.py                # Configuration file
├── dataset.py               # Dataset loading and processing
├── florence_adapter.py      # Main Florence-2 adapter implementation
├── model_factory.py         # Multi-task model factory
├── train_florence.py        # Main training script
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/florence-2-adapter.git
cd florence-2-adapter
```

2. Install dependencies:

```bash
pip install -r docker/requirements.txt
```

### Model Download

The TAP-SLF requires the Florence-2 base model files. Follow these steps to download and prepare the model:

1. Visit the [Florence-2 base model page](https://huggingface.co/microsoft/Florence-2-base) on Hugging Face
2. Download the following files:
   - `modeling_florence2.py`
   - `pytorch_model.bin`
3. Create the model directory and place the files:

```bash
mkdir -p model/florence-2-base
# Place the downloaded files in this directory
```

The final directory structure should be:

```
model/florence-2-base/
├── modeling_florence2.py
└── pytorch_model.bin
```

---

## 📊 Dataset Preparation

The dataset should be organized in the following structure:

```
train/
├── csv_files/              # CSV index files
│   ├── Seg-*.csv          # Segmentation task CSVs
│   ├── Cls-*.csv          # Classification task CSVs
│   ├── Det-*.csv          # Detection task CSVs
│   └── Reg-*.csv          # Regression task CSVs
├── Segmentation/          # Segmentation task images and masks
│   ├── Two/               # 2-class segmentation
│   ├── Three/             # 3-class segmentation
│   ├── Four/              # 4-class segmentation
│   └── Five/              # 5-class segmentation
├── Classification/         # Classification task images
├── Detection/             # Detection task images
└── Regression/            # Regression task images
```

Each CSV file should contain the following columns:
- `image_path`: Path to the image file
- `mask_path`: Path to the mask file (for segmentation tasks)
- `task_id`: Unique task identifier
- `label`: Ground truth label (for classification tasks)
- `bbox`: Bounding box coordinates (for detection tasks)
- `points`: Keypoint coordinates (for regression tasks)

---

## 🏋️ Model Training

### Main Training Script

Use the `train_florence.py` script to train the main model:

```bash
python train_florence.py --data_root train/train_8_csv --batch_size 8 --epochs 100
```

Key parameters:
- `--data_root`: Path to the training data directory
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 1e-5)
- `--unfreeze`: Unfreeze the encoder for end-to-end fine-tuning

### Ablation Studies

The project includes several ablation study scripts to evaluate different adapter configurations:

1. **Full LoRA**: Applies LoRA to all transformer layers
   ```bash
   python ablation_study/train_florence_full_lora.py --data_root train/train_8_csv --batch_size 8 --epochs 100
   ```

2. **No LoRA**: Trains without LoRA adaptation
   ```bash
   python ablation_study/train_florence_no_lora.py --data_root train/train_8_csv --batch_size 8 --epochs 100
   ```

3. **No Prompt**: Trains without soft prompt tuning
   ```bash
   python ablation_study/train_florence_no_prompt.py --data_root train/train_8_csv --batch_size 8 --epochs 100
   ```

4. **Freeze All**: Freezes all encoder parameters, only trains task heads
   ```bash
   python ablation_study/train_florence_freeze_all.py --data_root train/train_8_csv --batch_size 8 --epochs 100
   ```

---

## 📈 Inference and Evaluation

To run inference with the trained model:

```bash
python model_inference.py --data_root /path/to/data --output_dir predictions/
```

The model will generate predictions in the following format:

- **Segmentation**: Image files in the same directory structure as the input masks
- **Classification**: JSON file (`classification_predictions.json`)
- **Detection**: JSON file (`detection_predictions.json`)
- **Regression**: JSON file (`regression_predictions.json`)

### Evaluation Metrics

The framework computes the following metrics for each task type:
- **Segmentation**: Dice coefficient (DSC), Hausdorff distance (HD95)
- **Classification**: AUC, F1 score, MCC
- **Detection**: IoU
- **Regression**: Mean Relative Error (MRE)

---

## 🐳 Docker Support

The project includes complete Docker support for consistent deployment:

1. **Build the Docker image**:
   ```bash
   cd docker
   ./build.sh
   ```

2. **Test the Docker image**:
   ```bash
   ./run_test.sh /path/to/validation /path/to/output
   ```

3. **Run the Docker container**:
   ```bash
   docker run --gpus all --rm \
     -v /path/to/data:/app/valdataset_for_participants:ro \
     -v /path/to/output:/app/output \
     florence-2-adapter:latest
   ```

The Docker container expects:
- Input data mounted at `/app/valdataset_for_participants`
- Output directory mounted at `/app/output`

---

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   - **Cause**: The Florence-2 model files are missing or in the wrong location
   - **Solution**: Follow the [Model Download](#model-download) instructions to properly download and place the model files

2. **CUDA Out of Memory**
   - **Cause**: Batch size is too large for your GPU memory
   - **Solution**: Reduce the batch size using the `--batch_size` parameter

3. **Import Errors**
   - **Cause**: Missing dependencies or incorrect Python environment
   - **Solution**: Install all required dependencies using `pip install -r docker/requirements.txt`

4. **Data Loading Errors**
   - **Cause**: Incorrect dataset structure or missing files
   - **Solution**: Verify that your dataset follows the required directory structure and all files are present

---

## 📄 License

This project is for research and educational purposes only. The Florence-2 model is subject to Microsoft's license terms.

---

## 🎉 Acknowledgments

- [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) - Vision foundation model
- [Train dataset_1](https://drive.google.com/file/d/1SivTzkK6IVLH44S_2B5xSR00j96jkJ4-/view?usp=sharing) - Training dataset_1 for the project
- [Train dataset_2](https://drive.google.com/file/d/18oU6FraMa3ybs_XmQhDNrZESRorbGqaG/view?usp=sharing) - Training dataset_2 for the project
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Albumentations](https://albumentations.ai/) - Image augmentation library

---

## 📞 Contact

For questions or issues, please open an issue in the repository or contact the maintainers.

---

**Happy Training! 🚀**