# IaC Violation Detection

A machine learning framework for detecting Infrastructure as Code (IaC) security violations and misconfigurations in cloud infrastructure templates.

## 🎯 Overview

This project implements a comprehensive training pipeline for detecting security violations in Infrastructure as Code (IaC) configurations, specifically focusing on AWS CloudFormation and Terraform templates. The framework uses advanced NLP techniques and transformer models to identify potential security issues, misconfigurations, and policy violations in cloud infrastructure code.

## 📊 Dataset

The project includes three cleaned datasets with anonymized AWS credentials:

- **`aws_train.csv`** (100MB+): Primary training dataset with IaC templates and violation labels
- **`aws_eval.csv`** (13MB): Evaluation dataset for model validation  
- **`aws_gold.csv`** (1.6MB): Gold standard test set for final performance assessment

> **Note**: All AWS access keys and secrets in the datasets have been replaced with placeholder values (`AKIAXXXXXXXXXXXXXXXX`) for security compliance. Large files are managed via Git LFS.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data processing, model training, and evaluation
- **Security-First**: Comprehensive data cleaning utilities to remove sensitive credentials
- **Scalable Training**: Support for various transformer models and training configurations
- **Rich Metrics**: Detailed evaluation metrics and visualization tools
- **Preprocessing Pipeline**: Advanced text preprocessing for IaC code analysis
- **Flexible Tokenization**: Customizable tokenization strategies for code analysis

## 🏗️ Project Structure

```
IaC_Violation_Detection/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── train_iac.py               # Main training script
├── clean_secrets.py           # Utility to clean sensitive data
├── .gitignore                 # Git ignore rules
├── .gitattributes             # Git LFS configuration
├── dataset/                   # Training datasets (Git LFS)
│   ├── aws_train.csv         # Primary training data
│   ├── aws_eval.csv          # Evaluation data
│   └── aws_gold.csv          # Gold standard test data
└── iac_train/                 # Core framework modules
    ├── __init__.py           # Package initialization
    ├── data.py               # Data loading and processing
    ├── models.py             # Model architectures
    ├── preprocessing.py      # Text preprocessing utilities
    ├── tokenization.py       # Custom tokenization strategies
    ├── collators.py          # Data collation for training
    ├── metrics.py            # Evaluation metrics
    ├── callbacks.py          # Training callbacks
    ├── plots.py              # Visualization utilities
    ├── history.py            # Training history tracking
    ├── state.py              # Model state management
    └── utils.py              # General utilities
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ManassehV2/IaC_Violation_Detection.git
   cd IaC_Violation_Detection
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Git LFS (if not already installed):**
   ```bash
   git lfs install
   git lfs pull  # Download large dataset files
   ```

## 🚂 Training

### Basic Usage

Run the main training script with default settings:

```bash
python train_iac.py
```

### CLI Examples

The training script supports extensive command-line configuration. Here are common usage patterns:

#### **Basic Training with Custom Datasets**
```bash
python train_iac.py \
    --train_csv dataset/aws_train.csv \
    --val_csv dataset/aws_eval.csv \
    --output_dir ./my_results
```

#### **Model Architecture Selection**
```bash
# Use GraphCodeBERT (default)
python train_iac.py --backbone microsoft/graphcodebert-base

# Use CodeBERT
python train_iac.py --backbone microsoft/codebert-base

# Use DistilBERT (faster, smaller)
python train_iac.py --backbone distilbert-base-uncased
```

#### **Training Mode Options**
```bash
# Full sliding window with attention (default)
python train_iac.py --mode full

# No sliding windows (faster training)
python train_iac.py --mode no_sliding_windows

# Sliding windows without attention mechanism
python train_iac.py --mode no_attention
```

#### **Hyperparameter Tuning**
```bash
python train_iac.py \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --epochs 15 \
    --window_size 512 \
    --stride 256
```

#### **Testing Only**
```bash
python train_iac.py \
    --task test \
    --test_csv dataset/aws_gold.csv \
    --load_checkpoint ./results/best_model
```

#### **Complete Training Pipeline**
```bash
python train_iac.py \
    --task train_and_test \
    --train_csv dataset/aws_train.csv \
    --val_csv dataset/aws_eval.csv \
    --test_csv dataset/aws_gold.csv \
    --backbone microsoft/graphcodebert-base \
    --mode full \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --epochs 10 \
    --window_size 384 \
    --stride 192 \
    --max_windows 6 \
    --output_dir ./results \
    --seed 42
```

#### **Early Stopping Configuration**
```bash
python train_iac.py \
    --early_stopping_patience 3 \
    --early_stopping_delta 0.01 \
    --early_stopping_metric eval_f1_macro

# Disable early stopping
python train_iac.py --disable_early_stopping
```

### CLI Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--backbone` | str | `microsoft/graphcodebert-base` | Pre-trained model architecture |
| `--mode` | str | `full` | Training mode: `full`, `no_sliding_windows`, `no_attention` |
| `--task` | str | `train` | Task: `train`, `test`, `train_and_test` |
| `--train_csv` | str | - | Path to training dataset CSV |
| `--val_csv` | str | - | Path to validation dataset CSV |
| `--test_csv` | str | `None` | Path to test dataset CSV |
| `--output_dir` | str | `./results` | Output directory for results |
| `--learning_rate` | float | `2e-5` | Learning rate for optimizer |
| `--batch_size` | int | `4` | Training batch size |
| `--epochs` | int | `10` | Number of training epochs |
| `--window_size` | int | `384` | Token window size for sliding windows |
| `--stride` | int | `192` | Stride for sliding windows |
| `--max_windows` | int | `6` | Maximum number of windows per sample |
| `--load_checkpoint` | str | `None` | Path to checkpoint for resuming/testing |
| `--early_stopping_patience` | int | `2` | Early stopping patience |
| `--early_stopping_delta` | float | `0.005` | Minimum improvement for early stopping |
| `--early_stopping_metric` | str | `eval_f1_micro` | Metric for early stopping |
| `--disable_early_stopping` | flag | `False` | Disable early stopping |
| `--seed` | int | `42` | Random seed for reproducibility |

The training script will automatically:
- Load and preprocess the datasets
- Initialize the specified model architecture
- Execute the training loop with proper validation
- Save model checkpoints and training metrics
- Generate evaluation reports and visualizations

## 📈 Model Architecture

The framework supports multiple transformer-based architectures optimized for code analysis:

- **BERT variants**: Fine-tuned for IaC code understanding
- **CodeBERT**: Specialized for programming language comprehension
- **Custom architectures**: Domain-specific models for security violation detection

## 🔧 Configuration

The framework provides flexible configuration options:

- **Model selection**: Choose from various pre-trained models
- **Preprocessing options**: Customize text cleaning and tokenization
- **Training parameters**: Adjust learning rates, batch sizes, and epochs
- **Evaluation metrics**: Configure validation and testing procedures

## 📊 Evaluation

The project includes comprehensive evaluation tools:

- **Classification metrics**: Precision, recall, F1-score
- **Confusion matrices**: Detailed error analysis
- **Visualization tools**: Training curves and performance plots
- **Model comparison**: Benchmark different architectures

## 🔒 Security Features

- **Credential Sanitization**: Automatic removal of sensitive AWS credentials
- **Data Cleaning**: Comprehensive utilities for dataset security compliance
- **Privacy Protection**: Anonymized examples while preserving learning value

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of academic research. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- Dataset sources and cloud security research community
- Open source transformer model implementations
- AWS security best practices documentation

## 📞 Contact

For questions about this research project, please open an issue in the repository.

---

**Note**: This project is designed for research and educational purposes. Always follow your organization's security policies when working with infrastructure code.