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

Run the main training script:

```bash
python train_iac.py
```

The training script supports various configuration options and will automatically:
- Load and preprocess the datasets
- Initialize the specified model architecture
- Execute the training loop with proper validation
- Save model checkpoints and training metrics
- Generate evaluation reports

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