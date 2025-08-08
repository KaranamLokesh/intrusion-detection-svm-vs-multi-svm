# Intrusion Detection System: SVM vs Multi-SVM Approaches

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/KaranamLokesh/intrusion-detection-svm-vs-multi-svm)

## 📋 Overview

This repository contains a comprehensive implementation of Support Vector Machine (SVM) based intrusion detection systems using the NSL-KDD dataset. The project compares various SVM approaches including One-vs-One, ensemble methods, and specialized techniques for handling imbalanced classes in network security.

## 🎯 Key Features

- **Multiple SVM Approaches**: One-vs-One, Pairwise Meta SVM, Ensemble Methods
- **Imbalanced Class Handling**: SMOTE, balanced class weights, focused minority detection
- **Advanced Techniques**: Cost-sensitive learning, hybrid ensembles, feature selection
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Visualization**: ROC curves, performance comparisons, confusion matrix heatmaps

## 🏗️ Architecture

### SVM Methods Implemented

1. **One-vs-One SVM**: Standard multiclass classification using pairwise comparisons
2. **Pairwise Meta SVM**: Ensemble approach using pairwise classifiers as meta-features
3. **Balanced Meta SVM**: Enhanced version with balanced class weights
4. **Enhanced Meta SVM**: Advanced version with adaptive parameters for minority classes
5. **Focused Minority SVM**: Specialized approach for detecting rare attack types (R2L, U2R)
6. **Advanced Minority SVM**: Multi-kernel ensemble specifically for minority classes
7. **Cost-Sensitive SVM**: Adaptive class weights based on class distribution
8. **Hybrid Ensemble SVM**: Combination of multiple approaches

## 📊 Dataset

The project uses the **NSL-KDD dataset**, a refined version of the KDD Cup 1999 dataset containing network intrusion detection data.

### Attack Categories
- **Normal (0)**: Legitimate network traffic
- **DoS (1)**: Denial of Service attacks
- **Probe (2)**: Surveillance and probing attacks
- **R2L (3)**: Remote to Local attacks
- **U2R (4)**: User to Root attacks

### Features
- 41 features including basic, content, and traffic features
- Categorical features encoded using LabelEncoder
- Numerical features standardized using StandardScaler

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or using Poetry (recommended)
poetry install
```

### Required Dependencies

```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
```

### Basic Usage

```python
# Load and preprocess data
from nsl_kdd_svm_combined import load_nsl_kdd, preprocess_nsl_kdd

train_set, test_set = load_nsl_kdd()
X_train, X_test, y_train, y_test = preprocess_nsl_kdd(train_set, test_set)

# Run One-vs-One SVM
from nsl_kdd_svm_combined import run_onevsone_svm
results = run_onevsone_svm(X_train, y_train, X_test, y_test)

# Run Advanced Minority SVM for better detection of rare attacks
from nsl_kdd_svm_combined import run_advanced_minority_svm
results = run_advanced_minority_svm(X_train, y_train, X_test, y_test)
```

## 📈 Performance Results

### Overall Performance Comparison

| Method | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------|----------|-----------|--------|----------|---------------|
| One-vs-One SVM | 0.749 | 0.785 | 0.749 | 0.701 | ~2.5s |
| Pairwise Meta SVM | 0.751 | 0.801 | 0.751 | 0.704 | ~3.2s |
| Balanced Meta SVM | 0.754 | 0.819 | 0.754 | 0.726 | ~3.5s |
| Enhanced Meta SVM | 0.768 | 0.787 | 0.768 | 0.730 | ~4.1s |
| Advanced Minority SVM | 0.772 | 0.791 | 0.772 | 0.735 | ~5.2s |

### Minority Class Detection (R2L & U2R)

| Method | R2L Recall | U2R Recall | Combined F1 |
|--------|------------|------------|-------------|
| One-vs-One SVM | 0.01 | 0.09 | 0.01 |
| Enhanced Meta SVM | 0.07 | 0.10 | 0.11 |
| Advanced Minority SVM | 0.15 | 0.25 | 0.18 |
| Cost-Sensitive SVM | 0.12 | 0.18 | 0.14 |

## 🔧 Advanced Usage

### Feature Selection

```python
from nsl_kdd_svm_combined import perform_feature_selection

X_train_selected, X_test_selected, selected_features, selector = perform_feature_selection(
    X_train, y_train, X_test, y_test
)
```

### Comprehensive Evaluation

```python
from nsl_kdd_svm_combined import create_comprehensive_report

# Generate all visualizations and reports
create_comprehensive_report(X_train, y_train, X_test, y_test, results_dict)
```

### Custom Class Weights

```python
# Define custom class weights for minority classes
class_weights = {
    0: 1.0,  # Normal
    1: 1.0,  # DoS
    2: 1.0,  # Probe
    3: 25.0, # R2L (high weight)
    4: 25.0  # U2R (high weight)
}
```

## 📁 Project Structure

```
intrusion-detection-svm-vs-multi-svm/
├── nsl_kdd_svm_combined.py      # Main implementation
├── ids_svm.py                    # Alternative SVM implementation
├── train.py                      # Training script
├── train.ipynb                   # Jupyter notebook
├── generate_results_report.py    # Report generation
├── plot_paper_comparison.py      # Visualization utilities
├── extract_figures_from_pdfs.py  # Figure extraction
├── KDD/                          # Dataset directory
│   ├── KDDTrain+_2.csv
│   └── KDDTest+_2.csv
├── paper_images/                 # Generated visualizations
├── extracted_figures/            # Extracted figures
├── pyproject.toml               # Poetry configuration
├── poetry.lock                  # Locked dependencies
└── README.md                    # This file
```

## 🎨 Visualization Features

The project includes comprehensive visualization capabilities:

- **Confusion Matrix Heatmaps**: Publication-quality confusion matrices
- **Performance Comparisons**: Bar charts comparing different methods
- **ROC Curves**: Receiver Operating Characteristic curves for each class
- **Precision-Recall Curves**: Detailed performance analysis
- **Class Distribution**: Training and test set distributions
- **Detection vs False Alarm**: Security-focused metrics
- **Training Time Comparison**: Computational efficiency analysis

## 🔬 Research Contributions

### Novel Approaches

1. **Pairwise Meta SVM**: Uses pairwise classifier outputs as meta-features
2. **Enhanced Minority Detection**: Specialized techniques for R2L and U2R attacks
3. **Cost-Sensitive Learning**: Adaptive class weights based on class distribution
4. **Hybrid Ensemble**: Combination of multiple SVM approaches

### Key Findings

- **Ensemble methods** show improved performance over single SVM
- **Minority class detection** significantly improved with specialized approaches
- **Cost-sensitive learning** provides better balance between classes
- **Feature selection** reduces computational complexity with minimal performance loss

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Lokesh Karanam**
- GitHub: [@KaranamLokesh](https://github.com/KaranamLokesh)
- Research Focus: Machine Learning, Network Security, Intrusion Detection

## 🙏 Acknowledgments

- NSL-KDD dataset creators
- Scikit-learn development team
- Open-source community contributors

## 📚 References

1. Tavallaee, M., et al. "A detailed analysis of the KDD CUP 99 data set." IEEE Symposium on Computational Intelligence for Security and Defense Applications, 2009.
2. Cortes, C., & Vapnik, V. "Support-vector networks." Machine learning, 1995.
3. Chawla, N. V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research, 2002.

---

⭐ **Star this repository if you find it useful!**
