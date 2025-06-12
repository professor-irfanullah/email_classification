# 📧 Email Spam/Ham Classifier

A machine learning model trained to classify emails as **spam** or **ham** (legitimate) using `scikit-learn`'s Naive Bayes. Achieves **97% accuracy** on test data.

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/scikit--learn-1.2.2-orange)
![GitHub](https://img.shields.io/badge/License-MIT-green)

---

## 📦 Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## ✨ Features

- **Pre-trained model** (`spam_model.pkl`) with 97% test accuracy.
- **CountVectorizer** integrated for text preprocessing.
- **Metadata tracking**: Version, accuracy, and training date.
- Easy-to-use prediction script.

---

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/spam-classifier.git
   cd spam-classifier
   ```

```python
pip install pandas scikit-learn nltk
```

## 📊 Model Performance

### Classification Metrics

| Metric        | Ham  | Spam |
| ------------- | ---- | ---- |
| **Precision** | 0.98 | 0.95 |
| **Recall**    | 0.98 | 0.94 |
| **F1-Score**  | 0.98 | 0.95 |

### Accuracy Scores

- **Overall Test Accuracy**: `97.06%`
- **Cross-Validation Accuracy**: `97.52%`
