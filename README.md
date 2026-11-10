# 💬 End-to-End Sentiment Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](#)
[![Domain: NLP](https://img.shields.io/badge/Domain-NLP-blue.svg)](#)

Production-ready modular sentiment classification pipeline using Recurrent Neural Networks (LSTM/Bi-LSTM) in PyTorch.

## 📌 Overview
A clean, modular Python codebase for sentiment analysis and text polarity scoring. Features dedicated modules for custom vocabulary construction, text tokenization and padding, recurrent deep learning architectures, and inference workflows.

## 🛠️ Module Architecture
- `vocab_builder.py`: Vocabulary construction, token-to-index mapping, OOV (out-of-vocabulary) handling, and frequency pruning.
- `data_tranform.py`: Text cleaning, regex normalization, sequence padding, and DataLoader batching.
- `model.py`: Embedding layer, Bi-directional LSTM with dropout regularization, and fully connected classification head.
- `main.py`: Modular training loop with Adam optimizer, binary cross-entropy loss, and validation checkpointing.
- `Sentiment Prediction.ipynb`: Interactive inference notebook for real-time sentence polarity testing.

## 🚀 Quickstart
```bash
git clone https://github.com/FirdowsRahaman/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install torch numpy pandas scikit-learn
python main.py
```

## 📄 License
This project is licensed under the MIT License.
