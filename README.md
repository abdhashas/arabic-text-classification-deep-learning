# 🤖 Arabic Text Classification & Intent Detection for Chatbots

An advanced Natural Language Processing (NLP) pipeline designed to classify Arabic text intents for intelligent chatbot engines. This project spans the entire text processing lifecycle, from raw text normalization to deep learning classification using custom semantic embeddings.

## 📂 Project Structure & Core Files

The repository is structured with the following essential components to handle the text processing and training workflow:
- **`NLP_HOMEWORK_1_BY_[your_name].ipynb`**: The main Jupyter Notebook housing the text preprocessing, architecture implementation, model training, and evaluation pipelines.
- **`stop_word_in_arabic/`**: Custom data directory containing comprehensive lists of Arabic stop-words targeted for removal.
- **Embeddings Components (`w2v_SG_*`)**: Pre-trained Word2Vec Skip-Gram (`SG`) embedding models used to extract 300-dimensional continuous vector representations for semantic text processing.
- **`train.csv` / `val.csv` / `test.csv`**: Structured data splits containing the labeled intent text data used during optimization and final accuracy evaluations.
- **`font/`**: Supporting resources utilized for rendering correct Arabic text configurations during plot generations.

## 🚀 Key Pipeline Features

- **Custom Arabic Text Preprocessing**: Implements robust text normalization, punctuation removal, custom stop-word filtering via the `stop_word_in_arabic` assets, and tokenization.
- **Baseline Machine Learning Models**: Built statistical learning baselines using **TF-IDF** feature extraction combined with **Logistic Regression** to establish benchmark performance.
- **Semantic Word Embeddings**: Explores deep vector spaces by leveraging the embedded **Word2Vec Skip-Gram model** to capture deep contextual and semantic relations between Arabic phrases.
- **Deep Learning Architecture**: Designed and trained a **Convolutional Neural Network (Text CNN)** optimized for text sequence classification, enabling accurate intent mapping.
- **Comprehensive Evaluation**: Measures classification performance using balanced evaluation metrics, including Balanced Accuracy and confusion matrices.

## 🛠️ Tech Stack & Tooling

- **Programming Language**: Python
- **Frameworks & Libraries**: TensorFlow / Keras, Gensim (Word2Vec processing), Scikit-Learn, NLTK, Pandas, NumPy, Matplotlib
- **Algorithms**: Convolutional Neural Networks (Text CNN), Logistic Regression, TF-IDF Vectorization, Word2Vec (Skip-Gram)

## ⚙️ Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install the necessary machine learning and NLP dependencies:
   ```bash
   pip install tensorflow gensim scikit-learn nltk pandas numpy matplotlib
   ```
3. Open and execute the main Jupyter Notebook `NLP_HOMEWORK_1_BY_[your_name].ipynb` sequentially to reproduce training benchmarks and semantic evaluations.
