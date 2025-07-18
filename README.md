# Natural Language Processing in Python

This repository is a hands-on learning resource for mastering Natural Language Processing (NLP) using Python. It combines theory-rich Jupyter Notebooks, practical exercises, and real-world applications, covering everything from text preprocessing to deep learning with transformers.

## 🙌 Credits

This project is based on the [Pierian Data NLP course](https://www.pieriantraining.com/), extended with personal insights, fixes, and new experiments.

---

## 📁 Repository Structure

```
natural-language-processing-in-python/
├── 00-Intro-to-NLP/
├── 01-Working-with-Text-Data/
├── 02-Text-Classification/
├── 03-Topic-Modeling/
├── 04-Word-Embeddings/
├── 05-Deep-Learning/
├── 06-Transformers/
├── Final-Project/
├── resources/
├── README.md
```

---

## 📘 Learning Modules

### 00. **Introduction to NLP**

* What is NLP?
* History and applications
* Overview of NLP pipeline

### 01. **Working with Text Data**

* Text cleaning and normalization
* Tokenization, stemming, and lemmatization
* Stopwords and n-grams

### 02. **Text Classification**

* Bag-of-Words and TF-IDF
* Logistic regression, Naive Bayes, and SVM
* Evaluation metrics

### 03. **Topic Modeling**

* Latent Dirichlet Allocation (LDA)
* Non-negative Matrix Factorization (NMF)
* Visualizing topics with `pyLDAvis`

### 04. **Word Embeddings**

* Word2Vec (CBOW and Skip-gram)
* GloVe and FastText
* Visualizing embeddings with t-SNE

### 05. **Deep Learning for NLP**

* Using RNNs and LSTMs with Keras
* Sequence padding and embeddings
* Sentiment analysis with deep models

### 06. **Transformers and BERT**

* Introduction to the Transformer architecture
* Pretrained models via `transformers` (HuggingFace)
* Fine-tuning BERT for text classification

---

## 🎓 Final Project

A complete NLP pipeline project demonstrating:

* End-to-end preprocessing
* Feature extraction
* Classification using traditional ML and BERT
* Model evaluation and insights

---

## 🛠️ Setup Instructions

### Recommended: Conda

```bash
conda create -n nlp python=3.8
conda activate nlp
pip install -r requirements.txt
```

### Key Libraries

* `nltk`, `spaCy`, `scikit-learn`
* `gensim`, `wordcloud`, `pyLDAvis`
* `keras`, `tensorflow`, `transformers`
* `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## 📂 Resources

All datasets and helper files used in the notebooks are available in the `resources/` folder or loaded automatically from open sources (e.g., NLTK corpora or HuggingFace models).
