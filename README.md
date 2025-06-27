NLP Assignments Repository
Welcome to the NLP Assignments Repository! This repository contains a collection of hands-on projects from Chapters 1 to 5 of a natural language processing (NLP) course, showcasing a progression from foundational techniques to advanced multimodal applications and ethical considerations. Developed as part of an academic exercise, these assignments demonstrate practical implementations using Python, PyTorch, TensorFlow, and various NLP libraries, covering preprocessing, embeddings, Transformer-based translation, retrieval-augmented generation (RAG), multi-agent systems, prompt engineering, ethics, and multimodal learning with CLIP and BLIP.

Overview
This repository encapsulates the following assignments:

Chapter 1: Foundations of NLP

Assignment 1.1: NLP Preprocessing Basics - Implements tokenization, lemmatization, stemming, POS tagging, and NER with a Streamlit interface and Flask API.
Assignment 1.2: Word Embeddings and Visualization - Utilizes GloVe embeddings to compute nearest neighbors and visualize them with PCA/t-SNE, featuring a Streamlit app and API.
Assignment 1.3: Seq2Seq Summarization with LSTM - Develops a sequence-to-sequence model for news summarization, trained with a Jupyter notebook and deployed via Streamlit.


Chapter 2: Advanced Sequence Modeling

Assignment 2.1: Custom Transformer for Machine Translation - Builds a Transformer from scratch for English-to-French translation using the OPUS Books dataset, with a Jupyter notebook, evaluation script, and Streamlit demo.


Chapter 3: Advanced Techniques and Optimization

Assignment 3.1: Retrieval-Augmented Generation (RAG) Pipeline - Creates a RAG system using updated Jupyter notebook implementation.
Assignment 3.2: Multi-Agent System for Summarization and QA - Implements a multi-agent system with updated Jupyter notebook.
Assignment 3.3: BERT Fine-Tuning with and without LoRA - Fine-tunes BERT for classification with updated Jupyter notebook.


Chapter 4: Prompt Engineering and Ethics

Assignment 4.1: Prompt Design and Comparison for QA - Compares Direct, Few-Shot, and Chain-of-Thought prompts using Flan-T5 with a Jupyter notebook and PDF report.
Assignment 4.2: Prompt Tuning for Sentiment Analysis - Tests Direct, Contextual, and Pattern-based prompts with Flan-T5 using a Jupyter notebook and PDF report.
Assignment 4.3: Ethics in LLM Applications - Analyzes bias, fairness, and privacy challenges in LLMs, presented in a PDF essay.


Chapter 5: Advanced Multimodal Applications

Assignment 5.1: Comparison of CLIP and BLIP Multimodal LLMs - Provides a detailed report comparing CLIP and BLIP architectures in a Word document.
Assignment 5.2: BLIP Multimodal Application - Implements a Python script for image captioning and visual question answering (VQA) using the BLIP model, with a screenshot of output.



These projects collectively illustrate the evolution of NLP, from text-only foundations to multimodal systems, aligning with trends like Transformers, RAG, and responsible AI.

Repository Structure:

Assignments/
│
├───AICL434_LLM_REPORT_ASSIGNMENTS.docx
│
├───Assignment_1/
│   ├───assignment_1.1/
│   │   ├───api.py
│   │   ├───nlp_preprocessing.py
│   │   ├───streamlit_app.py
│   │   └───__pycache__/
│   │           nlp_preprocessing.cpython-310.pyc
│   │
│   ├───assignment_1.2/
│   │   ├───api_embeddings.py
│   │   ├───embedding_preprocessing.py
│   │   ├───streamlit_embeddings_app.py
│   │   └───__pycache__/
│   │           embedding_preprocessing.cpython-310.pyc
│   │
│   ├───assignment_1.3/
│   │       summarizer_app.py
│   │       training_loss.png
│   │       train_model.ipynb
│   │       x_tokenizer.pkl
│   │       y_tokenizer.pkl
│   │
│   └───screenshots/
│           assignment_1.1.png
│           assignment_1.2.png
│           assignment_1.3.png
│
├───Assignment_2/
│       app.py
│       attn_weights.npy
│       eval.py
│       eval_metrics.json
│       loss_curve.png
│       scratch_transformer.ipynb
│       scratch_transformer.py
│       screenshot.png
│       train_losses.npy
│       val_losses.npy
│
├───Assignment_3/
│       Assignment_3.1_updated.ipynb
│       Assignment_3.2_updated.ipynb
│       Assignment_3.3_updated.ipynb
│
├───Assignment_4/
│       Assignment_4.1_prompt_comparison_flan_t5.ipynb
│       Assignment_4.2_prompt_tuning_experiment.ipynb
│       assignment_4.3_essay.pdf
│       report_4.1.pdf
│       report_4.2.pdf
│
├───Assignment_5/
│       Assignment_5.1_CLIP_vs_BLIP_Report.docx
│       Assignment_5.2.py
│       Screenshot_Output_5.2.png
│
├───requirements.txt
└───README.md

Installation

Clone the repository or copy the Assignments folder to your local machine:
git clone https://github.com/your-username/nlp-assignments.git
cd nlp-assignments/Assignments

Alternatively, use the existing Assignments folder structure.

Install dependencies:
pip install -r requirements.txt

The requirements.txt file should include:

Python 3.10
PyTorch
TensorFlow 2.17.0
spaCy 3.7.2
NLTK 3.8.1
Streamlit 1.39.0
Flask 3.0.3
gensim 4.3.3
scikit-learn 1.5.2
transformers 4.45.1
sentence-transformers
chromadb
pandas 2.2.3
numpy
matplotlib
tqdm
pillow
requests
jupyter


Download models and datasets:

Run python -m spacy download en_core_web_sm for spaCy.
Ensure GloVe embeddings (‘glove-wiki-gigaword-50’) are available or downloaded via gensim.
Place datasets (e.g., news_summary_more.csv, opus_books/) in the data/ directory if not already included.
BLIP models will be downloaded automatically by Assignment_5/Assignment_5.2.py from Hugging Face.



Usage

Chapter 1:

Assignment 1.1: Run Streamlit app: streamlit run Assignment_1/assignment_1.1/streamlit_app.py. Access Flask API: python Assignment_1/assignment_1.1/api.py.
Assignment 1.2: Run Streamlit app: streamlit run Assignment_1/assignment_1.2/streamlit_embeddings_app.py. Access Flask API: python Assignment_1/assignment_1.2/api_embeddings.py.
Assignment 1.3: Train model: Open Assignment_1/assignment_1.3/train_model.ipynb in Jupyter. Run app: streamlit run Assignment_1/assignment_1.3/summarizer_app.py.


Chapter 2:

Train Transformer: Run python Assignment_2/scratch_transformer.py.
Evaluate: python Assignment_2/eval.py.
Demo: streamlit run Assignment_2/app.py.


Chapter 3:

Run RAG: Open Assignment_3/Assignment_3.1_updated.ipynb in Jupyter.
Multi-agent: Open Assignment_3/Assignment_3.2_updated.ipynb in Jupyter.
Fine-tune BERT: Open Assignment_3/Assignment_3.3_updated.ipynb in Jupyter.


Chapter 4:

Test prompts (4.1): Open Assignment_4/Assignment_4.1_prompt_comparison_flan_t5.ipynb in Jupyter. View report: Open Assignment_4/report_4.1.pdf.
Test prompts (4.2): Open Assignment_4/Assignment_4.2_prompt_tuning_experiment.ipynb in Jupyter. View report: Open Assignment_4/report_4.2.pdf.
Analyze ethics (4.3): Open Assignment_4/assignment_4.3_essay.pdf in a PDF reader.


Chapter 5:

View report: Open Assignment_5/Assignment_5.1_CLIP_vs_BLIP_Report.docx in a word processor.
Run BLIP app: python Assignment_5/Assignment_5.2.py, then follow the interactive prompts to caption images or answer questions.



Features:

Comprehensive NLP pipeline from preprocessing to multimodal applications.
Interactive Streamlit interfaces, RESTful APIs, and Jupyter notebooks.
Detailed reports comparing CLIP and BLIP, with visual outputs.


Performance:

Assignment 1.3: LSTM training at 120 samples/second on GPU.
Assignment 2.1: Transformer inference at 1.5s latency.
Assignment 3.3: LoRA reduced training time by 40%.
Assignment 5.2: BLIP inference varies by image size, typically 2-5s on CPU.


Limitations:

CPU bottlenecks in Assignments 1.1 and 1.2.
Limited dataset size in Assignment 2.1 (127,085 pairs).
Noisy retrieval in Assignment 3.1.
Assignment 5.2 requires internet for initial model download and may lag on large images.


Future Work:

Scale preprocessing for multilingual support (1.1).
Integrate pre-trained Transformers (2.1).
Enhance RAG with dense retrieval (3.1).
Explore zero-shot learning (4.1, 4.2).
Develop automated bias detection (4.3).
Optimize BLIP for real-time inference (5.2).
