# Medical Chatbot

## Description

> The **Medical Chatbot** is an intelligent conversational system built with **Retrieval-Augmented Generation (RAG)** pipeline to provide users with reliable, evidence-based medical information. It integrates a **FAISS Vector Database** with **HuggingFace Embeddings** and **Groq-hosted LLMs** to retrieve and generate responses grounded in a curated medical PDF knowledge base. This approach ensures that the chatbot delivers accurate, context-aware answers while minimizing hallucinations by relying on verified document sources.

## Features

1. **FAISS Vector Store for Fast Semantic Retrieval**  
   Enables high-speed and accurate retrieval of relevant medical information based on semantic similarity.
   Optimized for efficient search across large document datasets.

2. **Groq or HuggingFace Backends**  
   Integrates with **Groq-hosted** or **HuggingFace** large language models for fast, contextually relevant responses.

3. **Modular Prompt Template Injection**  
   Supports dynamic prompt customization to adapt responses for different medical query contexts.
   Improves flexibility and maintainability of the **Retrieval-Augmented Generation (RAG)** pipeline.

## Setup Instructions

### Backend Setup (Python Server)

```bash
# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment for Windows
venv/Scripts/activate

# Activate Virtual Environment for MacOS/Linux
. venv/bin/activate

# Install Required Libraries
pip install -r requirements.txt

# Deactivate Virtual Environment
exit
```

### Usage

```bash
# Activate Virtual Environment for Windows
venv/Scripts/activate

# Activate Virtual Environment for MacOS/Linux
. venv/bin/activate

# Run Semantic Book Recommender Dashboard
streamlit run dashboard.py

# Deactivate Virtual Environment
exit
```

## Get In Touch

[**Email**](mailto:hangjk0612@gmail.com) | [**GitHub**](https://github.com/jhannng) | [**LinkedIn**](https://linkedin.com/in/jhannng)
