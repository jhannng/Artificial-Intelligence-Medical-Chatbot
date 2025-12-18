## Retrieval-Augmented Medical Chatbot

![Project Dashboard Demo Image](/imgs/dashboard_demo.png)

### Description

<hr />

<div style="text-align: justify;">

> The **Medical Chatbot** is an intelligent conversational system built with **Retrieval-Augmented Generation (RAG)** architecture to deliver reliable, document-grounded medical information.
>
> The system levverage **LangChain** as the orchestration layer, **OpenAI Large Language Models (LLMs)** for response generation, and **Chroma Vector Database** for semantic document retrieval. A curated medical PDF knolwledge base is embedded and indexed to ensure answers are grounded in verified sources.
>
> This project is implemented with **Streamlit**, providing an interactive web-based interface that allows users to query medical topics, view cited source documents, and adjuste retrieval and model settings dynamicallly, which help to minimise hallucinations while maintaining transperency and usability.

</div>

### Features

<hr />

<div style="text-align: justify;">

1. **Retrieval-Augmented Generation (RAG) Pipeline**  
   Combines **semantic search** and **Large Language Model based generation** to produce context-aware medical responses, which to ensure answers are grounded in retrieved medical documents rather than purely generative outputs, improves reliability and reduces hallucinations in sensitive medical domains.

2. **Chroma Vector Store for Semantic Retrieval**  
   Chroma is used as the vector store to persist and retrieve document embeddings efficiently, which enables fast similarity search across medical PDF documents.

3. **OpenAI Large Language Models (LLMs)**  
   Integrates **OpenAI LLMs** for high-quality natural language understanding and response generation, which support configurable model selection and optimized for coherent, concise, and medically relevant explanations.

4. **Source Transparency & Document Attribution**  
   Displays **retrieved source documents** alongside chatbot responses, enables users to verify where information orginates from while enhancing trust, explainability, and auditability of the chatbot’s outputs.

</div>

### Setup Instructions

<hr />

**<span style="color: #A1AEB1;">Backend Setup (Python Server)</span>**

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

**<span style="color: #A1AEB1;">Usage</span>**

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

### Get In Touch

<hr />

[**Email**](mailto:hangjk0612@gmail.com) | [**GitHub**](https://github.com/jhannng) | [**LinkedIn**](https://linkedin.com/in/jhannng)
