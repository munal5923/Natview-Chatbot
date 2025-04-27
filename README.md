# Natview Chatbot

A Retrieval-Augmented Generation (RAG)-based chatbot built to assist users by answering questions about Natview and guiding applicants through the Data Science Fellowship Program (DSFP) application process.

---

## Key Features

- **RAG Pipeline**: Combines local knowledge retrieval with LLM reasoning.
- **DeepSeek LLaMA**: Lightweight, fine-tuned LLM for efficient response generation.
- **Production Ready**: Modular, scalable, and optimized for real-world usage.

---

## Tech Stack

- **Python 3.10**
- **LangChain**
- **DeepSeek LLaMA (via Groq API)**
- **Pinecone**

---

## Setup

```bash
git clone https://github.com/munal5923/Natview-Chatbot.git
cd Natview-Chatbot
pip install -r requirements.txt
python app.py
