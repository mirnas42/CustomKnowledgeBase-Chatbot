
# Custom Knowledge Base Chatbot Using LangChain

## Project Overview

This project presents a **Custom Knowledge Base Chatbot** designed to provide intelligent responses to user queries based on domain-specific documents. Built using **LangChain**, **Pinecone**, and **LLMs** like **Mistral 7B**, this solution ingests organizational documents (e.g., HR policies, job postings, internal directories) and enables semantic search and contextual responses through a user-friendly interface.

 Features

- 1. PDF Document Ingestion and Preprocessing
- 2.Embedding Generation using Hugging Face Transformers
- 3.Semantic Search via Pinecone Vector Database
- 4.Retrieval-based Question Answering with LangChain
  -5. Interactive Chat Interface (Streamlit / HTML + JS + Flask)
- 6. Modular, Scalable Architecture for Future Enhancements

---

## Requirements

| Component                | Technology Used                       |
|--------------------------|----------------------------------------|
| Programming Language     | Python                                |
| LLM                      | Mistral 7B via Hugging Face Inference |
| Embeddings               | HuggingFaceEmbeddings                 |
| Vector Store             | Pinecone                              |
| Framework                | LangChain                             |
| Document Loader          | PyPDFDirectoryLoader, PDFLoader       |
| Frontend                 | Streamlit                             |  |                                |
| OS Platform              | Windows                               |

---

## Project Modules

1. **Document Ingestion**
   - Load PDF files using `PyPDFDirectoryLoader`.
   - Preprocess and clean the content.

2. **Embedding Generation**
   - Convert text chunks into embeddings using Hugging Face models.

3. **Vector Database Connection**
   - Store and manage embeddings in Pinecone for fast similarity search.

4. **LLM Integration**
   - Connect to Mistral 7B using Hugging Face Inference API.

5. **RetrievalQA Orchestration**
   - Use LangChain’s `RetrievalQA` to fetch context and answer queries.

6. **Interactive Chatbot Interface**
   - Streamlit UI for interaction (or custom HTML + Flask interface).

---

