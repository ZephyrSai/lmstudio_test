# LMStudio RAG Chatbot with LlamaIndex

A comprehensive AI chatbot that combines the power of locally-hosted LLMs via LMStudio with Retrieval-Augmented Generation (RAG) capabilities using LlamaIndex. This chatbot maintains persistent conversation context per user and can answer questions about your documents.

## 🚀 Features

### 💬 **Advanced Chat Capabilities**
- **Persistent Context**: Conversations continue where you left off
- **Multi-User Support**: Each user has isolated conversation history
- **RAG Integration**: Answer questions based on your documents
- **Real-time Document Search**: Searches your knowledge base for relevant context

### 📚 **Document Management**
- **Multiple Format Support**: PDF, TXT, DOCX, MD, and more
- **Automatic Ingestion**: Process documents into vector embeddings
- **User-Specific Knowledge Base**: Each user has their own document collection
- **Persistent Storage**: Documents remain available across sessions

### 🔧 **Technical Features**
- **Local LLM Integration**: Works with any model hosted on LMStudio
- **Vector Database**: ChromaDB for efficient document retrieval
- **Embedding Model**: HuggingFace sentence transformers (runs locally)
- **SQLite Database**: Stores conversations and document metadata
- **Error Handling**: Robust error handling and recovery

## 📋 Requirements

### Software Requirements
- Python 3.8+
- LMStudio ([Download here](https://lmstudio.ai))
- At least 8GB RAM (16GB recommended for larger models)

### Python Dependencies
```bash
pip install openai llama-index llama-index-embeddings-huggingface llama-index-vector-stores-chroma chromadb sentence-transformers
```
