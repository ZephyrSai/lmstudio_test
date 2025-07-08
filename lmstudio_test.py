#!/usr/bin/env python3
"""
LMStudio RAG Chatbot with LlamaIndex Integration

A comprehensive chatbot that:
1. Interfaces with LMStudio's local server
2. Maintains persistent conversation context per user
3. Uses LlamaIndex for RAG (Retrieval-Augmented Generation)
4. Supports document ingestion and vector storage
5. Provides context-aware responses using user documents

Requirements:
- pip install openai llama-index llama-index-embeddings-huggingface llama-index-vector-stores-chroma
- LMStudio running locally with server enabled on port 1234
- A model loaded in LMStudio

Usage:
    python lmstudio_rag_chatbot.py
"""

import sqlite3
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
import sys

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
import chromadb

class LMStudioLLM(CustomLLM):
    """Custom LLM wrapper for LMStudio"""
    
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.context_window = 4096
        self.num_output = 512
        
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return CompletionResponse(text=response.choices[0].message.content)
    
    @llm_completion_callback()
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        openai_messages = []
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=response.choices[0].message.content
            )
        )
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Not implementing streaming for simplicity
        response = self.complete(prompt, **kwargs)
        yield response
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        # Not implementing streaming for simplicity
        response = self.chat(messages, **kwargs)
        yield response

class LMStudioRAGChatbot:
    def __init__(self, 
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio",
                 db_path: str = "chatbot_conversations.db",
                 documents_dir: str = "documents",
                 vector_store_dir: str = "vector_store",
                 chroma_db_dir: str = "chroma_db"):
        """
        Initialize the LMStudio RAG Chatbot
        
        Args:
            base_url: LMStudio server URL
            api_key: API key for LMStudio
            db_path: Path to SQLite database file
            documents_dir: Directory containing documents for RAG
            vector_store_dir: Directory for vector store persistence
            chroma_db_dir: Directory for ChromaDB persistence
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.db_path = db_path
        self.documents_dir = Path(documents_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)
        self.chroma_db_dir.mkdir(exist_ok=True)
        
        self.current_user = None
        self.conversation_history = []
        self.rag_enabled = False
        self.query_engine = None
        
        # Initialize database
        self._init_database()
        
        # Get available models
        self.available_models = self._get_available_models()
        self.current_model = None
        self.llm = None
        
        # Initialize LlamaIndex settings
        self._init_llamaindex()
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message_role TEXT NOT NULL,
                message_content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                rag_context TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create document tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                document_name TEXT NOT NULL,
                document_path TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _init_llamaindex(self):
        """Initialize LlamaIndex with embedding model"""
        # Use HuggingFace embedding model (runs locally)
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Configure global settings
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
    def _get_available_models(self) -> List[str]:
        """Get list of available models from LMStudio"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Warning: Could not fetch models from LMStudio: {e}")
            print("Make sure LMStudio is running with server enabled on port 1234")
            return []
    
    def _get_or_create_user(self, username: str) -> int:
        """Get existing user or create new one"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try to get existing user
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
            # Update last active timestamp
            cursor.execute("UPDATE users SET last_active = ? WHERE id = ?", 
                         (datetime.now(), user_id))
        else:
            # Create new user
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid
            print(f"Created new user profile for: {username}")
        
        conn.commit()
        conn.close()
        return user_id
    
    def _load_conversation_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Load conversation history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message_role, message_content, timestamp, model_used, rag_context
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to list of message dictionaries (reverse to get chronological order)
        history = []
        for row in reversed(results):
            history.append({
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "model_used": row[3],
                "rag_context": row[4]
            })
        
        return history
    
    def _save_message(self, user_id: int, role: str, content: str, model_used: str = None, rag_context: str = None):
        """Save a message to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (user_id, message_role, message_content, model_used, rag_context)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, role, content, model_used, rag_context))
        
        conn.commit()
        conn.close()
    
    def _save_user_document(self, user_id: int, document_name: str, document_path: str):
        """Save document information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_documents (user_id, document_name, document_path)
            VALUES (?, ?, ?)
        """, (user_id, document_name, document_path))
        
        conn.commit()
        conn.close()
    
    def _get_user_documents(self, user_id: int) -> List[Dict]:
        """Get list of documents for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT document_name, document_path, ingested_at
            FROM user_documents
            WHERE user_id = ?
            ORDER BY ingested_at DESC
        """, (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"name": row[0], "path": row[1], "ingested_at": row[2]} for row in results]
    
    def ingest_documents(self, document_paths: List[str] = None) -> bool:
        """
        Ingest documents for RAG functionality
        
        Args:
            document_paths: List of specific document paths to ingest.
                          If None, ingests all documents in documents_dir
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.current_user:
            print("Error: No user logged in. Please login first.")
            return False
        
        if not self.current_model:
            print("Error: No model selected. Please select a model first.")
            return False
        
        try:
            print("Starting document ingestion...")
            
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
            
            # Create collection name based on user
            collection_name = f"user_{self.current_user.replace(' ', '_').lower()}"
            
            # Initialize vector store
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load documents
            documents = []
            user_id = self._get_or_create_user(self.current_user)
            
            if document_paths:
                # Load specific documents
                for doc_path in document_paths:
                    if os.path.exists(doc_path):
                        reader = SimpleDirectoryReader(input_files=[doc_path])
                        docs = reader.load_data()
                        documents.extend(docs)
                        
                        # Save to user documents
                        doc_name = os.path.basename(doc_path)
                        self._save_user_document(user_id, doc_name, doc_path)
                        print(f"Loaded: {doc_name}")
            else:
                # Load all documents from documents directory
                if any(self.documents_dir.iterdir()):
                    reader = SimpleDirectoryReader(str(self.documents_dir))
                    documents = reader.load_data()
                    
                    # Save all documents to user documents
                    for doc_path in self.documents_dir.iterdir():
                        if doc_path.is_file():
                            self._save_user_document(user_id, doc_path.name, str(doc_path))
                            print(f"Loaded: {doc_path.name}")
                else:
                    print(f"No documents found in {self.documents_dir}")
                    print("Please add documents to the documents/ directory or specify document paths.")
                    return False
            
            if not documents:
                print("No documents to ingest.")
                return False
            
            # Create/update the index
            print(f"Processing {len(documents)} documents...")
            
            # Set up the LLM for LlamaIndex
            Settings.llm = LMStudioLLM(self.client, self.current_model)
            
            # Create the index
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            # Create query engine
            self.query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            
            self.rag_enabled = True
            print(f"✅ Successfully ingested {len(documents)} documents!")
            print("RAG functionality is now enabled.")
            
            return True
            
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return False
    
    def load_existing_index(self) -> bool:
        """Load existing vector index for current user"""
        if not self.current_user or not self.current_model:
            return False
        
        try:
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
            collection_name = f"user_{self.current_user.replace(' ', '_').lower()}"
            
            # Check if collection exists
            try:
                chroma_collection = chroma_client.get_collection(collection_name)
                
                # Check if collection has documents
                if chroma_collection.count() == 0:
                    return False
                
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Load the index
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context
                )
                
                # Set up the LLM for LlamaIndex
                Settings.llm = LMStudioLLM(self.client, self.current_model)
                
                # Create query engine
                self.query_engine = index.as_query_engine(
                    similarity_top_k=3,
                    response_mode="compact"
                )
                
                self.rag_enabled = True
                print("✅ Loaded existing RAG index!")
                return True
                
            except Exception:
                # Collection doesn't exist
                return False
                
        except Exception as e:
            print(f"Error loading existing index: {e}")
            return False
    
    def login_user(self, username: str):
        """Login user and load their conversation history"""
        self.current_user = username
        user_id = self._get_or_create_user(username)
        
        # Load conversation history
        history = self._load_conversation_history(user_id)
        
        # Convert to OpenAI message format (exclude metadata)
        self.conversation_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in history
        ]
        
        print(f"Welcome back, {username}!")
        if len(self.conversation_history) > 0:
            print(f"Loaded {len(self.conversation_history)} previous messages.")
        else:
            print("Starting a new conversation.")
        
        # Try to load existing RAG index
        if self.current_model and self.load_existing_index():
            user_docs = self._get_user_documents(user_id)
            print(f"Found {len(user_docs)} documents in your knowledge base.")
    
    def select_model(self) -> bool:
        """Let user select a model to use"""
        if not self.available_models:
            print("No models available. Make sure LMStudio is running with a model loaded.")
            return False
        
        print("\nAvailable models:")
        for i, model in enumerate(self.available_models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(self.available_models)}): ").strip()
                if choice == "":
                    # Default to first model
                    self.current_model = self.available_models[0]
                    break
                else:
                    choice = int(choice) - 1
                    if 0 <= choice < len(self.available_models):
                        self.current_model = self.available_models[choice]
                        break
                    else:
                        print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        print(f"Selected model: {self.current_model}")
        
        # Try to load existing RAG index if user is logged in
        if self.current_user:
            if self.load_existing_index():
                user_id = self._get_or_create_user(self.current_user)
                user_docs = self._get_user_documents(user_id)
                print(f"Loaded existing knowledge base with {len(user_docs)} documents.")
        
        return True
    
    def chat_with_model(self, user_message: str) -> str:
        """Send message to model (with or without RAG)"""
        if not self.current_model:
            return "Error: No model selected. Please select a model first."
        
        try:
            rag_context = None
            
            # Use RAG if enabled
            if self.rag_enabled and self.query_engine:
                print("🔍 Searching knowledge base...")
                
                # Query the RAG system
                rag_response = self.query_engine.query(user_message)
                rag_context = str(rag_response)
                
                # Enhanced prompt with RAG context
                enhanced_message = f"""Based on the following context from the knowledge base, please answer the user's question:

Context: {rag_context}

User Question: {user_message}

Please provide a comprehensive answer using the context above, and indicate if the information comes from the knowledge base or your general knowledge."""
                
                # Add enhanced message to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": enhanced_message
                })
            else:
                # Regular chat without RAG
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
            
            # Make API call to LMStudio
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            
            # Extract assistant response
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # Save both messages to database
            user_id = self._get_or_create_user(self.current_user)
            self._save_message(user_id, "user", user_message, self.current_model, rag_context)
            self._save_message(user_id, "assistant", assistant_response, self.current_model)
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error communicating with LMStudio: {e}"
            print(error_msg)
            return error_msg
    
    def show_conversation_stats(self):
        """Show conversation statistics for current user"""
        if not self.current_user:
            print("No user logged in.")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        user_id = self._get_or_create_user(self.current_user)
        
        # Get message count
        cursor.execute("""
            SELECT COUNT(*) FROM conversations WHERE user_id = ?
        """, (user_id,))
        total_messages = cursor.fetchone()[0]
        
        # Get RAG-enhanced message count
        cursor.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE user_id = ? AND rag_context IS NOT NULL
        """, (user_id,))
        rag_messages = cursor.fetchone()[0]
        
        # Get unique models used
        cursor.execute("""
            SELECT DISTINCT model_used FROM conversations 
            WHERE user_id = ? AND model_used IS NOT NULL
        """, (user_id,))
        models_used = [row[0] for row in cursor.fetchall()]
        
        # Get document count
        user_docs = self._get_user_documents(user_id)
        
        conn.close()
        
        print(f"\n--- Conversation Stats for {self.current_user} ---")
        print(f"Total messages: {total_messages}")
        print(f"RAG-enhanced messages: {rag_messages}")
        print(f"Documents in knowledge base: {len(user_docs)}")
        print(f"Models used: {', '.join(models_used) if models_used else 'None'}")
        print(f"RAG status: {'✅ Enabled' if self.rag_enabled else '❌ Disabled'}")
        
        if user_docs:
            print(f"\nDocuments:")
            for doc in user_docs[:5]:  # Show first 5
                print(f"  - {doc['name']} (added: {doc['ingested_at'][:10]})")
            if len(user_docs) > 5:
                print(f"  ... and {len(user_docs) - 5} more")
    
    def list_documents(self):
        """List all documents in the knowledge base"""
        if not self.current_user:
            print("No user logged in.")
            return
        
        user_id = self._get_or_create_user(self.current_user)
        user_docs = self._get_user_documents(user_id)
        
        if not user_docs:
            print("No documents in knowledge base.")
            print("Use 'ingest' command to add documents.")
            return
        
        print(f"\n--- Documents in Knowledge Base ---")
        for i, doc in enumerate(user_docs, 1):
            print(f"{i}. {doc['name']}")
            print(f"   Path: {doc['path']}")
            print(f"   Added: {doc['ingested_at']}")
            print()
    
    def clear_conversation_history(self):
        """Clear current conversation history (but keep in database)"""
        self.conversation_history = []
        print("Current conversation history cleared. Database history preserved.")
    
    def run(self):
        """Main chat loop"""
        print("=" * 70)
        print("LMStudio RAG Chatbot with LlamaIndex")
        print("=" * 70)
        print("This chatbot maintains conversation context and supports RAG!")
        print()
        print("Commands:")
        print("  'quit' - Exit the chatbot")
        print("  'clear' - Clear current session")
        print("  'stats' - Show conversation statistics")
        print("  'models' - Change model")
        print("  'ingest' - Ingest documents for RAG")
        print("  'docs' - List documents in knowledge base")
        print("  'rag on/off' - Enable/disable RAG")
        print()
        
        # Get username
        while True:
            username = input("Enter your username: ").strip()
            if username:
                break
            print("Username cannot be empty. Please try again.")
        
        # Login user
        self.login_user(username)
        
        # Select model
        if not self.select_model():
            return
        
        # Show RAG status
        if self.rag_enabled:
            print("🤖 RAG is enabled - I can answer questions about your documents!")
        else:
            print("📝 To enable RAG, add documents to 'documents/' folder and use 'ingest' command.")
        
        print(f"\nYou can now chat with {self.current_model}!")
        print("Type your message and press Enter.\n")
        
        # Main chat loop
        while True:
            try:
                user_input = input(f"{self.current_user}: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_conversation_history()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_conversation_stats()
                    continue
                elif user_input.lower() == 'models':
                    if self.select_model():
                        print(f"Switched to {self.current_model}")
                    continue
                elif user_input.lower() == 'ingest':
                    print("Ingesting documents from 'documents/' directory...")
                    if self.ingest_documents():
                        print("Documents ingested successfully!")
                    else:
                        print("Failed to ingest documents.")
                    continue
                elif user_input.lower() == 'docs':
                    self.list_documents()
                    continue
                elif user_input.lower() == 'rag on':
                    if self.load_existing_index():
                        print("✅ RAG enabled!")
                    else:
                        print("❌ No documents found. Use 'ingest' to add documents first.")
                    continue
                elif user_input.lower() == 'rag off':
                    self.rag_enabled = False
                    self.query_engine = None
                    print("❌ RAG disabled.")
                    continue
                elif user_input == '':
                    print("Please enter a message or command.")
                    continue
                
                # Get response from model
                rag_indicator = "🧠" if self.rag_enabled else "💭"
                print(f"\n{rag_indicator} {self.current_model}: ", end="", flush=True)
                response = self.chat_with_model(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

def main():
    """Main function"""
    chatbot = LMStudioRAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()