"""Memory management system using FAISS for vector storage and retrieval."""

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import os
from pathlib import Path

class MemoryManager:
    def __init__(self, dimension: int = 1536, index_path: str = "memory_index"):
        """Initialize the memory manager with a FAISS index and metadata storage.
        
        Args:
            dimension: Dimension of the embeddings (default: 1536 for text-embedding-ada-002)
            index_path: Path to save/load the FAISS index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = f"{index_path}_metadata.json"
        
        # Initialize or load the FAISS index
        if os.path.exists(f"{index_path}.index"):
            self.index = faiss.read_index(f"{index_path}.index")
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Initialize or load metadata storage
        self.metadata: List[Dict[str, Any]] = []
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for input text using OpenAI's API.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings
        """
        from ..llm_client import openai_client
        
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def store_chunk(self, text: str, metadata: Dict[str, Any]) -> int:
        """Store a text chunk and its metadata in the vector store.
        
        Args:
            text: Text content to store
            metadata: Associated metadata (source, date, etc.)
            
        Returns:
            ID of the stored chunk
        """
        # Generate embeddings
        vector = self.embed_text(text)
        
        # Add to FAISS index
        self.index.add(np.array([vector]))
        
        # Store metadata
        chunk_id = len(self.metadata)
        metadata['chunk_id'] = chunk_id
        metadata['timestamp'] = datetime.now().isoformat()
        self.metadata.append(metadata)
        
        # Save to disk
        self._save_to_disk()
        
        return chunk_id
    
    def query_chunks(self, query_text: str, k: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """Query the vector store for relevant chunks.
        
        Args:
            query_text: Query text to find relevant chunks
            k: Number of chunks to retrieve
            
        Returns:
            List of (text, metadata) tuples for the most relevant chunks
        """
        # Generate query embeddings
        query_vec = self.embed_text(query_text)
        
        # Search the index
        distances, indices = self.index.search(np.array([query_vec]), k)
        
        # Retrieve chunks and metadata
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):  # Ensure valid index
                metadata = self.metadata[idx]
                results.append((metadata['text'], metadata))
        
        return results
    
    def _save_to_disk(self):
        """Save the FAISS index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_chunk_by_id(self, chunk_id: int) -> Tuple[str, Dict[str, Any]]:
        """Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Tuple of (text, metadata) for the requested chunk
        """
        if 0 <= chunk_id < len(self.metadata):
            metadata = self.metadata[chunk_id]
            return metadata['text'], metadata
        raise ValueError(f"Chunk ID {chunk_id} not found")
    
    def clear_memory(self):
        """Clear all stored chunks and reset the index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save_to_disk() 