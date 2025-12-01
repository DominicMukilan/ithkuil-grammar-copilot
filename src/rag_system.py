"""
RAG System - Semantic Search Over Grammar Rules
==============================================

Uses ChromaDB + sentence-transformers for retrieval-augmented generation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")


@dataclass
class RetrievedChunk:
    """A grammar chunk retrieved from RAG"""
    case_code: str
    case_name: str
    semantic_role: str
    description: str
    citation: str
    score: float  # Similarity score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_code": self.case_code,
            "case_name": self.case_name,
            "semantic_role": self.semantic_role,
            "description": self.description,
            "citation": self.citation,
            "score": self.score
        }


class RAGSystem:
    """
    Retrieval-Augmented Generation system for Ithkuil grammar
    
    Uses semantic search to find relevant grammar rules based on queries.
    """
    
    def __init__(self, grammar_file: Path, collection_name: str = "ithkuil_grammar"):
        """
        Initialize RAG system
        
        Args:
            grammar_file: Path to grammar_chunks.json
            collection_name: Name for ChromaDB collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB required for RAG")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for RAG")
        
        self.grammar_file = grammar_file
        self.collection_name = collection_name
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
        print("✅ Embedding model loaded")
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Load or create collection
        self.collection = self._initialize_collection()
        
        print(f"✅ RAG system ready with {self.collection.count()} chunks")
    
    def _initialize_collection(self):
        """Load grammar chunks into ChromaDB"""
        # Try to get existing collection
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"📚 Loaded existing collection: {collection.count()} chunks")
            return collection
        except:
            # Create new collection
            print("🔄 Creating new collection...")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Ithkuil grammar rules"}
            )
            
            # Load and embed grammar chunks
            chunks = self._load_grammar_chunks()
            self._embed_and_store(chunks, collection)
            
            return collection
    
    def _load_grammar_chunks(self) -> List[Dict[str, Any]]:
        """Load grammar chunks from JSON"""
        with open(self.grammar_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter to only cases (most important for validation)
        cases = [item for item in data if item.get('type') == 'case']
        print(f"📖 Loaded {len(cases)} case definitions")
        return cases
    
    def _embed_and_store(self, chunks: List[Dict], collection):
        """Embed chunks and store in ChromaDB"""
        print(f"🔄 Embedding {len(chunks)} chunks...")
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Create embedding text
            embed_text = chunk.get('embedding_text', '')
            if not embed_text:
                # Fallback: combine key fields
                embed_text = f"{chunk['name']} {chunk['code']} {chunk.get('semantic_role', '')} {chunk.get('description', '')}"
            
            documents.append(embed_text)
            metadatas.append({
                "code": chunk['code'],
                "name": chunk['name'],
                "semantic_role": chunk.get('semantic_role', ''),
                "description": chunk.get('description', '')[:200],  # Truncate
                "citation": chunk.get('citation', ''),
            })
            ids.append(chunk['id'])
        
        # Embed all at once (faster)
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        filter_case: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant grammar chunks
        
        Args:
            query: Semantic query (e.g., "unwilled experience")
            n_results: Number of results to return
            filter_case: Optional case code to filter by
            
        Returns:
            List of retrieved chunks with similarity scores
        """
        # Build query
        where = {"code": filter_case} if filter_case else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Parse results
        chunks = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (1 = perfect match, 0 = no match)
                score = 1.0 - (distance / 2.0)  # Normalize L2 distance
                
                chunks.append(RetrievedChunk(
                    case_code=metadata['code'],
                    case_name=metadata['name'],
                    semantic_role=metadata['semantic_role'],
                    description=metadata['description'],
                    citation=metadata['citation'],
                    score=max(0.0, min(1.0, score))  # Clamp to [0, 1]
                ))
        
        return chunks
    
    def retrieve_for_case(self, case_code: str) -> Optional[RetrievedChunk]:
        """
        Retrieve specific case information
        
        Args:
            case_code: Case code (e.g., "AFF")
            
        Returns:
            Case information or None if not found
        """
        chunks = self.retrieve(
            query=case_code,
            n_results=1,
            filter_case=case_code
        )
        return chunks[0] if chunks else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_chunks": self.collection.count(),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "collection_name": self.collection_name
        }


# Quick test
if __name__ == "__main__":
    # Test with grammar data
    grammar_file = Path("data_grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data/grammar_chunks.json")
    
    print("=" * 70)
    print("RAG SYSTEM TEST")
    print("=" * 70)
    
    # Initialize
    rag = RAGSystem(grammar_file)
    
    print("\n" + "=" * 70)
    print("TEST 1: Query by semantic meaning")
    print("=" * 70)
    
    # Test query
    query = "unwilled experience feeling emotion"
    print(f"\nQuery: '{query}'")
    results = rag.retrieve(query, n_results=3)
    
    print(f"\nTop {len(results)} results:")
    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. {chunk.case_code} - {chunk.case_name}")
        print(f"   Semantic role: {chunk.semantic_role}")
        print(f"   Score: {chunk.score:.3f}")
        print(f"   Description: {chunk.description[:80]}...")
    
    print("\n" + "=" * 70)
    print("TEST 2: Retrieve specific case")
    print("=" * 70)
    
    case = "AFF"
    print(f"\nRetrieving: {case}")
    result = rag.retrieve_for_case(case)
    
    if result:
        print(f"\n✅ Found: {result.case_name}")
        print(f"   Semantic role: {result.semantic_role}")
        print(f"   Citation: {result.citation}")
    
    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)
    print(json.dumps(rag.get_stats(), indent=2))