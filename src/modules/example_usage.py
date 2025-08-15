"""
Example usage of Vertex AI Vector Search modules.

This file demonstrates how to use the various components
for vector search operations, embedding generation, and CMS integration.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Import our modules
from config import config
from embeddings import EmbeddingGenerator
from vector_search import VectorSearchManager
from cms_integration import CMSIntegration, WordPressConnector


def setup_environment():
    """Setup and validate environment configuration."""
    print("=== Environment Setup ===")
    
    try:
        # Validate configuration
        config.validate()
        print("✓ Configuration validated successfully")
        
        # Setup authentication
        config.setup_authentication()
        print("✓ Authentication setup completed")
        
        # Display configuration
        print(f"Project ID: {config.project_id}")
        print(f"Location: {config.location}")
        print(f"Index ID: {config.index_id}")
        print(f"Embedding Model: {config.embedding_model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False


def embedding_example():
    """Example of embedding generation with long text support."""
    print("\n=== Embedding Generation Example ===")
    
    try:
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator()
        print("✓ Embedding generator initialized")
        
        # Example 1: Short text
        short_text = "This is a short text for embedding generation."
        embedding = embedding_gen.generate_embedding(short_text)
        print(f"✓ Generated embedding for short text (dimension: {len(embedding)})")
        
        # Example 2: Long text (will be split into chunks)
        long_text = """
        This is a very long text that exceeds the token limit for the embedding model. 
        It contains multiple sentences and paragraphs that need to be processed in chunks.
        The embedding generator will automatically split this text into appropriate chunks,
        generate embeddings for each chunk, and then average them to create a single
        representative embedding for the entire text. This approach ensures that we can
        handle documents of any length while maintaining the quality of the embeddings.
        
        The chunking process is intelligent and tries to split at sentence boundaries
        first, then at word boundaries if necessary. This preserves the semantic meaning
        of the text as much as possible while staying within the token limits of the
        embedding model. The final embedding represents the overall meaning of the
        entire document, making it suitable for vector search operations.
        
        This capability is particularly useful for processing large documents,
        articles, or any content that might exceed the typical token limits of
        embedding models. It allows us to work with real-world content without
        having to truncate or lose important information.
        """
        
        embedding = embedding_gen.generate_embedding(long_text)
        print(f"✓ Generated embedding for long text (dimension: {len(embedding)})")
        
        # Example 3: Batch embeddings
        texts = [
            "First sample text for batch processing.",
            "Second sample text with different content.",
            "Third sample text to demonstrate batch embedding generation."
        ]
        
        embeddings = embedding_gen.generate_embeddings_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings in batch")
        
        # Example 4: Long text with multiple chunks
        chunk_embeddings = embedding_gen.generate_embeddings_for_long_text(long_text)
        print(f"✓ Generated {len(chunk_embeddings)} embeddings for long text chunks")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding example failed: {e}")
        return False


def vector_search_example():
    """Example of vector search operations."""
    print("\n=== Vector Search Example ===")
    
    try:
        # Initialize vector search manager
        vector_manager = VectorSearchManager()
        print("✓ Vector search manager initialized")
        
        # Example documents
        documents = [
            {
                "id": "doc_001",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"title": "Python Programming", "category": "programming"}
            },
            {
                "id": "doc_002",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "metadata": {"title": "Machine Learning Basics", "category": "ai"}
            },
            {
                "id": "doc_003",
                "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "metadata": {"title": "Data Science Overview", "category": "data"}
            }
        ]
        
        # Insert documents
        results = vector_manager.insert_documents_batch(documents)
        successful = len([r for r in results.values() if r])
        print(f"✓ Inserted {successful} documents successfully")
        
        # Search for similar documents
        query = "What is programming?"
        search_results = vector_manager.search_similar(query, num_neighbors=3)
        print(f"✓ Found {len(search_results)} similar documents for query: '{query}'")
        
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. Document ID: {result['document_id']}, Distance: {result['distance']:.4f}")
        
        # Update a document
        success = vector_manager.update_document(
            document_id="doc_001",
            content="Python is a versatile high-level programming language known for its simplicity, readability, and extensive library ecosystem.",
            metadata={"title": "Python Programming - Updated", "category": "programming", "updated": True}
        )
        print(f"✓ Document update: {'Success' if success else 'Failed'}")
        
        # Get a specific document
        document = vector_manager.get_document("doc_002")
        if document:
            print(f"✓ Retrieved document: {document['document_id']}")
        
        # Clean up - delete documents
        doc_ids = ["doc_001", "doc_002", "doc_003"]
        delete_results = vector_manager.delete_documents_batch(doc_ids)
        deleted = len([r for r in delete_results.values() if r])
        print(f"✓ Deleted {deleted} documents")
        
        return True
        
    except Exception as e:
        print(f"✗ Vector search example failed: {e}")
        return False


def cms_integration_example():
    """Example of CMS integration (requires actual CMS credentials)."""
    print("\n=== CMS Integration Example ===")
    
    try:
        # Initialize vector search manager
        vector_manager = VectorSearchManager()
        
        # Initialize CMS integration
        cms_integration = CMSIntegration(vector_manager)
        print("✓ CMS integration initialized")
        
        # Example WordPress connector (commented out as it requires actual credentials)
        # wordpress_connector = WordPressConnector(
        #     site_url="https://your-wordpress-site.com",
        #     username="your-username",
        #     password="your-application-password"
        # )
        
        # Example content processor function
        def content_processor(content_item: Dict[str, Any]) -> Dict[str, Any]:
            """Process content before migration."""
            # Add custom processing logic here
            content_item["processed"] = True
            content_item["processed_at"] = datetime.utcnow().isoformat()
            return content_item
        
        # Example migration (commented out as it requires actual CMS)
        # migration_result = cms_integration.migrate_content(
        #     cms_connector=wordpress_connector,
        #     content_processor=content_processor,
        #     batch_size=50
        # )
        # print(f"Migration result: {migration_result}")
        
        print("✓ CMS integration example completed (requires actual CMS credentials)")
        print("  To use this functionality:")
        print("  1. Configure WordPress credentials in .env file")
        print("  2. Uncomment the WordPress connector code")
        print("  3. Uncomment the migration code")
        
        return True
        
    except Exception as e:
        print(f"✗ CMS integration example failed: {e}")
        return False


def chatbot_integration_example():
    """Example of chatbot integration with vector search."""
    print("\n=== Chatbot Integration Example ===")
    
    try:
        # Initialize vector search manager
        vector_manager = VectorSearchManager()
        
        # Insert sample knowledge base documents
        knowledge_base = [
            {
                "id": "kb_001",
                "content": "Our company provides cloud computing solutions for businesses of all sizes. We offer scalable infrastructure, managed services, and 24/7 support.",
                "metadata": {"category": "company_info", "topic": "services"}
            },
            {
                "id": "kb_002",
                "content": "We offer 24/7 customer support through phone, email, and live chat. Our support team is available around the clock to help with any issues.",
                "metadata": {"category": "support", "topic": "customer_service"}
            },
            {
                "id": "kb_003",
                "content": "Our pricing plans start at $29/month for basic features and scale up to enterprise solutions. Contact us for custom pricing.",
                "metadata": {"category": "pricing", "topic": "costs"}
            }
        ]
        
        # Insert knowledge base
        results = vector_manager.insert_documents_batch(knowledge_base)
        successful = len([r for r in results.values() if r])
        print(f"✓ Inserted {successful} knowledge base documents")
        
        # Simulate chatbot queries
        chatbot_queries = [
            "What services do you offer?",
            "How can I get customer support?",
            "What are your pricing options?",
            "Do you have enterprise solutions?"
        ]
        
        print("\nChatbot Query Results:")
        for query in chatbot_queries:
            print(f"\nQuery: {query}")
            results = vector_manager.search_similar(query, num_neighbors=2)
            
            print("Relevant context:")
            for i, result in enumerate(results, 1):
                relevance_score = 1.0 - result["distance"]
                print(f"  {i}. {result['document_id']} (relevance: {relevance_score:.3f})")
        
        # Clean up
        doc_ids = ["kb_001", "kb_002", "kb_003"]
        vector_manager.delete_documents_batch(doc_ids)
        print(f"\n✓ Cleaned up knowledge base documents")
        
        return True
        
    except Exception as e:
        print(f"✗ Chatbot integration example failed: {e}")
        return False


def main():
    """Run all examples."""
    print("Vertex AI Vector Search - Example Usage")
    print("=" * 50)
    
    # Check if environment is properly configured
    if not setup_environment():
        print("\nPlease configure your environment variables and service account before running examples.")
        print("See env_template.txt for required variables.")
        return
    
    # Run examples
    examples = [
        ("Embedding Generation", embedding_example),
        ("Vector Search Operations", vector_search_example),
        ("CMS Integration", cms_integration_example),
        ("Chatbot Integration", chatbot_integration_example)
    ]
    
    results = []
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        success = example_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("EXAMPLE SUMMARY")
    print("=" * 50)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} examples passed")


if __name__ == "__main__":
    main() 