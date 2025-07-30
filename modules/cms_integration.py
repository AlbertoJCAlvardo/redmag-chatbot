"""
CMS Integration module for Vertex AI Vector Search.

This module provides functionality to integrate with various CMS systems
and migrate content from CMS databases to the vector database.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import json
from datetime import datetime
import hashlib

from vector_search import VectorSearchManager
from embeddings import EmbeddingGenerator
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class CMSConnector(ABC):
    """
    Abstract base class for CMS connectors.
    
    This class defines the interface that all CMS connectors must implement
    to enable content migration to the vector database.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the CMS.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_content(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve content from the CMS.
        
        Args:
            limit (Optional[int]): Maximum number of content items to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of content items with metadata.
        """
        pass
    
    @abstractmethod
    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific content by ID.
        
        Args:
            content_id (str): ID of the content to retrieve.
            
        Returns:
            Optional[Dict[str, Any]]: Content item if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def get_updated_content(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve content updated since a specific date.
        
        Args:
            since (datetime): Date to check for updates since.
            
        Returns:
            List[Dict[str, Any]]: List of updated content items.
        """
        pass


class WordPressConnector(CMSConnector):
    """
    WordPress CMS connector for content migration.
    
    This connector handles integration with WordPress sites using the REST API
    to extract content for vector database migration.
    """
    
    def __init__(self, site_url: str, username: str, password: str):
        """
        Initialize WordPress connector.
        
        Args:
            site_url (str): WordPress site URL.
            username (str): WordPress username or application password.
            password (str): WordPress password or application password.
        """
        self.site_url = site_url.rstrip('/')
        self.username = username
        self.password = password
        self.api_base = f"{self.site_url}/wp-json/wp/v2"
        
    def connect(self) -> bool:
        """
        Test connection to WordPress site.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            import requests
            from requests.auth import HTTPBasicAuth
            
            response = requests.get(
                f"{self.api_base}/posts",
                auth=HTTPBasicAuth(self.username, self.password),
                params={"per_page": 1},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to WordPress site: {self.site_url}")
                return True
            else:
                logger.error(f"Failed to connect to WordPress: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to WordPress: {e}")
            return False
    
    def get_content(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all content from WordPress.
        
        Args:
            limit (Optional[int]): Maximum number of posts to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of WordPress posts with metadata.
        """
        try:
            import requests
            from requests.auth import HTTPBasicAuth
            
            all_posts = []
            page = 1
            per_page = 100
            
            while True:
                response = requests.get(
                    f"{self.api_base}/posts",
                    auth=HTTPBasicAuth(self.username, self.password),
                    params={
                        "per_page": per_page,
                        "page": page,
                        "status": "publish",
                        "_embed": True
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    break
                
                posts = response.json()
                if not posts:
                    break
                
                for post in posts:
                    content_item = {
                        "id": f"wp_{post['id']}",
                        "title": post.get('title', {}).get('rendered', ''),
                        "content": post.get('content', {}).get('rendered', ''),
                        "excerpt": post.get('excerpt', {}).get('rendered', ''),
                        "author": post.get('author', ''),
                        "date": post.get('date', ''),
                        "modified": post.get('modified', ''),
                        "categories": [cat['name'] for cat in post.get('_embedded', {}).get('wp:term', [[]])[0] if cat.get('taxonomy') == 'category'],
                        "tags": [tag['name'] for tag in post.get('_embedded', {}).get('wp:term', [[]])[1] if tag.get('taxonomy') == 'post_tag'],
                        "url": post.get('link', ''),
                        "type": "wordpress_post"
                    }
                    all_posts.append(content_item)
                
                if limit and len(all_posts) >= limit:
                    all_posts = all_posts[:limit]
                    break
                
                page += 1
            
            logger.info(f"Retrieved {len(all_posts)} posts from WordPress")
            return all_posts
            
        except Exception as e:
            logger.error(f"Error retrieving WordPress content: {e}")
            return []
    
    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific post by ID.
        
        Args:
            content_id (str): WordPress post ID.
            
        Returns:
            Optional[Dict[str, Any]]: Post data if found, None otherwise.
        """
        try:
            import requests
            from requests.auth import HTTPBasicAuth
            
            # Remove 'wp_' prefix if present
            post_id = content_id.replace('wp_', '')
            
            response = requests.get(
                f"{self.api_base}/posts/{post_id}",
                auth=HTTPBasicAuth(self.username, self.password),
                params={"_embed": True},
                timeout=30
            )
            
            if response.status_code == 200:
                post = response.json()
                return {
                    "id": f"wp_{post['id']}",
                    "title": post.get('title', {}).get('rendered', ''),
                    "content": post.get('content', {}).get('rendered', ''),
                    "excerpt": post.get('excerpt', {}).get('rendered', ''),
                    "author": post.get('author', ''),
                    "date": post.get('date', ''),
                    "modified": post.get('modified', ''),
                    "categories": [cat['name'] for cat in post.get('_embedded', {}).get('wp:term', [[]])[0] if cat.get('taxonomy') == 'category'],
                    "tags": [tag['name'] for tag in post.get('_embedded', {}).get('wp:term', [[]])[1] if tag.get('taxonomy') == 'post_tag'],
                    "url": post.get('link', ''),
                    "type": "wordpress_post"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving WordPress post {content_id}: {e}")
            return None
    
    def get_updated_content(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve posts updated since a specific date.
        
        Args:
            since (datetime): Date to check for updates since.
            
        Returns:
            List[Dict[str, Any]]: List of updated posts.
        """
        try:
            import requests
            from requests.auth import HTTPBasicAuth
            
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
            
            response = requests.get(
                f"{self.api_base}/posts",
                auth=HTTPBasicAuth(self.username, self.password),
                params={
                    "per_page": 100,
                    "modified_after": since_str,
                    "status": "publish",
                    "_embed": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                posts = response.json()
                updated_posts = []
                
                for post in posts:
                    content_item = {
                        "id": f"wp_{post['id']}",
                        "title": post.get('title', {}).get('rendered', ''),
                        "content": post.get('content', {}).get('rendered', ''),
                        "excerpt": post.get('excerpt', {}).get('rendered', ''),
                        "author": post.get('author', ''),
                        "date": post.get('date', ''),
                        "modified": post.get('modified', ''),
                        "categories": [cat['name'] for cat in post.get('_embedded', {}).get('wp:term', [[]])[0] if cat.get('taxonomy') == 'category'],
                        "tags": [tag['name'] for tag in post.get('_embedded', {}).get('wp:term', [[]])[1] if tag.get('taxonomy') == 'post_tag'],
                        "url": post.get('link', ''),
                        "type": "wordpress_post"
                    }
                    updated_posts.append(content_item)
                
                logger.info(f"Retrieved {len(updated_posts)} updated posts from WordPress")
                return updated_posts
            
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving updated WordPress content: {e}")
            return []


class CMSIntegration:
    """
    Main CMS integration class for content migration.
    
    This class coordinates the migration of content from various CMS systems
    to the Vertex AI Vector Search database.
    """
    
    def __init__(self, vector_manager: VectorSearchManager):
        """
        Initialize CMS integration.
        
        Args:
            vector_manager (VectorSearchManager): Vector search manager instance.
        """
        self.vector_manager = vector_manager
        self.embedding_generator = EmbeddingGenerator()
        
    def migrate_content(
        self, 
        cms_connector: CMSConnector,
        content_processor: Optional[Callable] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Migrate content from CMS to vector database.
        
        Args:
            cms_connector (CMSConnector): CMS connector instance.
            content_processor (Optional[Callable]): Function to process content before migration.
            batch_size (Optional[int]): Size of batches for processing.
            
        Returns:
            Dict[str, Any]: Migration results and statistics.
        """
        try:
            # Connect to CMS
            if not cms_connector.connect():
                raise Exception("Failed to connect to CMS")
            
            # Retrieve content
            logger.info("Retrieving content from CMS...")
            content_items = cms_connector.get_content()
            
            if not content_items:
                logger.warning("No content found in CMS")
                return {"success": False, "message": "No content found"}
            
            # Process content if processor provided
            if content_processor:
                logger.info("Processing content...")
                content_items = [content_processor(item) for item in content_items]
                content_items = [item for item in content_items if item]  # Remove None items
            
            # Prepare documents for vector database
            documents = []
            for item in content_items:
                # Create document ID if not present
                if 'id' not in item or not item['id']:
                    item['id'] = self._generate_document_id(item)
                
                # Combine title and content for embedding
                title = item.get('title', '')
                content = item.get('content', '')
                excerpt = item.get('excerpt', '')
                
                combined_content = f"{title}\n\n{excerpt}\n\n{content}".strip()
                
                if not combined_content:
                    logger.warning(f"Skipping item {item['id']} - no content")
                    continue
                
                document = {
                    'id': item['id'],
                    'content': combined_content,
                    'metadata': {
                        'title': title,
                        'excerpt': excerpt,
                        'url': item.get('url', ''),
                        'author': item.get('author', ''),
                        'date': item.get('date', ''),
                        'modified': item.get('modified', ''),
                        'categories': item.get('categories', []),
                        'tags': item.get('tags', []),
                        'type': item.get('type', 'cms_content'),
                        'source': 'cms_migration'
                    }
                }
                documents.append(document)
            
            # Insert documents into vector database
            logger.info(f"Migrating {len(documents)} documents to vector database...")
            results = self.vector_manager.insert_documents_batch(documents, batch_size)
            
            # Calculate statistics
            successful = len([r for r in results.values() if r])
            failed = len(results) - successful
            
            migration_stats = {
                "success": True,
                "total_content_items": len(content_items),
                "total_documents": len(documents),
                "successful_insertions": successful,
                "failed_insertions": failed,
                "results": results
            }
            
            logger.info(f"Migration completed: {successful} successful, {failed} failed")
            return migration_stats
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def sync_updates(
        self, 
        cms_connector: CMSConnector,
        last_sync: datetime,
        content_processor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Sync updated content from CMS since last sync.
        
        Args:
            cms_connector (CMSConnector): CMS connector instance.
            last_sync (datetime): Last sync timestamp.
            content_processor (Optional[Callable]): Function to process content.
            
        Returns:
            Dict[str, Any]: Sync results and statistics.
        """
        try:
            # Connect to CMS
            if not cms_connector.connect():
                raise Exception("Failed to connect to CMS")
            
            # Get updated content
            logger.info(f"Retrieving content updated since {last_sync}...")
            updated_items = cms_connector.get_updated_content(last_sync)
            
            if not updated_items:
                logger.info("No updates found since last sync")
                return {"success": True, "updates": 0}
            
            # Process and update documents
            updated_count = 0
            for item in updated_items:
                try:
                    # Process content if processor provided
                    if content_processor:
                        item = content_processor(item)
                        if not item:
                            continue
                    
                    # Create document ID if not present
                    if 'id' not in item or not item['id']:
                        item['id'] = self._generate_document_id(item)
                    
                    # Combine content
                    title = item.get('title', '')
                    content = item.get('content', '')
                    excerpt = item.get('excerpt', '')
                    combined_content = f"{title}\n\n{excerpt}\n\n{content}".strip()
                    
                    if not combined_content:
                        continue
                    
                    # Update document in vector database
                    success = self.vector_manager.update_document(
                        document_id=item['id'],
                        content=combined_content,
                        metadata={
                            'title': title,
                            'excerpt': excerpt,
                            'url': item.get('url', ''),
                            'author': item.get('author', ''),
                            'date': item.get('date', ''),
                            'modified': item.get('modified', ''),
                            'categories': item.get('categories', []),
                            'tags': item.get('tags', []),
                            'type': item.get('type', 'cms_content'),
                            'source': 'cms_sync',
                            'last_sync': datetime.utcnow().isoformat()
                        }
                    )
                    
                    if success:
                        updated_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to update item {item.get('id', 'unknown')}: {e}")
            
            logger.info(f"Sync completed: {updated_count} documents updated")
            return {
                "success": True,
                "updates": updated_count,
                "total_updated_items": len(updated_items)
            }
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_document_id(self, content_item: Dict[str, Any]) -> str:
        """
        Generate a unique document ID for content item.
        
        Args:
            content_item (Dict[str, Any]): Content item data.
            
        Returns:
            str: Unique document ID.
        """
        # Use URL if available, otherwise use title and date
        if content_item.get('url'):
            return hashlib.md5(content_item['url'].encode()).hexdigest()
        
        title = content_item.get('title', '')
        date = content_item.get('date', '')
        content_hash = hashlib.md5(f"{title}{date}".encode()).hexdigest()
        
        return f"cms_{content_hash}" 