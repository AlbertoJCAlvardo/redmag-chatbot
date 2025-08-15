"""
BigQuery adapter for MentorIA Chatbot API.

Provides data access layer for BigQuery operations.
"""

import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from src.config import config

logger = logging.getLogger(__name__)


class BigQueryAdapter:
    """Adapter for BigQuery database operations."""

    def __init__(self):
        """Initialize BigQuery client."""
        try:
            self.client = bigquery.Client(project=config.project_id)
            self.users_table_id = config.bigquery_users_table
            self.messages_table_id = config.bigquery_messages_table
            self.context_table_id = config.bigquery_context_table
            logger.info("BigQueryAdapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
            # No raise exception to allow application to continue without BigQuery
            self.client = None

    def _is_available(self) -> bool:
        """Check if BigQuery is available."""
        return self.client is not None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile data from BigQuery.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User profile data dictionary or None if not found
        """
        if not self._is_available():
            logger.warning("BigQuery not available, returning empty profile")
            return {}
            
        query = f"""
            SELECT profile_data
            FROM `{self.users_table_id}`
            WHERE user_id = @user_id
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        )
        try:
            query_job = self.client.query(query, job_config=job_config)
            rows = list(query_job)
            if rows:
                return rows[0].profile_data
            return None
        except Exception as e:
            logger.error(f"Failed to get user profile for user_id {user_id}: {e}", exc_info=True)
            return None

    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Update or create user profile in BigQuery.
        
        Args:
            user_id: Unique user identifier
            profile_data: User profile data dictionary
            
        Returns:
            True if operation was successful, False otherwise
        """
        if not self._is_available():
            logger.warning("BigQuery not available, skipping profile update")
            return True
            
        query = f"""
            MERGE `{self.users_table_id}` T
            USING (SELECT @user_id AS user_id) S
            ON T.user_id = S.user_id
            WHEN MATCHED THEN
                UPDATE SET profile_data = @profile_data, last_created = @now
            WHEN NOT MATCHED THEN
                INSERT (user_id, created_at, last_created, profile_data)
                VALUES (@user_id, @now, @now, @profile_data)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("profile_data", "JSON", json.dumps(profile_data)),
                bigquery.ScalarQueryParameter("now", "DATETIME", datetime.now(timezone.utc)),
            ]
        )
        try:
            self.client.query(query, job_config=job_config).result()
            logger.info(f"Successfully updated profile for user_id: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user profile for user_id {user_id}: {e}", exc_info=True)
            return False

    def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context from BigQuery.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Conversation context data dictionary or None if not found
        """
        if not self._is_available():
            logger.warning("BigQuery not available, returning empty context")
            return {}
            
        query = f"""
            SELECT context_data
            FROM `{self.context_table_id}`
            WHERE conversation_id = @conversation_id AND is_active = TRUE
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id)]
        )
        try:
            rows = list(self.client.query(query, job_config=job_config))
            if rows:
                context_data = rows[0].context_data
                # Si context_data es un string JSON, parsearlo a diccionario
                if isinstance(context_data, str):
                    try:
                        return json.loads(context_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse context_data JSON for conversation_id {conversation_id}")
                        return {}
                return context_data
            return None
        except Exception as e:
            logger.error(f"Failed to get context for conversation_id {conversation_id}: {e}", exc_info=True)
            return None

    def update_conversation_context(self, conversation_id: str, user_id: str, context_data: Dict[str, Any]) -> bool:
        """
        Update or create conversation context in BigQuery.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            context_data: Conversation context data dictionary
            
        Returns:
            True if operation was successful, False otherwise
        """
        if not self._is_available():
            logger.warning("BigQuery not available, skipping context update")
            return True
            
        query = f"""
            MERGE `{self.context_table_id}` T
            USING (SELECT @conversation_id AS conversation_id) S
            ON T.conversation_id = S.conversation_id
            WHEN MATCHED THEN
                UPDATE SET context_data = @context_data, last_updated = @now
            WHEN NOT MATCHED THEN
                INSERT (conversation_id, user_id, created_at, last_updated, context_data, is_active)
                VALUES (@conversation_id, @user_id, @now, @now, @context_data, TRUE)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("context_data", "JSON", json.dumps(context_data)),
                bigquery.ScalarQueryParameter("now", "TIMESTAMP", datetime.now(timezone.utc)),
            ]
        )
        try:
            self.client.query(query, job_config=job_config).result()
            logger.info(f"Successfully updated context for conversation_id: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update context for conversation_id {conversation_id}: {e}", exc_info=True)
            return False

    def log_message(self, conversation_id: str, user_id: str, message_id: str, role: str, content: str, agent_response: bool) -> bool:
        """
        Log message to BigQuery.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            message_id: Unique message identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            agent_response: True if message is from agent
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._is_available():
            logger.warning("BigQuery not available, skipping message logging")
            return True
            
        row_to_insert = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": content,
            "agent_response": str(agent_response),
            "flow": "chat"
        }
        try:
            errors = self.client.insert_rows_json(self.messages_table_id, [row_to_insert])
            if not errors:
                logger.debug(f"Successfully logged message {message_id}")
                return True
            else:
                logger.error(f"Errors occurred while inserting message {message_id}: {errors}")
                return False
        except Exception as e:
            logger.error(f"Failed to log message {message_id}: {e}", exc_info=True)
            return False 