"""
Gemini adapter for MentorIA Chatbot API.

Provides data access layer for Google Gemini AI operations.
"""

import logging
import json
from typing import Dict, Any, Optional

import google.generativeai as genai
from src.config import config
from src.adapters.agents.router_agent import RouterAgent
from src.adapters.agents.complex_query_agent import ComplexQueryAgent

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Adapter for Google Gemini AI operations."""

    def __init__(self):
        """Initialize Gemini AI client and agents."""
        try:
            if not config.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            genai.configure(api_key=config.gemini_api_key)
            self.pro_model = genai.GenerativeModel("gemini-1.5-pro-latest")
            self.flash_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            self.router = RouterAgent()
            self.complex_query_agent = ComplexQueryAgent()
            self.config = config
            logger.info("GeminiAdapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiAdapter: {e}", exc_info=True)
            raise

    def get_routing_plan(self, user_message: str, user_profile: Dict[str, Any], conversation_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get routing plan from RouterAgent using Flash model.
        
        Args:
            user_message: User's message
            user_profile: User profile data
            conversation_context: Conversation context
            
        Returns:
            Routing plan dictionary or None if error
        """
        try:
            prompt = self.router.build_prompt(
                user_message, user_profile, conversation_context,
                self.config.knowledge_base_nem, self.config.sep_knowledge_base
            )
            return self._execute_agent_prompt(prompt, model=self.pro_model)
        except Exception as e:
            logger.error(f"Error getting routing plan: {e}", exc_info=True)
            return None

    def get_complex_analysis(self, user_message: str, user_profile: Dict[str, Any], selected_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get complex analysis from ComplexQueryAgent using Pro model.
        
        Args:
            user_message: User's message
            user_profile: User profile data
            selected_context: Selected context for analysis
            
        Returns:
            Complex analysis result or None if error
        """
        try:
            prompt = self.complex_query_agent.build_prompt(
                user_message, user_profile, selected_context
            )
            return self._execute_agent_prompt(prompt, model=self.pro_model)
        except Exception as e:
            logger.error(f"Error getting complex analysis: {e}", exc_info=True)
            return None

    def _execute_agent_prompt(self, prompt: str, model) -> Optional[Dict[str, Any]]:
        """
        Execute agent prompt with specified model.
        
        Args:
            prompt: Prompt to execute
            model: Gemini model to use
            
        Returns:
            Agent response or None if error
        """
        try:
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            response = model.generate_content(prompt, generation_config=generation_config)
            result = json.loads(response.text)
            logger.info(f"Agent response received from {model.model_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing agent prompt on {model.model_name}: {e}", exc_info=True)
            return None 