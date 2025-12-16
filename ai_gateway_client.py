"""
AI Gateway Client
Enhanced client for organization's AI Gateway with HMAC authentication
"""

import httpx
import os
import json
import hmac
import hashlib
import base64
import time
import uuid
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AIGatewayClient:
    """Enhanced client for AI Gateway with HMAC authentication"""
    
    def __init__(self):
        self.gateway_url = os.getenv("HOST_URL")  # Changed to match LLMConnection.py
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        self.model = os.getenv("AI_MODEL", "gpt-4o")
        self.max_tokens = int(os.getenv("AI_MAX_TOKENS", "9000"))  # Match LLMConnection.py
        self.temperature = float(os.getenv("AI_TEMPERATURE", "0"))  # Match LLMConnection.py
        
        if not all([self.gateway_url, self.api_key, self.api_secret]):
            logger.warning("AI Gateway credentials not fully configured")
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response using the organization's AI Gateway with HMAC authentication
        
        Args:
            prompt: The user prompt/question
            context: Additional context for the LLM
            
        Returns:
            Dict containing the LLM response
        """
        try:
            if not self._is_configured():
                return {
                    "success": False,
                    "response": "AI Gateway is not properly configured. Please check your credentials.",
                    "error": "missing_credentials"
                }
            
            # Prepare system message for DevOps context
            system_message = ""
            if context:
                system_message = """You are an expert DevOps assistant with deep knowledge of:
- AWS services and cloud architecture
- CI/CD pipelines and automation
- Infrastructure as Code (Terraform, CloudFormation)
- Container technologies (Docker, Kubernetes)
- Monitoring and observability
- Security best practices
- Deployment strategies and rollback procedures

Provide practical, actionable advice with specific examples when possible."""
            
            # Create request body in OpenAI format (matching LLMConnection.py)
            request_body = {
                "messages": [
                    {"content": system_message, "role": "system"},
                    {"content": prompt, "role": "user"},
                ],
                "frequency_penalty": 0,
                "max_tokens": self.max_tokens,
                "n": 1,
                "presence_penalty": 0,
                "response_format": {"type": "text"},
                "stream": False,
                "temperature": self.temperature,
                "top_p": 1
            }
            
            # Create HMAC signature (matching LLMConnection.py)
            timestamp = int(time.time() * 1000)
            request_id = uuid.uuid4()
            hmac_source_data = self.api_key + str(request_id) + str(timestamp) + json.dumps(request_body)
            computed_hash = hmac.new(self.api_secret.encode(), hmac_source_data.encode(), hashlib.sha256)
            hmac_signature = base64.b64encode(computed_hash.digest()).decode()
            
            # Prepare headers with HMAC authentication (matching LLMConnection.py)
            headers = {
                "api-key": self.api_key,
                "Client-Request-Id": str(request_id),
                "Timestamp": str(timestamp),
                "Authorization": hmac_signature,
                "Accept": "application/json",
            }
            
            # Make request to /chat/completions endpoint
            endpoint_url = f"{self.gateway_url}/chat/completions"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    endpoint_url,
                    json=request_body,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response in OpenAI format (matching LLMConnection.py)
                    ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response received')
                    
                    return {
                        "success": True,
                        "response": ai_response,
                        "model": result.get("model", self.model),
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "finish_reason": result.get('choices', [{}])[0].get('finish_reason', 'completed')
                    }
                else:
                    logger.error(f"AI Gateway returned status {response.status_code}: {response.text}")
                    return {
                        "success": False,
                        "response": f"AI Gateway error (status {response.status_code})",
                        "error": "gateway_error",
                        "status_code": response.status_code
                    }
                    
        except httpx.TimeoutException:
            logger.error("AI Gateway request timed out")
            return {
                "success": False,
                "response": "AI Gateway request timed out. Please try again.",
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"AI Gateway client error: {str(e)}")
            return {
                "success": False,
                "response": f"Error communicating with AI Gateway: {str(e)}",
                "error": "client_error"
            }
    
    async def generate_with_tools_context(self, prompt: str, available_tools: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response with tools context for LangChain integration"""
        
        tools_description = f"""
Available DevOps Tools:
{chr(10).join(f"- {tool}" for tool in available_tools)}

Use these tools when you need to perform specific operations.
"""
        
        enhanced_prompt = f"{tools_description}\n\nUser Request: {prompt}"
        
        return await self.generate_response(enhanced_prompt, context)
    
    def _is_configured(self) -> bool:
        """Check if the AI Gateway is properly configured"""
        return bool(self.gateway_url and self.api_key and self.api_secret)
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to AI Gateway"""
        try:
            test_response = await self.generate_response(
                "Hello, this is a connection test. Please respond with 'Connection successful' and confirm you're using GPT-4o."
            )
            return test_response
        except Exception as e:
            return {
                "success": False,
                "response": f"Connection test failed: {str(e)}",
                "error": "connection_test_failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "gateway_url": self.gateway_url[:50] + "..." if self.gateway_url and len(self.gateway_url) > 50 else self.gateway_url,
            "configured": self._is_configured()
        }