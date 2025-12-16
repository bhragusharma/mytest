from typing import Any, Dict, List, Optional, Callable

import hmac
import hashlib
import base64
import json
import requests
import time
import uuid
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from secrets_loader import load_secrets
import os


from typing import Sequence, Union, Dict, Any, Type, Optional, Literal
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.language_models.base import LanguageModelInput

STAGE = "dev"
secret_name = f"/ch/{STAGE}/service-config/commercehub-devops-chatbot"
load_secrets(secret_name)
 
api_url=os.getenv("HOST_URL")
api_key=os.getenv("API_KEY")
api_secret_key=os.getenv("API_SECRET")
 
class HMACAuthenticatedLLM(LLM):
    """A custom LLM that uses HMAC authentication to connect to a model."""
 
    api_key: str
    secret_key: str
    endpoint: str
   
    def _call(
        self,
        
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input with HMAC authentication.
 
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments.
 
        Returns:
            The model output as a string.
        """
 
 
        # Create request body
        request_body = {
            "messages": [
                {"content": "", "role": "system"},
                {"content": prompt, "role": "user"},
            ],
           # "model":"azure-gpt-3.5",
            "frequency_penalty": 0,
            "max_tokens": 9000,
            "n": 1,
            "presence_penalty": 0,
            "response_format": {"type": "text"},
            "stream": False,
            "temperature": 0,
            "top_p": 1
        }
 
        # Create HMAC signature
        timestamp = int(time.time() * 1000)
        request_id = uuid.uuid4()
        hmac_source_data = self.api_key + str(request_id) + str(timestamp) + json.dumps(request_body)
        computed_hash = hmac.new(self.secret_key.encode(), hmac_source_data.encode(), hashlib.sha256)
        hmac_signature = base64.b64encode(computed_hash.digest()).decode()
 
        # Send request to the model
        headers = {
            "api-key": self.api_key,
            "Client-Request-Id": str(request_id),
            "Timestamp": str(timestamp),
            "Authorization": hmac_signature,
            "Accept": "application/json",
        }
        response = requests.post(self.endpoint + "/chat/completions", headers=headers, json=request_body)
        response.raise_for_status()
 
        #return response.json().get('choices', [{}])[0].get('message', {})
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        #return response.json()
 
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "hmac_authenticated"
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and (len(formatted_tools) != 1):
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, dict) and (
                formatted_tools[0]["function"]["name"]
                != tool_choice["function"]["name"]
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tool was {formatted_tools[0]['function']['name']}."
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
 
 
if __name__=='__main__':
# Example usage
    llm = HMACAuthenticatedLLM(api_key=api_key, secret_key=api_secret_key, endpoint=api_url)
    print(llm.invoke("This is a test prompt. what is the capital of France."))

