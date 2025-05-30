import logging
from typing import Any, Dict, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM(LLM):
    """LLM wrapper for Google's Gemini API that handles both simple prompts and chat messages."""
    
    model_name: str = GEMINI_CONFIG["model_name"]
    temperature: float = GEMINI_CONFIG["temperature"]
    top_p: float = GEMINI_CONFIG["top_p"]
    top_k: int = GEMINI_CONFIG["top_k"]
    max_output_tokens: int = GEMINI_CONFIG["max_output_tokens"]
    
    def __init__(self, api_key=None, **kwargs):
        """Initialize with API key and optional parameters."""
        super().__init__(**kwargs)
        genai.configure(api_key=api_key or GEMINI_API_KEY)
        
        # Override default parameters with any provided in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        logger.info(f"Initialized GeminiChatLLM with model {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemini_chat"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API with chat format and return the output."""
        try:
            # Create generation config
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                stop_sequences=stop if stop else None
            )
            
            # Create model and generate content
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # Extract and return the generated text
            return response.text
            
        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Process a list of messages in LangChain format and return a response."""
        try:
            # Extract system message and user/assistant messages
            system_prompt = ""
            chat = []
            
            for message in messages:
                if isinstance(message, SystemMessage):
                    # For system messages, add to system prompt
                    system_prompt += message.content + "\n"
                elif isinstance(message, HumanMessage):
                    # For human messages, add as a user message
                    chat.append({"role": "user", "parts": [message.content]})
                elif isinstance(message, AIMessage):
                    # For AI messages, add as a model message
                    chat.append({"role": "model", "parts": [message.content]})
                elif isinstance(message, dict):
                    # Handle dictionary format
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "system":
                        system_prompt += content + "\n"
                    elif role == "user":
                        chat.append({"role": "user", "parts": [content]})
                    elif role in ["assistant", "ai"]:
                        chat.append({"role": "model", "parts": [content]})
            
            # Create generation config
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                stop_sequences=stop if stop else None
            )
            
            # Create model with system prompt if available
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt if system_prompt else None
            )
            
            # Generate content with chat history if available, otherwise use the last user message
            if chat:
                response = model.generate_content(chat, generation_config=generation_config)
            else:
                # Fallback to using the last user message if no chat history
                last_user_msg = ""
                for message in reversed(messages):
                    if isinstance(message, HumanMessage):
                        last_user_msg = message.content
                        break
                    elif isinstance(message, dict) and message.get('role') == 'user':
                        last_user_msg = message.get('content', '')
                        break
                        
                response = model.generate_content(last_user_msg, generation_config=generation_config)
            
            # Extract and return the generated text
            return response.text
            
        except Exception as e:
            logger.error(f"Error in Gemini chat API call: {str(e)}")
            return f"Error generating response: {str(e)}"
