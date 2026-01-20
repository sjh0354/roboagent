"""
Gemini Vision-Language Model Client
Integrates with Google Gemini APIs for visual observation and planning
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
from google import genai
from google.genai import types

class GeminiVLMClient:
    """
    Client for Google Gemini Vision-Language Models (gemini-2.0-flash-exp, etc.)

    Supports:
    - Image analysis with text queries
    - Multi-modal conversation (image + text)
    - Action verification through visual comparison
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"), verbose: bool = True):
        """
        Initialize Gemini VLM client

        Args:
            api_key: Google GenAI API key (or use GENAI_API_KEY env var)
            model_name: VLM model to use
            verbose: Print debug information
        """
        self.api_key = api_key or os.getenv("GENAI_API_KEY")
        if not self.api_key:
            # Fallback for older env var if needed, or raise
            self.api_key = os.getenv("DASHSCOPE_API_KEY") # Check if user reused this
            if not self.api_key:
                 # Just warning here, let the client init fail if it must
                 pass

        if not self.api_key:
             raise ValueError("API key not found. Set GENAI_API_KEY environment variable.")

        self.model_name = model_name
        self.verbose = verbose

        # Initialize Google GenAI client
        self.base_url = os.getenv("GENAI_BASE_URL")
        client_options = {"api_key": self.api_key}
        if self.base_url:
            if self.verbose:
                print(f"🌐 Using custom Base URL: {self.base_url}")
            # For google-genai SDK, usually http_options is the way for custom endpoints
            client_options["http_options"] = {"base_url": self.base_url}
            
        self.client = genai.Client(**client_options)

        if verbose:
            print(f"✅ GeminiVLMClient initialized with model: {model_name}")

    def _extract_content(self, response) -> str:
        """Helper to extract text and handle thoughts from response"""
        final_text = []
        
        if not response.candidates:
            return ""
            
        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content.parts:
            return ""
            
        for part in candidate.content.parts:
            # Skip empty text
            if not part.text:
                continue
                
            # Check for thought content (gemini-3-pro-preview etc)
            is_thought = False
            if hasattr(part, 'thought') and part.thought:
                is_thought = True
                
            if is_thought:
                if self.verbose:
                    print(f"\n💭 [THOUGHT]:\n{part.text}\n")
            else:
                final_text.append(part.text)
                
        return "".join(final_text)

    def create_image_message(self, image_path: str) -> Dict:
        """
        Create image message payload.
        For Gemini via google.genai, we usually pass bytes or PIL Image.
        However, to keep compatibility with the planner's message construction which expects a dict 
        (often for OpenAI format), we might need to handle this.
        
        The planner logic constructs:
        { "type": "image_url", "image_url": { "url": ... } } 
        
        If we want to minimize planner changes, we should probably return something that 
        our `analyze_image` or `chat_with_image` can understand, OR return the OpenAI format 
        and parse it back in our methods.
        
        Let's return a special dict that identifies it as a local image path, 
        which we will process in `chat` methods.
        """
        # Check if path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # For compatibility with existing planner structure which puts this in a list
        return {
            "type": "image_url", 
            "image_url": {"url": image_path}, # We store path directly to load it later
            "_internal_path": image_path # Helper key
        }

    def _load_image_content(self, image_path: str):
        """Helper to load image bytes for Gemini"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    def analyze_image(self, image_path: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Analyze single image with a query
        """
        if self.verbose:
            print(f"\n🔍 Analyzing image: {image_path}")
            print(f"📝 Query: {query}")

        try:
            image_part = self._load_image_content(image_path)
            
            contents = [image_part, query]
            
            config = types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_prompt if system_prompt else None
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            result = response.text
            if self.verbose:
                print(f"✅ VLM Response: {result[:200]}...")
            return result

        except Exception as e:
            error_msg = f"VLM analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def compare_images(self,
                      image_before: str,
                      image_after: str,
                      action_description: str) -> Dict[str, Any]:
        """
        Compare two images to verify action result
        """
        if self.verbose:
            print(f"\n🔄 Comparing images for action: {action_description}")

        query = f"""
You are verifying the result of a robot action. Compare these two images (before and after).

ACTION PERFORMED: {action_description}

Please analyze:
1. What changed between the two images?
2. Did the action succeed based on visual evidence?
3. What is the current state after the action?

Respond in JSON format:
{{
    "changes_detected": ["list of visible changes"],
    "success": true/false,
    "observation": "detailed description of current state",
    "verification": "explanation of success/failure"
}}
"""
        try:
            img1 = self._load_image_content(image_before)
            img2 = self._load_image_content(image_after)
            
            # Context: Before image, After image, Query
            contents = [
                "BEFORE IMAGE:", img1,
                "AFTER IMAGE:", img2,
                query
            ]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result_text = response.text
            
            # Parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback cleaning
                cleaned = result_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned)

            if self.verbose:
                print(f"✅ Comparison result: {result.get('verification', 'No verification text')}")

            return result

        except Exception as e:
            error_msg = f"Image comparison failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "changes_detected": [],
                "success": False,
                "observation": "Comparison error",
                "verification": error_msg
            }

    def chat_with_image(self,
                       image_path: str,
                       messages: List[Dict],
                       system_prompt: Optional[str] = None) -> str:
        """
        Multi-turn conversation with image context.
        Adapts the OpenAI-style message format used in the planner to Gemini format.
        """
        # Convert messages to Gemini format
        gemini_contents = []
        
        # Add system prompt? generate_content handles it in config, or we prepend.
        # But this method usually is called for single-turn or managing history manually.
        
        # The planner passes a list of messages. We need to construct the `contents` for generate_content.
        # Gemini stateless API uses a list of Content objects or just a list of parts.
        # Ideally, we should use chat sessions, but the planner reconstructs history every time.
        
        # We will reconstruct the whole history for the generate_content call.
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item["text"]))
                        elif item.get("type") == "image_url":
                            # Use the internal path we stored
                            path = item.get("_internal_path") or item.get("image_url", {}).get("url")
                            if path and not path.startswith("data:"):
                                parts.append(self._load_image_content(path))
            
            # Map role: user -> user, assistant -> model
            g_role = "user" if role == "user" else "model"
            gemini_contents.append(types.Content(role=g_role, parts=parts))

        # Add current image if provided separately (though usually planner adds it to messages)
        # The planner calls `plan_next_step_with_image` which builds the message list including the image.
        # So `messages` should already contain the image.
        
        # However, `chat_with_image` in Qwen client seemed to check if image was added. 
        # Here we assume the caller (planner) constructs messages correctly using `create_image_message`.

        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_prompt
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config
            )

            return response.text

        except Exception as e:
            error_msg = f"VLM chat failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def get_observation_description(self, image_path: str) -> str:
        """Get detailed description of current observation"""
        query = """
Describe this scene in detail for a robot planning system. Focus on:
- Objects present and their states
- People and their activities
- Spatial layout and relationships
- Any indicators of device states (lights, displays, etc.)
- Overall environment condition

Be specific and factual.
"""
        return self.analyze_image(image_path, query)

    def switch_model(self, model_name: str):
        """Switch to different VLM model"""
        old_model = self.model_name
        self.model_name = model_name
        if self.verbose:
            print(f"🔄 Switched VLM model: {old_model} → {model_name}")

    # OpenAI-compatible method for easy replacement in planner
    def create_chat_completion(self, messages, max_tokens=None, temperature=None, model=None):
        """
        Helper to mimic openai.chat.completions.create behavior
        """
        # Extract system prompt if present
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                filtered_messages.append(msg)
        
        # Use local model or passed model
        target_model = model or self.model_name
        
        # Convert messages
        gemini_contents = []
        for msg in filtered_messages:
            role = msg['role']
            content = msg['content']
            
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item["text"]))
                        elif item.get("type") == "image_url":
                            path = item.get("_internal_path") or item.get("image_url", {}).get("url")
                            if path and not path.startswith("data:"):
                                parts.append(self._load_image_content(path))
            
            g_role = "user" if role == "user" else "model"
            gemini_contents.append(types.Content(role=g_role, parts=parts))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt
        )

        response = self.client.models.generate_content(
            model=target_model,
            contents=gemini_contents,
            config=config
        )
        
        # Return object that mimics OpenAI response structure
        class MockMessage:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        return MockResponse(response.text)
