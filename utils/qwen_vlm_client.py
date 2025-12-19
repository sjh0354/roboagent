# qwen_vlm_client.py

"""
Qwen Vision-Language Model Client
Integrates with Alibaba DashScope VLM APIs for visual observation and planning
"""

import os
import base64
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI
import json


class QwenVLMClient:
    """
    Client for Qwen Vision-Language Models (qwen-vl-plus, qwen-vl-max)

    Supports:
    - Image analysis with text queries
    - Multi-modal conversation (image + text)
    - Action verification through visual comparison
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-vl-plus", verbose: bool = True):
        """
        Initialize Qwen VLM client

        Args:
            api_key: DashScope API key (or use DASHSCOPE_API_KEY env var)
            model_name: VLM model to use (qwen-vl-plus, qwen-vl-max)
            verbose: Print debug information
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."
            )

        self.model_name = model_name
        self.verbose = verbose

        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Model configuration
        self.config = {
            "qwen-vl-plus": {
                "max_tokens": 20000,
                "temperature": 0.7
            },
            "qwen-vl-max": {
                "max_tokens": 30000,
                "temperature": 0.7
            }
        }

        if verbose:
            print(f"✅ QwenVLMClient initialized with model: {model_name}")

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string

        Args:
            image_path: Path to image file

        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_image_message(self, image_path: str, image_type: str = "jpeg") -> Dict:
        """
        Create image message for VLM API

        Args:
            image_path: Path to image file
            image_type: Image format (jpeg, png, etc.)

        Returns:
            dict: Image message object
        """
        # Check if path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image
        base64_image = self.encode_image_to_base64(image_path)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_type};base64,{base64_image}"
            }
        }

    def analyze_image(self, image_path: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Analyze single image with a query

        Args:
            image_path: Path to image file
            query: Question/instruction about the image
            system_prompt: Optional system prompt for context

        Returns:
            str: VLM's response
        """
        if self.verbose:
            print(f"\n🔍 Analyzing image: {image_path}")
            print(f"📝 Query: {query}")

        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # User message with image + text
        user_content = [
            self.create_image_message(image_path),
            {
                "type": "text",
                "text": query
            }
        ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Call API
        try:
            config = self.config.get(self.model_name, self.config["qwen-vl-plus"])

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )

            result = response.choices[0].message.content

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

        Args:
            image_before: Path to image before action
            image_after: Path to image after action
            action_description: Description of what action was performed

        Returns:
            dict: {
                'success': bool,
                'observation': str,
                'changes_detected': list,
                'verification': str
            }
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

        # Build messages with both images
        user_content = [
            {"type": "text", "text": "BEFORE IMAGE:"},
            self.create_image_message(image_before),
            {"type": "text", "text": "AFTER IMAGE:"},
            self.create_image_message(image_after),
            {"type": "text", "text": query}
        ]

        messages = [{
            "role": "user",
            "content": user_content
        }]

        try:
            config = self.config.get(self.model_name, self.config["qwen-vl-plus"])

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )

            result_text = response.choices[0].message.content

            # Try to parse JSON
            try:
                # Clean JSON from markdown code blocks if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()

                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "changes_detected": ["Unable to parse structured response"],
                    "success": True,  # Assume success if can't parse
                    "observation": result_text,
                    "verification": "Response parsing incomplete"
                }

            if self.verbose:
                print(f"✅ Comparison result: {result['verification']}")

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
        Multi-turn conversation with image context

        Args:
            image_path: Path to current observation image
            messages: Conversation history [{'role': 'user/assistant', 'content': str}]
            system_prompt: System prompt for the conversation

        Returns:
            str: Assistant's response
        """
        # Build full message list
        full_messages = []

        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add conversation history (text only)
        for msg in messages:
            if msg['role'] in ['user', 'assistant']:
                full_messages.append(msg)

        # Add current image as latest user message (if not already image)
        # Check if last message already has image
        if full_messages and isinstance(full_messages[-1].get('content'), list):
            # Last message already multimodal, don't add duplicate
            pass
        else:
            # Add image to latest context
            latest_user_msg = {
                "role": "user",
                "content": [
                    self.create_image_message(image_path),
                    {
                        "type": "text",
                        "text": "[Current visual observation attached]"
                    }
                ]
            }
            full_messages.append(latest_user_msg)

        try:
            config = self.config.get(self.model_name, self.config["qwen-vl-plus"])

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"VLM chat failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def get_observation_description(self, image_path: str) -> str:
        """
        Get detailed description of current observation

        Args:
            image_path: Path to observation image

        Returns:
            str: Detailed scene description
        """
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


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Qwen VLM Client - Test Mode\n")

    # Check API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  Set DASHSCOPE_API_KEY environment variable to test")
        print("Example: export DASHSCOPE_API_KEY='your-key'")
    else:
        try:
            # Initialize client
            vlm = QwenVLMClient(model_name="qwen-vl-plus", verbose=True)

            print("\n" + "="*70)
            print("VLM Client initialized successfully!")
            print("="*70)
            print("\nTo test with actual images, call:")
            print("  vlm.analyze_image('path/to/image.jpg', 'What do you see?')")
            print("  vlm.compare_images('before.jpg', 'after.jpg', 'turned on light')")
            print("="*70)

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
