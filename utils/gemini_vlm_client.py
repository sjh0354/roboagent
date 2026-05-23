"""
Gemini Vision-Language Model Client
Integrates with Google Gemini APIs for visual observation and planning
"""

import os
import json
import base64
import urllib.error
import urllib.request
from typing import Dict, List, Any, Optional, Union
from google import genai
from google.genai import types

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class VLMAccessError(RuntimeError):
    """Raised when a configured VLM token cannot access the requested model."""

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

    def perform_web_search(self, query: str) -> str:
        """
        Perform a web search using Gemini's grounding capabilities.
        
        Args:
            query: The search query.
            
        Returns:
            str: A summary of the search results provided by Gemini.
        """
        if self.verbose:
            print(f"\n🔍 Executing Web Search via Gemini: '{query}'")
            
        try:
            # Configure the tool for Google Search
            tools = [types.Tool(google_search=types.GoogleSearch())]
            
            # Create a prompt that asks for the search
            prompt = f"Please search the web for '{query}' and provide a concise summary of the key information found."
            
            config = types.GenerateContentConfig(
                tools=tools,
                temperature=0.7
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text result
            # For grounded responses, response.text usually contains the answer
            result = response.text
            
            if self.verbose:
                print(f"✅ Search Result: {result[:200]}...")
                
            return result
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return f"Error performing web search: {str(e)}"

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

    def _encode_image_data_url(self, image_path: str, image_type: str = "jpeg") -> str:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/{image_type};base64,{encoded}"

    def _extract_json_object(self, result_text: str) -> Dict[str, Any]:
        text = result_text or ""
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)

    def analyze_image(self, image_path: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Analyze single image with a query
        """
        if self.verbose:
            print(f"\n🔍 Analyzing image: {image_path}")
            print(f"📝 Query: {query}")

        try:
            if self._should_use_openai_compatible_planner(self.model_name):
                try:
                    return self._analyze_image_openai_compatible(
                        image_path=image_path,
                        query=query,
                        system_prompt=system_prompt,
                        model=self.model_name,
                    )
                except VLMAccessError:
                    fallback_model = os.getenv("PLANNER_VLM_FALLBACK_MODEL", "").strip()
                    if not fallback_model or fallback_model == self.model_name:
                        raise
                    if self.verbose:
                        print(f"⚠️  Image analysis model access denied for {self.model_name}; falling back to {fallback_model}")
                    old_model = self.model_name
                    try:
                        self.model_name = fallback_model
                        return self.analyze_image(image_path, query, system_prompt=system_prompt)
                    finally:
                        self.model_name = old_model

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

    def _analyze_image_openai_compatible(
        self,
        *,
        image_path: str,
        query: str,
        system_prompt: Optional[str],
        model: str,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}, "_internal_path": image_path},
                    {"type": "text", "text": query},
                ],
            }
        )
        response = self._openai_compatible_chat_completion(
            model=model,
            messages=messages,
            max_tokens=int(os.getenv("PLANNER_VLM_ANALYZE_MAX_TOKENS", "1200")),
            temperature=float(os.getenv("PLANNER_VLM_ANALYZE_TEMPERATURE", "0")),
        )
        result = response.choices[0].message.content
        if self.verbose:
            print(f"✅ VLM Response: {result[:200]}...")
        return result

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

    def analyze_action_visual_delta(
        self,
        *,
        before_image: str,
        after_image: str,
        action_name: str,
        action_parameters: Dict[str, Any],
        process_images: Optional[List[str]] = None,
        include_action_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Produce detailed visual feedback from before/process/after images.

        By default this call is action-blind: the model is not told what the robot
        intended to do, so the result is less likely to be biased toward command
        completion. Set include_action_context=True only for debugging old behavior.
        This is intended for planner state updates, not full scene description.
        """
        if self.verbose:
            context_note = f"action context: {action_name}" if include_action_context else "action-blind"
            print(f"\n🔄 Analyzing visual delta ({context_note})")

        if include_action_context:
            action_context = (
                "Requested action:\n"
                f"{json.dumps({'action': action_name, 'parameters': action_parameters}, ensure_ascii=False)}\n\n"
                "If the observed change contradicts the requested action parameters, report that explicitly.\n"
            )
        else:
            action_context = (
                "You are not given the robot's intended command. Do not infer what should have happened.\n"
                "If an object appears unchanged, say it remains unchanged. If the images are ambiguous, say so.\n"
            )

        query = f"""
You are an independent visual differencing model for a robot planner.

{action_context}
Compare the images in temporal order. Focus on visible changes between the images.
Try to enumerate all scene changes that could matter to the next robot planning step, including:
- object presence, absence, position, orientation, and containment changes
- objects that remained in their original place despite nearby motion
- gripper or arm pose changes, if visible
- occlusions, blur, lighting/camera viewpoint shifts, or other uncertainty sources

Do not describe the whole scene from scratch. Do not assume an action succeeded. Do not use prior task context.
Prefer concrete visual evidence over short generic summaries. If there is no meaningful visible change, say that explicitly.

Return JSON only with this schema:
{{
  "observed_changes": ["detailed visible changes, one fact per item"],
  "objects_moved_or_removed": [
    {{"object": "name or description", "from": "observed source", "to": "observed destination or unknown", "evidence": "specific visual evidence"}}
  ],
  "objects_remaining": ["relevant objects still visible, including their observed locations"],
  "gripper_or_arm_state": "visible state and pose change if relevant, otherwise unknown",
  "scene_change_notes": ["camera, lighting, occlusion, blur, or ambiguity notes"],
  "action_context_used": {str(bool(include_action_context)).lower()},
  "contradicts_requested_action": false,
  "state_update": "planner-facing summary covering the key visible changes and unchanged relevant objects",
  "confidence": "low|medium|high"
}}
"""
        try:
            contents = ["BEFORE IMAGE:", self._load_image_content(before_image)]
            for index, image_path in enumerate(process_images or [], start=1):
                if image_path and os.path.exists(image_path):
                    contents.extend([f"PROCESS IMAGE {index}:", self._load_image_content(image_path)])
            contents.extend(["AFTER IMAGE:", self._load_image_content(after_image), query])

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            result_text = response.text or ""
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                cleaned = result_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned)

            if self.verbose:
                print(f"✅ Visual delta: {result.get('state_update', 'No state update')}")
            result["action_context_used"] = bool(include_action_context)
            if not include_action_context:
                result["contradicts_requested_action"] = False
            return result

        except Exception as e:
            error_msg = f"Visual delta analysis failed: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "observed_changes": [],
                "objects_moved_or_removed": [],
                "objects_remaining": [],
                "gripper_or_arm_state": "unknown",
                "scene_change_notes": [],
                "action_context_used": bool(include_action_context),
                "contradicts_requested_action": False,
                "state_update": error_msg,
                "confidence": "low",
                "error": error_msg,
            }

    def assess_long_action_progress(
        self,
        *,
        instruction: str,
        start_image: str,
        current_image: str,
        process_images: Optional[List[Any]] = None,
        elapsed_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Monitor a long-running robot action and decide whether it should continue.

        This is an execution monitor, not a planner. It receives only the local action
        instruction and transient images so it can stop continuous policies that have
        completed, stalled, or entered an unsafe/irrelevant behavior.
        """
        if self.verbose:
            print(f"\n🔎 Assessing long action progress after {elapsed_seconds:.1f}s")

        query = f"""
You are a visual execution monitor for a long-running robot manipulation policy.
Your job is to decide whether to STOP the continuous controller now, not to produce a full task plan.
This controller may never emit its own terminal signal, and after success it may hover or jitter near the destination.

Instruction being executed:
{instruction}

Compare the images in temporal order. Decide whether the robot should keep executing or be stopped now.

Use a stop-biased policy for late-stage execution.
The images are an ordered temporal sequence. Do not treat them as independent static frames.
Read them strictly in order, and use the labels/order markers to infer the motion trend across time.

- Start image = earliest state
- Process images = ordered intermediate states
- Current image = latest state

Use a stop-biased policy for late-stage execution:
- Return "complete" when the requested object appears placed at the intended destination, is no longer at the source, appears inside/near the destination container, or the useful manipulation goal is visibly achieved.
- Return "complete" when the object seems delivered and the arm is hovering, jittering, or making repeated micro-motions near the basket/destination.
- Return "stuck" when the robot appears to be making no useful progress, repeatedly approaching without grasping, oscillating, moving the wrong object, disturbing unrelated objects, or operating in an out-of-distribution/unsafe state.
- Return "continue" only when the target object is still clearly at the source AND there is visible useful progress toward grasping or transporting it.

Elapsed time is {round(float(elapsed_seconds), 2)} seconds. If elapsed time is already long and the evidence is ambiguous, prefer stopping with "complete" if the object may have been delivered, or "stuck" if there is no useful progress. Do not keep returning "continue" just because you cannot perfectly verify the final object pose.

Return JSON only with this schema:
{{
  "decision": "continue|complete|stuck",
  "task_complete": false,
  "stuck_or_ood": false,
  "confidence": "low|medium|high",
  "source_object_state": "still_at_source|removed_from_source|uncertain",
  "destination_state": "object_visible_at_destination|destination_occluded|not_at_destination|uncertain",
  "arm_motion_assessment": "useful_progress|hovering_or_jittering|no_progress|uncertain",
  "visual_evidence": ["specific visual evidence"],
  "recommended_stop": false,
  "reason": "brief execution-monitor reasoning",
  "elapsed_seconds": {round(float(elapsed_seconds), 2)}
}}
"""
        try:
            contents = ["START IMAGE (earliest):", self._load_image_content(start_image)]
            for index, frame in enumerate(process_images or [], start=1):
                image_path = frame.get("image_path") if isinstance(frame, dict) else frame
                label = frame.get("label") if isinstance(frame, dict) else None
                order = frame.get("order") if isinstance(frame, dict) else index
                if image_path and os.path.exists(image_path):
                    header = f"PROCESS IMAGE {order}"
                    if label:
                        header += f" [{label}]"
                    contents.extend([f"{header}:", self._load_image_content(image_path)])
            contents.extend(["CURRENT IMAGE (latest):", self._load_image_content(current_image), query])

            monitor_model = os.getenv("PI0_ACTION_MONITOR_MODEL", self.model_name)
            if self._should_use_openai_compatible_monitor(monitor_model):
                result = self._assess_long_action_progress_openai_compatible(
                    model=monitor_model,
                    query=query,
                    start_image=start_image,
                    current_image=current_image,
                    process_images=process_images or [],
                )
            else:
                response = self.client.models.generate_content(
                    model=monitor_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )

                result = self._extract_json_object(response.text or "")

            decision = str(result.get("decision") or "").strip().lower()
            if decision not in {"continue", "complete", "stuck"}:
                decision = "continue"
            result["decision"] = decision
            result["task_complete"] = bool(result.get("task_complete") or decision == "complete")
            result["stuck_or_ood"] = bool(result.get("stuck_or_ood") or decision == "stuck")
            result["recommended_stop"] = bool(result.get("recommended_stop") or decision in {"complete", "stuck"})
            result.setdefault("source_object_state", "uncertain")
            result.setdefault("destination_state", "uncertain")
            result.setdefault("arm_motion_assessment", "uncertain")
            result["elapsed_seconds"] = round(float(elapsed_seconds), 2)

            if self.verbose:
                print(f"✅ Long action monitor: {decision} - {result.get('reason', '')}")
            return result

        except Exception as e:
            error_msg = f"Long action progress assessment failed: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "decision": "continue",
                "task_complete": False,
                "stuck_or_ood": False,
                "confidence": "low",
                "source_object_state": "uncertain",
                "destination_state": "uncertain",
                "arm_motion_assessment": "uncertain",
                "visual_evidence": [],
                "recommended_stop": False,
                "reason": error_msg,
                "elapsed_seconds": round(float(elapsed_seconds), 2),
                "error": error_msg,
            }

    def _should_use_openai_compatible_monitor(self, model: str) -> bool:
        if os.getenv("PI0_ACTION_MONITOR_PROVIDER", "").strip().lower() in {"openai", "openai_compatible", "sharesai"}:
            return True
        if os.getenv("PI0_ACTION_MONITOR_BASE_URL"):
            return True
        lowered = (model or "").lower()
        return lowered.startswith("gpt-")

    def _should_use_openai_compatible_planner(self, model: str) -> bool:
        if os.getenv("PLANNER_VLM_PROVIDER", "").strip().lower() in {"openai", "openai_compatible", "sharesai"}:
            return True
        if os.getenv("PLANNER_VLM_BASE_URL"):
            return True
        lowered = (model or "").lower()
        return lowered.startswith("gpt-")

    def _assess_long_action_progress_openai_compatible(
        self,
        *,
        model: str,
        query: str,
        start_image: str,
        current_image: str,
        process_images: List[Any],
    ) -> Dict[str, Any]:
        api_key = (
            os.getenv("PI0_ACTION_MONITOR_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SHARESAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "PI0 action monitor API key not found. Set PI0_ACTION_MONITOR_API_KEY, OPENAI_API_KEY, or SHARESAI_API_KEY."
            )

        base_url = os.getenv("PI0_ACTION_MONITOR_BASE_URL", "https://api.sharesai.xyz/v1")

        user_content = [
            {"type": "text", "text": "START IMAGE (earliest):"},
            {"type": "image_url", "image_url": {"url": self._encode_image_data_url(start_image)}},
        ]
        for index, frame in enumerate(process_images, start=1):
            image_path = frame.get("image_path") if isinstance(frame, dict) else frame
            label = frame.get("label") if isinstance(frame, dict) else None
            order = frame.get("order") if isinstance(frame, dict) else index
            if image_path and os.path.exists(image_path):
                user_content.extend([
                    {"type": "text", "text": f"PROCESS IMAGE {order}" + (f" [{label}]" if label else "") + ":"},
                    {"type": "image_url", "image_url": {"url": self._encode_image_data_url(image_path)}},
                ])
        user_content.extend([
            {"type": "text", "text": "CURRENT IMAGE (latest):"},
            {"type": "image_url", "image_url": {"url": self._encode_image_data_url(current_image)}},
            {"type": "text", "text": query},
        ])

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": int(os.getenv("PI0_ACTION_MONITOR_MAX_TOKENS", "1200")),
            "temperature": float(os.getenv("PI0_ACTION_MONITOR_TEMPERATURE", "0")),
            "stream": False,
        }

        if OpenAI is not None and os.getenv("PI0_ACTION_MONITOR_USE_URLOPEN", "0") != "1":
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(**payload)
            return self._extract_json_object(response.choices[0].message.content or "")

        endpoint = base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        timeout_seconds = float(os.getenv("PI0_ACTION_MONITOR_HTTP_TIMEOUT_SECONDS", "60"))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI-compatible monitor HTTP {error.code}: {body}") from error

        content = response_payload["choices"][0]["message"]["content"]
        return self._extract_json_object(content or "")

    def _openai_compatible_api_key(self, prefix: str = "PLANNER_VLM") -> str:
        api_key = (
            os.getenv(f"{prefix}_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SHARESAI_API_KEY")
            or os.getenv("PI0_ACTION_MONITOR_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                f"{prefix} API key not found. Set {prefix}_API_KEY, OPENAI_API_KEY, SHARESAI_API_KEY, "
                "or PI0_ACTION_MONITOR_API_KEY."
            )
        return api_key

    def _openai_compatible_chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens=None,
        temperature=None,
        response_mime_type=None,
    ):
        api_key = self._openai_compatible_api_key("PLANNER_VLM")
        base_url = os.getenv("PLANNER_VLM_BASE_URL", os.getenv("PI0_ACTION_MONITOR_BASE_URL", "https://api.sharesai.xyz/v1"))
        converted_messages = self._convert_messages_for_openai_compatible(messages)
        payload = {
            "model": model,
            "messages": converted_messages,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if response_mime_type == "application/json" and os.getenv("PLANNER_VLM_RESPONSE_FORMAT", "1") != "0":
            payload["response_format"] = {"type": "json_object"}

        if OpenAI is not None and os.getenv("PLANNER_VLM_USE_URLOPEN", "0") != "1":
            client = OpenAI(api_key=api_key, base_url=base_url)
            try:
                response = client.chat.completions.create(**payload)
            except Exception as error:
                if self._is_access_error_text(str(error)):
                    raise VLMAccessError(str(error)) from error
                raise
            return response

        endpoint = base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        timeout_seconds = float(os.getenv("PLANNER_VLM_HTTP_TIMEOUT_SECONDS", "120"))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="ignore")
            message = f"OpenAI-compatible planner HTTP {error.code}: {body}"
            if error.code in {401, 403} or self._is_access_error_text(body):
                raise VLMAccessError(message) from error
            raise RuntimeError(message) from error

        content = response_payload["choices"][0]["message"]["content"]
        return self._mock_chat_completion_response(content or "")

    def _is_access_error_text(self, text: str) -> bool:
        lowered = (text or "").lower()
        access_markers = (
            "no access",
            "permission",
            "forbidden",
            "unauthorized",
            "does not have access",
            "has no access",
        )
        return any(marker in lowered for marker in access_markers)

    def _convert_messages_for_openai_compatible(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                converted.append({"role": role, "content": str(content)})
                continue

            converted_content = []
            for item in content:
                if not isinstance(item, dict):
                    converted_content.append({"type": "text", "text": str(item)})
                    continue
                if item.get("type") == "text":
                    converted_content.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    path = item.get("_internal_path") or image_url
                    if path and not str(path).startswith("data:") and os.path.exists(path):
                        image_url = self._encode_image_data_url(path)
                    converted_content.append({"type": "image_url", "image_url": {"url": image_url}})
            converted.append({"role": role, "content": converted_content})
        return converted

    def _mock_chat_completion_response(self, content: str):
        class MockMessage:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        return MockResponse(content)

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
    def create_chat_completion(
        self,
        messages,
        max_tokens=None,
        temperature=None,
        model=None,
        response_mime_type=None,
    ):
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

        if self._should_use_openai_compatible_planner(target_model):
            try:
                return self._openai_compatible_chat_completion(
                    model=target_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type=response_mime_type,
                )
            except VLMAccessError:
                fallback_model = os.getenv("PLANNER_VLM_FALLBACK_MODEL", "").strip()
                if not fallback_model or fallback_model == target_model:
                    raise
                if self.verbose:
                    print(f"⚠️  Planner model access denied for {target_model}; falling back to {fallback_model}")
                return self.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=fallback_model,
                    response_mime_type=response_mime_type,
                )
        
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
            system_instruction=system_prompt,
            response_mime_type=response_mime_type,
        )

        response = self.client.models.generate_content(
            model=target_model,
            contents=gemini_contents,
            config=config
        )
        
        return self._mock_chat_completion_response(response.text)
