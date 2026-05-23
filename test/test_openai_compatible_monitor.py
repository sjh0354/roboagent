import json

from utils.gemini_vlm_client import GeminiVLMClient


def test_openai_compatible_monitor_is_selected_by_base_url(monkeypatch):
    client = object.__new__(GeminiVLMClient)

    monkeypatch.setenv("PI0_ACTION_MONITOR_BASE_URL", "https://api.sharesai.xyz/v1")

    assert client._should_use_openai_compatible_monitor("gpt-5.5") is True


def test_extract_json_object_strips_markdown_fence():
    client = object.__new__(GeminiVLMClient)

    payload = {"decision": "complete", "recommended_stop": True}
    text = "```json\n" + json.dumps(payload) + "\n```"

    assert client._extract_json_object(text) == payload


def test_openai_compatible_monitor_builds_multimodal_request(monkeypatch, tmp_path):
    start = tmp_path / "start.jpg"
    current = tmp_path / "current.jpg"
    start.write_bytes(b"fake-start")
    current.write_bytes(b"fake-current")

    captured = {}

    class _FakeChoices:
        message = type(
            "Message",
            (),
            {
                "content": json.dumps(
                    {
                        "decision": "complete",
                        "task_complete": True,
                        "stuck_or_ood": False,
                        "confidence": "high",
                        "source_object_state": "removed_from_source",
                        "destination_state": "object_visible_at_destination",
                        "arm_motion_assessment": "hovering_or_jittering",
                        "visual_evidence": ["object appears in basket"],
                        "recommended_stop": True,
                        "reason": "done",
                    }
                )
            },
        )()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [_FakeChoices()]})()

    class _FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr("utils.gemini_vlm_client.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("PI0_ACTION_MONITOR_API_KEY", "test-key")
    monkeypatch.setenv("PI0_ACTION_MONITOR_BASE_URL", "https://api.sharesai.xyz/v1")

    client = object.__new__(GeminiVLMClient)
    result = client._assess_long_action_progress_openai_compatible(
        model="gpt-5.5",
        query="Return JSON.",
        start_image=str(start),
        current_image=str(current),
        process_images=[],
    )

    assert result["decision"] == "complete"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.sharesai.xyz/v1"
    assert captured["model"] == "gpt-5.5"
    assert captured["stream"] is False
    assert captured["messages"][0]["content"][1]["type"] == "image_url"


def test_openai_compatible_monitor_falls_back_to_urllib(monkeypatch, tmp_path):
    start = tmp_path / "start.jpg"
    current = tmp_path / "current.jpg"
    start.write_bytes(b"fake-start")
    current.write_bytes(b"fake-current")

    captured = {}

    class _FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "decision": "stuck",
                                        "recommended_stop": True,
                                        "reason": "no progress",
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse()

    monkeypatch.setattr("utils.gemini_vlm_client.OpenAI", None)
    monkeypatch.setattr("utils.gemini_vlm_client.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("PI0_ACTION_MONITOR_API_KEY", "test-key")
    monkeypatch.setenv("PI0_ACTION_MONITOR_BASE_URL", "https://api.sharesai.xyz/v1")

    client = object.__new__(GeminiVLMClient)
    result = client._assess_long_action_progress_openai_compatible(
        model="gpt-5.5",
        query="Return JSON.",
        start_image=str(start),
        current_image=str(current),
        process_images=[],
    )

    assert result["decision"] == "stuck"
    assert captured["url"] == "https://api.sharesai.xyz/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["messages"][0]["content"][1]["type"] == "image_url"
    assert captured["timeout"] == 60.0


def test_openai_compatible_planner_chat_uses_local_image_data_url(monkeypatch, tmp_path):
    image = tmp_path / "observation.jpg"
    image.write_bytes(b"fake-image")

    captured = {}

    class _FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "current_step_analysis": {
                                            "visual_state": "ok",
                                            "task_progress": "testing",
                                            "next_action_reasoning": "done",
                                        },
                                        "next_step": None,
                                        "needs_human_input": False,
                                        "user_question": None,
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse()

    monkeypatch.setattr("utils.gemini_vlm_client.OpenAI", None)
    monkeypatch.setattr("utils.gemini_vlm_client.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("PLANNER_VLM_API_KEY", "planner-key")
    monkeypatch.setenv("PLANNER_VLM_BASE_URL", "https://api.sharesai.xyz/v1")

    client = object.__new__(GeminiVLMClient)
    response = client.create_chat_completion(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(image)}, "_internal_path": str(image)},
                    {"type": "text", "text": "Plan next step."},
                ],
            },
        ],
        max_tokens=100,
        temperature=0,
        response_mime_type="application/json",
    )

    content = response.choices[0].message.content
    assert json.loads(content)["next_step"] is None
    assert captured["url"] == "https://api.sharesai.xyz/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    user_content = captured["payload"]["messages"][1]["content"]
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_analyze_image_uses_openai_compatible_path(monkeypatch, tmp_path):
    image = tmp_path / "observation.jpg"
    image.write_bytes(b"fake-image")

    captured = {}

    class _FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "scene description"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse()

    monkeypatch.setattr("utils.gemini_vlm_client.OpenAI", None)
    monkeypatch.setattr("utils.gemini_vlm_client.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("PLANNER_VLM_API_KEY", "planner-key")
    monkeypatch.setenv("PLANNER_VLM_BASE_URL", "https://api.sharesai.xyz/v1")

    client = object.__new__(GeminiVLMClient)
    client.model_name = "gpt-5.5"
    client.verbose = False

    result = client.analyze_image(str(image), "Describe this scene.")

    assert result == "scene description"
    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["messages"][0]["content"][0]["type"] == "image_url"


def test_planner_access_error_falls_back_to_configured_model(monkeypatch):
    client = object.__new__(GeminiVLMClient)
    client.model_name = "gpt-5.5"
    client.verbose = False
    calls = []

    def fake_openai_completion(**kwargs):
        calls.append(kwargs["model"])
        raise __import__("utils.gemini_vlm_client", fromlist=["VLMAccessError"]).VLMAccessError("no access")

    def fake_gemini_response(**kwargs):
        calls.append(kwargs["model"])
        return client._mock_chat_completion_response('{"next_step": null}')

    monkeypatch.setattr(client, "_openai_compatible_chat_completion", fake_openai_completion)
    monkeypatch.setattr(client, "_should_use_openai_compatible_planner", lambda model: model.startswith("gpt-"))
    monkeypatch.setattr(client, "_load_image_content", lambda path: None)
    monkeypatch.setenv("PLANNER_VLM_FALLBACK_MODEL", "gemini-2.5-flash-lite")

    class _FakeModels:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": '{"next_step": null}'})()

    client.client = type("Client", (), {"models": _FakeModels()})()

    response = client.create_chat_completion(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Return JSON."}],
        response_mime_type="application/json",
    )

    assert response.choices[0].message.content == '{"next_step": null}'
    assert calls == ["gpt-5.5"]
