from types import SimpleNamespace

from skills_gap_finder import rag

ANALYSIS = {
    "role": "data scientist",
    "covered": [{"label": "perform data cleansing", "importance": "essential", "evidence": "I clean spreadsheets"}],
    "gaps": [
        {"skill_uri": "s1", "label": "statistics", "importance": "essential"},
        {"skill_uri": "s2", "label": "data mining", "importance": "essential"},
        {"skill_uri": "s3", "label": "Hadoop", "importance": "optional"},
    ],
}


def describe(uris):
    return [{"skill_uri": u, "label": u.upper(), "description": f"Description of {u}."} for u in uris]


class FakeClient:
    """Stands in for anthropic.Anthropic so tests never call the real API."""

    def __init__(self, stop_reason="end_turn", text="## Plan"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = SimpleNamespace(
            stop_reason=stop_reason, model="claude-opus-5-5", content=[SimpleNamespace(type="text", text=text)]
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_context_keeps_only_essential_gaps_with_descriptions():
    context = rag.build_context(ANALYSIS, describe)
    assert [g["skill_uri"] for g in context["gaps"]] == ["s1", "s2"]
    assert context["gaps"][0]["description"] == "Description of s1."
    assert context["strengths"] == [{"label": "perform data cleansing", "evidence": "I clean spreadsheets"}]


def test_prompt_contains_only_retrieved_context():
    prompt = rag.build_prompt(rag.build_context(ANALYSIS, describe))
    assert "Description of s1." in prompt and "Description of s2." in prompt
    assert "Hadoop" not in prompt and "S3" not in prompt  # optional gap not retrieved
    assert 'they said: "I clean spreadsheets"' in prompt


def test_generate_plan_with_llm_sends_expected_request():
    client = FakeClient()
    result = rag.generate_plan(rag.build_context(ANALYSIS, describe), client=client)

    assert result == {"plan": "## Plan", "mode": "llm", "model": "claude-opus-5-5", "truncated": False}
    request = client.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert "Use only the skills" in request["system"]


def test_refusal_falls_back_to_offline_plan():
    result = rag.generate_plan(rag.build_context(ANALYSIS, describe), client=FakeClient(stop_reason="refusal"))
    assert result["mode"] == "offline"
    assert "Description of s1." in result["plan"]


def test_no_credentials_gives_offline_plan(monkeypatch):
    monkeypatch.setattr(rag, "has_credentials", lambda: False)
    result = rag.generate_plan(rag.build_context(ANALYSIS, describe))
    assert result["mode"] == "offline"
    assert result["reason"] == "no Anthropic credentials found"
