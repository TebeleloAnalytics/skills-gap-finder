"""Step 6 (optional): retrieval-augmented generation of a development plan.

Retrieve: the person's closest essential gaps and what they already do, with the official ESCO
          description of every skill involved.
Augment:  put only that retrieved context into the prompt.
Generate: ask Claude for a short plan that must stay within the retrieved skills.

Without Anthropic credentials the plan is assembled directly from the retrieved context instead,
so the project runs fully offline.
"""

import os
from pathlib import Path

import anthropic

from .config import LLM_MODEL

SYSTEM_PROMPT = """You write short, practical development plans for employees.
Use only the skills and descriptions given in the context. Don't invent skills, courses, certificates or URLs.
For each gap skill, write: why it matters for the target role (based on its description), one concrete
on-the-job activity that builds it, using what the person already does where possible, and what evidence
would show the skill was gained. Finish with a suggested order for the next 90 days. Use plain Markdown."""


def build_context(analysis: dict, describe, max_gaps: int = 5) -> dict:
    """Pick the gaps to plan for and attach ESCO descriptions (the retrieval step)."""
    gaps = [g for g in analysis["gaps"] if g["importance"] == "essential"][:max_gaps]
    covered = [c for c in analysis["covered"] if c["importance"] == "essential"]
    return {
        "role": analysis["role"],
        "gaps": describe([g["skill_uri"] for g in gaps]),
        "strengths": [{"label": c["label"], "evidence": c["evidence"]} for c in covered],
    }


def build_prompt(context: dict) -> str:
    strengths = (
        "\n".join(f'- {s["label"]} (they said: "{s["evidence"]}")' for s in context["strengths"]) or "- none identified"
    )
    gaps = "\n".join(f"- {g['label']}: {g['description']}" for g in context["gaps"])
    return (
        f"Target role: {context['role']}\n\n"
        f"<strengths>\n{strengths}\n</strengths>\n\n"
        f"<gap_skills>\n{gaps}\n</gap_skills>\n\n"
        "Write the development plan."
    )


def offline_plan(context: dict) -> str:
    lines = [
        f"# Development plan: {context['role']}",
        "",
        "_Generated without a language model, from ESCO descriptions only._",
        "",
    ]
    for g in context["gaps"]:
        lines += [f"## {g['label']}", g["description"], ""]
    if context["strengths"]:
        lines += ["## Build on", *(f"- {s['label']}" for s in context["strengths"])]
    return "\n".join(lines)


def has_credentials() -> bool:
    """True if the Anthropic SDK can find credentials: an API key, auth token or `ant auth login` profile."""
    env = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    return any(os.environ.get(name) for name in env) or (Path.home() / ".config" / "anthropic").exists()


def generate_plan(context: dict, client: anthropic.Anthropic | None = None) -> dict:
    """Return {"plan": markdown, "mode": "llm" | "offline", ...}."""
    if not context["gaps"]:
        return {"plan": "No essential gaps found for this role.", "mode": "offline"}
    if client is None and not has_credentials():
        return {"plan": offline_plan(context), "mode": "offline", "reason": "no Anthropic credentials found"}
    try:
        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            # if Claude declines, the API retries on a fallback model chosen for that refusal category
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(context)}],
        )
    except (anthropic.CredentialsError, anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return {"plan": offline_plan(context), "mode": "offline", "reason": "no valid Anthropic credentials"}
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as error:
        return {"plan": offline_plan(context), "mode": "offline", "reason": f"API unavailable: {type(error).__name__}"}

    if response.stop_reason == "refusal":
        return {"plan": offline_plan(context), "mode": "offline", "reason": "the model declined the request"}
    text = "".join(block.text for block in response.content if block.type == "text")
    return {"plan": text, "mode": "llm", "model": response.model, "truncated": response.stop_reason == "max_tokens"}
