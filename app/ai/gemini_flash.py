import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def create_gemini_client() -> genai.Client:
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not configured.")
	return genai.Client(
		api_key=api_key,
		http_options=types.HttpOptions(
			timeout=30_000,
			retry_options=types.HttpRetryOptions(attempts=1),
		),
	)


def generate_outline(client: genai.Client, form_values: Mapping[str, str]) -> str:
	model = os.getenv("OUTLINE_MODEL", "gemini-flash-lite-latest")
	prompt = "\n".join(
		[
			"Create a concise outline for a five-panel comic with exactly five numbered panels.",
			"Include a title, a clear beginning, development, a turning point, and ending.",
			"Keep the story suitable for a general audience.",
			_format_brief(form_values),
		]
	)
	return _generate_text(
		client,
		model,
		prompt,
		"outline",
		fallback_model="gemini-flash-lite-latest",
	)


def generate_story(
	client: genai.Client,
	form_values: Mapping[str, str],
	outline: str,
) -> str:
	model = os.getenv("STORY_MODEL", "gemini-pro-latest")
	prompt = "\n\n".join(
		[
			"Write the complete story for a five-panel comic from this outline.",
			"Format it as exactly five numbered panels. For each panel, include 'Visual:' and 'Caption/Dialogue:' fields.",
			"In each Caption/Dialogue field, write exactly one natural, grammatical spoken sentence of 5-15 words; do not write narration, captions, explanations, or labels there.",
			"Make all five dialogue lines different and specific to their panel scenes and this story. Use character names when natural, and use emotional or romantic speech when it fits.",
			"Keep dialogue conversational, with normal spelling and punctuation. Avoid generic repeated lines and sound-effect spellings.",
			"Keep the complete story concise and suitable for a general audience.",
			f"Brief:\n{_format_brief(form_values)}",
			f"Outline:\n{outline}",
		]
	)
	return _generate_text(
		client,
		model,
		prompt,
		"story",
		fallback_model="gemini-3.1-pro-preview",
	)


def _format_brief(form_values: Mapping[str, str]) -> str:
	labels = {
		"prompt": "Story prompt",
		"character_name": "Character",
		"setting": "Setting",
		"tone": "Tone",
		"art_style": "Art style",
	}
	return "\n".join(
		f"{label}: {form_values[field_name]}"
		for field_name, label in labels.items()
		if form_values.get(field_name)
	)


def is_transient_gemini_error(error: Exception) -> bool:
	status_codes = (
		getattr(error, "code", None),
		getattr(error, "status_code", None),
		getattr(getattr(error, "response", None), "status_code", None),
	)
	if any(str(status) in {"429", "503", "504"} for status in status_codes):
		return True
	if any(
		cls.__name__.lower() == "servererror" or "timeout" in cls.__name__.lower()
		for cls in type(error).__mro__
	):
		return True
	message = str(error).lower()
	return "timed out" in message or "timeout" in message


def _generate_text(
	client: genai.Client,
	model: str,
	prompt: str,
	stage: str,
	fallback_model: str | None = None,
) -> str:
	try:
		response = client.models.generate_content(model=model, contents=prompt)
	except Exception as error:
		if (
			fallback_model is None
			or model == fallback_model
			or getattr(error, "code", None) != 404
		):
			raise
		response = client.models.generate_content(model=fallback_model, contents=prompt)
	text = response.text
	if not isinstance(text, str) or not text.strip():
		raise RuntimeError(f"Gemini returned an empty {stage}.")
	return text.strip()
