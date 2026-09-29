from io import BytesIO
import os
import re
import secrets
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def generate_image(
    image_prompt: str,
    panel_number: int,
    output_path: Path | None = None,
):
    """
    Generate one AI comic panel using the current Pollinations API.
    """

    width = 768
    height = 768

    api_key = os.getenv("POLLINATIONS_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "POLLINATIONS_API_KEY is missing from the .env file."
        )

    scene_description = _clean_image_prompt(image_prompt)

    prompt = f"""
Create one high-quality, standalone, colorful comic-book illustration
showing one single continuous scene for one comic panel.

Do not create a comic page, collage, multi-panel layout, split screen,
storyboard, grid, or multiple frames.

Scene description:
{scene_description}

Requirements:

- cinematic medium-shot or medium-close-up composition
- main characters large and clearly visible
- natural front-facing or three-quarter camera angles
- sharp, clean, natural, attractive faces
- realistic proportions and anatomy
- clearly defined eyes, eyebrows, nose, lips, jawline, and skin details
- every visible human face must be fully rendered and in focus
- natural facial expressions and believable emotions
- consistent character identity across the image
- detailed background but visually secondary to characters
- polished professional-quality character artwork
- clean digital illustration
- bright colorful cinematic comic-book style
- square composition

CRITICAL:
Create artwork only.

Absolutely NO visible text of any kind.

No words, letters, numbers, symbols, typography, captions,
dialogue, speech bubbles, lettering, signs, labels, logos,
watermarks, posters with writing, or readable text anywhere.

Do not put dialogue inside the artwork.
Dialogue will be added separately by the application.
"""

    url = f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}"

    params = {
        "model": "flux",
        "width": width,
        "height": height,
        "seed": secrets.randbelow(2**31),
        "nologo": "true",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    response = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=180,
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    response.raise_for_status()

                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            break

        except requests.RequestException:
            if attempt == 2:
                raise

            time.sleep(2 ** attempt)

    if response is None:
        raise RuntimeError("Pollinations did not return a response.")

    content_type = response.headers.get("content-type", "").lower()

    if not content_type.startswith("image/"):
        raise RuntimeError(
            "Pollinations returned a non-image response."
        )

    output_path = (
        output_path
        or GENERATED_DIR / f"panel-{panel_number}.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(BytesIO(response.content)) as image:
            image.convert("RGB").save(
                output_path,
                "PNG",
            )
    except Exception as exc:
        raise RuntimeError(
            "Pollinations returned data that could not be saved as an image."
        ) from exc

    return output_path


def _clean_image_prompt(image_prompt: str) -> str:
    prompt = image_prompt.replace("\r\n", "\n").replace("\r", "\n")

    prompt = re.sub(
        r"(?is)(?:Caption|Dialogue|Narration)\s*:\s*.*?(?="
        r"Panel\s+\d+|Visual\s*:|Scene\s*:|$)",
        " ",
        prompt,
    )

    prompt = re.sub(
        r"(?i)(?:\d+\.\s*)?Panel\s+\d+\b(?:\s*\([^)]*\))?\s*:?\s*",
        " ",
        prompt,
    )

    prompt = re.sub(
        r"(?i)\bVisual\s*:\s*",
        " ",
        prompt,
    )

    prompt = re.sub(
        r"(?m)^\s*[-*]+\s*",
        "",
        prompt,
    )

    prompt = re.sub(
        r"[\*_\`#]+",
        "",
        prompt,
    )

    instruction_start = re.compile(
        r"^(?:create|generate|make|do not|don't|never|use|keep|"
        r"show|ensure|avoid|include|exclude|prioritize|maintain|"
        r"no\b|important\b|every panel\b|each panel\b|"
        r"main characters? must\b|characters? must\b|"
        r"faces? must\b|the generated image\b|this image\b)",
        re.IGNORECASE,
    )

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        prompt,
    )

    prompt = " ".join(
        sentence
        for sentence in sentences
        if sentence.strip()
        and not instruction_start.match(sentence.strip())
    )

    return re.sub(
        r"\s+",
        " ",
        prompt,
    ).strip(" \t\n-:;")