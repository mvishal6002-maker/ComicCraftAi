from pathlib import Path
import json
import logging
import re
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from .services.init import OUTPUT_DIR, run_comic_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@router.post("/generate")
async def generate_comic(request: Request):
    submitted_fields = parse_qs(
        (await request.body()).decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )
    form_values = {
        field_name: submitted_fields.get(field_name, [""])[0].strip()
        for field_name in ("prompt", "character_name", "setting", "tone", "art_style")
    }
    context = {"form_values": form_values}

    if not form_values["prompt"]:
        context["error_message"] = "Add a story prompt to continue."
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=422,
        )

    try:
        result = await run_in_threadpool(run_comic_pipeline, form_values)
    except Exception as error:
        status = getattr(error, "code", None)
        if status is None:
            status = getattr(getattr(error, "response", None), "status_code", None)
        logger.error(
            "Comic generation failed (%s, provider status %s).",
            type(error).__name__,
            status,
        )
        if status == 402:
            context["error_message"] = (
                "Hugging Face image-generation credits are depleted (HTTP 402). "
                "Add credits to the account used by HF_API_KEY, then retry."
            )
        elif status == 429 and str(getattr(error, "code", None)) == "429":
            context["error_message"] = (
                "Gemini rate limit or quota reached (HTTP 429). Wait or check API quota, then retry."
            )
        elif status == 503:
            provider_message = getattr(error, "message", None)
            if not isinstance(provider_message, str) or not provider_message.strip():
                provider_message = "The Gemini service is temporarily unavailable."
            context["error_message"] = (
                f"Gemini returned HTTP 503: {provider_message[:500]}"
            )
        elif status == 404:
            context["error_message"] = (
                "The configured Gemini model is unavailable (HTTP 404). Check the model settings."
            )
        else:
            context["error_message"] = (
                "Comic generation failed. Check the server logs and try again."
            )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=503,
        )

    return RedirectResponse(
        url=f"/preview/{result['job_id']}",
        status_code=303,
    )


@router.get("/preview/{job_id}")
def comic_preview(request: Request, job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Comic preview not found")

    result_path = OUTPUT_DIR / job_id / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Comic preview not found") from None

    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="Comic preview not found")
    return templates.TemplateResponse(
        request=request,
        name="comic_preview.html",
        context={"request": request, **result},
    )


@router.get("/generated/{job_id}/{asset_name}")
def generated_asset(job_id: str, asset_name: str):
    allowed_assets = {"comic-page.png", "comic.pdf"} | {
        f"panel-{index}.png" for index in range(1, 6)
    }
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or asset_name not in allowed_assets:
        raise HTTPException(status_code=404, detail="Generated file not found")

    asset_path = OUTPUT_DIR / job_id / asset_name
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Generated file not found")

    media_type = "application/pdf" if asset_name.endswith(".pdf") else "image/png"
    filename = asset_name if asset_name.endswith(".pdf") else None
    return FileResponse(asset_path, media_type=media_type, filename=filename)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ComicCraft AI"
    }