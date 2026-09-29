import json
import os
import re
import textwrap
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..ai.gemini_flash import (
	create_gemini_client,
	generate_outline,
	generate_story,
	is_transient_gemini_error,
)
from .image_generator import generate_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "generated"
PANEL_COUNT = 5


def run_comic_pipeline(form_values: Mapping[str, str]) -> dict[str, object]:
	client = create_gemini_client()
	fallback_panels = None
	try:
		outline = generate_outline(client, form_values)
		story = generate_story(client, form_values, outline)
	except Exception as error:
		if getattr(error, "code", None) != 429 and not is_transient_gemini_error(error):
			raise
		# TEMPORARY RELIABILITY FALLBACK: keep transient Gemini outages from stopping a comic.
		outline, fallback_panels = _create_fallback_comic(form_values)
		story = "\n".join(
			f"Panel {panel['panel_number']}: {panel['title']}\n"
			f"Visual: {panel['scene_description']}\n"
			f"Caption/Dialogue: {panel['dialogue']}"
			for panel in fallback_panels
		)
		story = f"Tone: {form_values.get('tone') or 'adventurous'}\n{story}"

	if fallback_panels is not None:
		captions = [panel["image_prompt"] for panel in fallback_panels]
		dialogues = [str(panel["dialogue"]) for panel in fallback_panels]
		fallback_dialogues = dialogues
	else:
		parsed_panels = [_parse_story_panel(story, index) for index in range(1, PANEL_COUNT + 1)]
		story_chunks = _split_story_into_panels(story)
		image_style = (
			f"{form_values.get('art_style') or 'Colorful comic-book art'}; "
			f"{form_values.get('tone') or 'adventurous'} tone"
		)
		captions = [
			f"{image_style}; {visual or story_chunks[index]}"
			for index, (_, visual, _) in enumerate(parsed_panels)
		]
		dialogues = [dialogue for _, _, dialogue in parsed_panels]
		_, local_fallback_panels = _create_fallback_comic(form_values)
		fallback_dialogues = [str(panel["dialogue"]) for panel in local_fallback_panels]

	validated_dialogues = []
	seen_dialogues = set()
	for index, dialogue in enumerate(dialogues):
		clean_dialogue = _normalize_panel_dialogue(dialogue, fallback_dialogues[index])
		if clean_dialogue.casefold() in seen_dialogues:
			clean_dialogue = next(
				candidate
				for candidate in fallback_dialogues
				if candidate.casefold() not in seen_dialogues
			)
		seen_dialogues.add(clean_dialogue.casefold())
		validated_dialogues.append(clean_dialogue)
	dialogues = validated_dialogues

	job_id = uuid.uuid4().hex
	job_dir = OUTPUT_DIR / job_id
	job_dir.mkdir(parents=True, exist_ok=False)

	panel_paths = generate_panel_images(captions, job_dir)
	for panel_path, dialogue in zip(panel_paths, dialogues):
		_render_panel_dialogue(panel_path, dialogue)
	page_path = build_comic_layout(
		panel_paths,
		form_values.get("character_name") or "Untitled Comic",
		job_dir / "comic-page.png",
	)
	export_pdf(
		panel_paths,
		job_dir / "comic.pdf",
		full_story=story,
		panels=fallback_panels or (),
		captions=captions,
	)
	asset_root = f"/generated/{job_id}"

	result = {
		"job_id": job_id,
		"title": form_values.get("character_name") or "Untitled Comic",
		"outline": outline,
		"story": story,
		"panel_count": len(captions),
		"writing_provider": "local fallback" if fallback_panels is not None else "Gemini",
		"image_provider": os.getenv("IMAGE_PROVIDER", "huggingface").lower(),
		"comic_page_url": f"{asset_root}/comic-page.png",
		"pdf_url": f"{asset_root}/comic.pdf",
		"panel_images": [
			{
				"url": f"{asset_root}/panel-{index}.png",
				"caption": caption,
			}
			for index, caption in enumerate(captions, start=1)
		],
	}
	if fallback_panels is not None:
		result["panels"] = fallback_panels
	(job_dir / "result.json").write_text(
		json.dumps(result, ensure_ascii=False),
		encoding="utf-8",
	)
	return result


def _create_fallback_comic(
	form_values: Mapping[str, str],
) -> tuple[str, list[dict[str, str | int]]]:
	character = form_values.get("character_name") or "the main character"
	setting = form_values.get("setting") or "a vivid, imaginative place"
	story_prompt = form_values.get("prompt") or "an unexpected adventure"
	tone = form_values.get("tone") or "adventurous"
	art_style = form_values.get("art_style") or "Colorful comic-book art"
	dialogue_place = " ".join(setting.split()[-4:]).strip(" ,.;:") or "our adventure"
	beats = [
		("The Discovery", f"{character} arrives in {setting} and discovers a clue tied to {story_prompt}.", f"Maybe this clue points us to '{dialogue_place}'."),
		("The Challenge", f"A surprising obstacle blocks {character}'s way in {setting}.", f"This obstacle won't stop us from reaching '{dialogue_place}'!"),
		("A Clever Plan", f"{character} spots a creative solution and bravely puts it into action.", f"I have a plan to solve this near '{dialogue_place}'!"),
		("The Turning Point", f"The plan takes an unexpected turn, and {character} adapts to help those nearby.", f"That plan failed, but we can still reach '{dialogue_place}'!"),
		("A Bright Finish", f"{character} resolves the challenge, bringing the adventure to a satisfying close.", f"We did it, and '{dialogue_place}' is safe again!"),
	]
	panels = [
		{
			"panel_number": index,
			"title": title,
			"scene_description": scene_description,
			"dialogue": dialogue,
			"image_prompt": f"{art_style}; {tone} tone; {scene_description}; expressive comic composition.",
		}
		for index, (title, scene_description, dialogue) in enumerate(beats, start=1)
	]
	outline = "\n".join(
		f"{panel['panel_number']}. {panel['title']}: {panel['scene_description']}"
		for panel in panels
	)
	return outline, panels


def generate_panel_images(
    captions: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    provider = os.getenv("IMAGE_PROVIDER", "pollinations").strip().lower()

    if provider == "pollinations":
        return [
            generate_image(
                caption,
                index,
                output_path=output_dir / f"panel-{index}.png",
            )
            for index, caption in enumerate(captions, start=1)
        ]

    if provider == "huggingface":
        return [
            generate_image(
                caption,
                index,
                output_path=output_dir / f"panel-{index}.png",
            )
            for index, caption in enumerate(captions, start=1)
        ]

    if provider == "placeholder":
        width = _image_dimension("IMAGE_WIDTH", 768)
        height = _image_dimension("IMAGE_HEIGHT", 512)

        return [
            _create_placeholder_panel(
                caption,
                index,
                width,
                height,
                output_dir,
            )
            for index, caption in enumerate(captions, start=1)
        ]

    raise RuntimeError(f"Unsupported IMAGE_PROVIDER: {provider}")

def build_comic_layout(
	panel_paths: Sequence[Path],
	title: str,
	output_path: Path,
) -> Path:
	page_width, page_height = 1240, 1754
	margin, gap, header_height = 64, 28, 230
	cell_width = (page_width - 2 * margin - gap) // 2
	row_count = (len(panel_paths) + 1) // 2
	cell_height = (page_height - header_height - margin - gap * (row_count - 1)) // row_count
	page = Image.new("RGB", (page_width, page_height), "#fffdf7")
	draw = ImageDraw.Draw(page)
	title_font = _font(58)
	label_font = _font(25)
	draw.text((margin, 52), "COMICCRAFT AI  /  COMIC FILE", font=label_font, fill="#587064")
	draw.text((margin, 98), _display_text(title), font=title_font, fill="#19231f")
	draw.line((margin, 190, page_width - margin, 190), fill="#19231f", width=4)

	for index, panel_path in enumerate(panel_paths):
		column = index % 2
		row = index // 2
		x = margin + column * (cell_width + gap)
		y = header_height + row * (cell_height + gap)
		with Image.open(panel_path) as panel:
			fitted_panel = ImageOps.fit(panel.convert("RGB"), (cell_width, cell_height))
			page.paste(fitted_panel, (x, y))
		draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#19231f", width=5)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	page.save(output_path, format="PNG")
	return output_path


def export_pdf(
	panel_paths: Sequence[Path],
	output_path: Path,
	full_story: str = "",
	panels: Sequence[Mapping[str, str | int]] = (),
	captions: Sequence[str] = (),
) -> Path:
	document = FPDF(unit="mm", format="A4")
	document.set_margins(16, 16, 16)
	document.set_auto_page_break(auto=True, margin=16)

	font_path = next(
		(
			path
			for path in (
				Path("C:/Windows/Fonts/arial.ttf"),
				Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
			)
			if path.is_file()
		),
		None,
	)
	font_name = "Helvetica"
	if font_path is not None:
		font_name = "ComicCraftNarration"
		document.add_font(font_name, fname=str(font_path))
	text_for_pdf = (lambda text: text) if font_path is not None else _display_text

	for index, caption in enumerate(captions, start=1):
		panel = panels[index - 1] if index <= len(panels) else {}
		story_title, story_scene, story_narration = _parse_story_panel(full_story, index)
		title = str(panel.get("title") or story_title)
		scene = str(panel.get("scene_description") or story_scene or caption)
		narration = next(
			(
				str(panel[field])
				for field in ("caption", "dialogue", "narration")
				if panel.get(field)
			),
			story_narration or "No separate dialogue or caption.",
		)
		panel_path = panel_paths[index - 1]
		with Image.open(panel_path) as panel_image:
			image_width, image_height = panel_image.size
		image_scale = min(178 / image_width, 140 / image_height)
		pdf_width = image_width * image_scale
		pdf_height = image_height * image_scale
		document.add_page()
		document.image(
			str(panel_path),
			x=(210 - pdf_width) / 2,
			y=16,
			w=pdf_width,
			h=pdf_height,
		)
		document.set_y(16 + pdf_height + 5)
		document.set_font(font_name, size=13)
		document.multi_cell(
			0,
			8,
			text_for_pdf(f"Panel {index} Content"),
			new_x="LMARGIN",
			new_y="NEXT",
		)
		document.set_font(font_name, size=11)
		if title:
			document.multi_cell(
				0,
				7,
				text_for_pdf(f"Title: {title}"),
				new_x="LMARGIN",
				new_y="NEXT",
			)
		document.multi_cell(
			0,
			7,
			text_for_pdf(f"Scene description: {scene}"),
			new_x="LMARGIN",
			new_y="NEXT",
		)
		document.multi_cell(
			0,
			7,
			text_for_pdf(f"Caption/dialogue/narration: {narration}"),
			new_x="LMARGIN",
			new_y="NEXT",
		)

	document.add_page()
	document.set_font(font_name, size=18)
	document.multi_cell(
		0,
		10,
		"Complete Story and Panel Narration",
		new_x="LMARGIN",
		new_y="NEXT",
	)
	document.ln(2)
	document.set_font(font_name, size=11)
	document.multi_cell(
		0,
		7,
		text_for_pdf(full_story),
		new_x="LMARGIN",
		new_y="NEXT",
	)

	document.output(output_path)
	return output_path


def _parse_story_panel(story: str, panel_number: int) -> tuple[str, str, str]:
	heading_pattern = re.compile(
		r"^\s*(?:[-*]\s*)?\*{0,2}(?:\d+\.\s*)?Panel\s+(\d+)\b([^\r\n]*)",
		re.IGNORECASE | re.MULTILINE,
	)
	headings = list(heading_pattern.finditer(story))
	for heading_index, heading in enumerate(headings):
		if int(heading.group(1)) != panel_number:
			continue
		section_end = headings[heading_index + 1].start() if heading_index + 1 < len(headings) else len(story)
		section = story[heading.end():section_end]
		heading_suffix = heading.group(2).strip().strip("*").strip()
		title = heading_suffix[1:].strip() if heading_suffix.startswith(":") else ""

		def field_value(labels: str) -> str:
			match = re.search(
				rf"^\s*(?:[-*]\s*)?\*{{0,2}}(?:{labels})\*{{0,2}}\s*:\s*\*{{0,2}}\s*(.*?)(?=^\s*(?:[-*]\s*)?\*{{0,2}}(?:Visual|Scene(?: Description)?|Caption(?:/Dialogue)?|Dialogue|Narration)\*{{0,2}}\s*:|\Z)",
				section,
				re.IGNORECASE | re.MULTILINE | re.DOTALL,
			)
			return match.group(1).strip() if match else ""

		return (
			title,
			field_value("Visual|Scene(?: Description)?"),
			field_value("Caption(?:/Dialogue)?|Dialogue|Narration"),
		)
	return "", "", ""


def _normalize_panel_dialogue(dialogue: str, fallback: str) -> str:
	text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", dialogue.strip())
	text = re.sub(
		r"(?i)^(?:(?:Panel\s+\d+|Caption(?:/Dialogue)?|Dialogue)\s*:\s*)+",
		"",
		text,
	)
	text = text.strip(" \t\r\n\"'“”")
	sentences = [
		sentence.strip()
		for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
		if sentence.strip()
	]
	if len(sentences) != 1 or not 5 <= len(sentences[0].split()) <= 15:
		return fallback
	return sentences[0]


def _split_story_into_panels(story: str) -> list[str]:
	words = story.split()
	if len(words) < PANEL_COUNT:
		raise RuntimeError("Gemini returned a story that is too short for five panels.")
	return [
		" ".join(words[index * len(words) // PANEL_COUNT:(index + 1) * len(words) // PANEL_COUNT])
		for index in range(PANEL_COUNT)
	]


def _create_placeholder_panel(
	caption: str,
	index: int,
	width: int,
	height: int,
	output_dir: Path,
) -> Path:
	palettes = [
		("#f2c785", "#e7795c", "#527669"),
		("#acd3c5", "#f0b95f", "#54768a"),
		("#e7a979", "#d8df8b", "#665e7b"),
		("#98b6c0", "#ed8768", "#607a61"),
	]
	sky, accent, ground = palettes[(index - 1) % len(palettes)]
	image = Image.new("RGB", (width, height), sky)
	draw = ImageDraw.Draw(image)
	scale = min(width, height)
	for dot_y in range(20, int(height * 0.58), 30):
		for dot_x in range(20, width, 30):
			draw.ellipse((dot_x, dot_y, dot_x + 3, dot_y + 3), fill="#ffffff")

	sun_radius = int(scale * 0.11)
	sun_x, sun_y = int(width * 0.76), int(height * 0.25)
	draw.ellipse(
		(sun_x - sun_radius, sun_y - sun_radius, sun_x + sun_radius, sun_y + sun_radius),
		fill="#f7e5a4",
		outline="#19231f",
		width=max(2, scale // 180),
	)
	draw.polygon(
		[(0, int(height * 0.68)), (int(width * 0.28), int(height * 0.42)),
		 (int(width * 0.57), int(height * 0.72)), (int(width * 0.8), int(height * 0.5)),
		 (width, int(height * 0.67)), (width, height), (0, height)],
		fill=ground,
		outline="#19231f",
	)

	figure_x, figure_y = int(width * (0.25 + (index % 2) * 0.12)), int(height * 0.62)
	figure_size = int(scale * 0.13)
	draw.ellipse(
		(figure_x, figure_y - figure_size, figure_x + figure_size, figure_y),
		fill="#f2cfa8",
		outline="#19231f",
		width=max(2, scale // 180),
	)
	draw.rounded_rectangle(
		(figure_x - figure_size // 4, figure_y, figure_x + figure_size * 1.25, int(height * 0.88)),
		radius=max(4, figure_size // 4),
		fill=accent,
		outline="#19231f",
		width=max(2, scale // 180),
	)

	footer_top = int(height * 0.76)
	draw.rectangle((0, footer_top, width, height), fill="#fffdf7", outline="#19231f", width=3)
	label_font = _font(max(14, scale // 28))
	body_font = _font(max(16, scale // 22))
	draw.text((int(width * 0.04), footer_top + int(height * 0.025)), f"PLACEHOLDER ART  /  PANEL {index:02}", font=label_font, fill="#587064")
	lines = _wrap_text(draw, _display_text(caption), body_font, int(width * 0.9), 3)
	draw.multiline_text(
		(int(width * 0.04), footer_top + int(height * 0.09)),
		"\n".join(lines),
		font=body_font,
		fill="#19231f",
		spacing=4,
	)

	output_path = output_dir / f"panel-{index}.png"
	image.save(output_path, format="PNG")
	return output_path


def _image_dimension(name: str, default: int) -> int:
	try:
		return max(320, min(int(os.getenv(name, str(default))), 1600))
	except ValueError:
		return default


def _render_panel_dialogue(image_path: Path, dialogue: str) -> None:
	text = re.sub(r"\*\*(.*?)\*\*", r"\1", dialogue.strip())
	text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
	if not text:
		return

	with Image.open(image_path) as source:
		image = source.convert("RGB")
	width, height = image.size
	if len(text) > 72 and not re.search(r"[.!?]\s+", text):
		words = text.split()
		middle = len(words) // 2
		bubble_texts = [" ".join(words[:middle]), " ".join(words[middle:])]
	else:
		sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
		if len(sentences) <= 2:
			bubble_texts = sentences
		else:
			split_at = min(
				range(1, len(sentences)),
				key=lambda index: abs(
					sum(map(len, sentences[:index])) - sum(map(len, sentences[index:]))
				),
			)
			bubble_texts = [" ".join(sentences[:split_at]), " ".join(sentences[split_at:])]

	draw = ImageDraw.Draw(image)
	margin = max(12, width // 45)
	padding = max(9, width // 52)
	max_bubble_width = int(width * 0.44)
	max_text_width = max_bubble_width - 2 * padding
	starting_size = min(27, max(16, width // 32))
	minimum_size = max(12, width // 64)
	font_size = starting_size
	while True:
		font = _font(font_size)
		bubble_lines = [_wrap_dialogue(draw, value, font, max_text_width) for value in bubble_texts]
		line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + max(2, font_size // 6)
		bubble_heights = [2 * padding + line_height * len(lines) for lines in bubble_lines]
		if max(bubble_heights) <= height * 0.18 or font_size <= minimum_size:
			break
		font_size -= 1

	bubble_widths = [
		min(
			max_bubble_width,
			max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 2 * padding,
		)
		for lines in bubble_lines
	]
	for index, (bubble_text, lines, bubble_width, bubble_height) in enumerate(
		zip(bubble_texts, bubble_lines, bubble_widths, bubble_heights)
	):
		is_left = index == 0
		left = margin if is_left else width - margin - bubble_width
		top = margin
		right = left + bubble_width
		bottom = top + bubble_height
		tail_size = max(10, width // 34)
		if is_left:
			tail_base = left + int(bubble_width * 0.72)
			tail = [(tail_base - tail_size // 2, bottom - 2), (tail_base + tail_size // 2, bottom - 2), (tail_base + tail_size, bottom + tail_size)]
		else:
			tail_base = left + int(bubble_width * 0.28)
			tail = [(tail_base - tail_size // 2, bottom - 2), (tail_base + tail_size // 2, bottom - 2), (tail_base - tail_size, bottom + tail_size)]
		draw.polygon(tail, fill="#fffdf7", outline="#19231f")
		draw.rounded_rectangle(
			(left, top, right, bottom),
			radius=bubble_height // 2,
			fill="#fffdf7",
			outline="#19231f",
			width=max(2, width // 320),
		)
		draw.multiline_text(
			(left + padding, top + padding),
			"\n".join(lines),
			font=font,
			fill="#19231f",
			spacing=max(2, font_size // 6),
		)
	image.save(image_path, format="PNG")


def _wrap_dialogue(
	draw: ImageDraw.ImageDraw,
	text: str,
	font: ImageFont.ImageFont,
	max_width: int,
) -> list[str]:
	lines: list[str] = []
	for paragraph in text.splitlines() or [text]:
		current = ""
		for word in paragraph.split():
			candidate = f"{current} {word}".strip()
			if current and draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
				current = candidate
				continue
			if current:
				lines.append(current)
			current = ""
			if draw.textbbox((0, 0), word, font=font)[2] <= max_width:
				current = word
				continue
			word_parts = []
			part = ""
			for character in word:
				if part and draw.textbbox((0, 0), part + character, font=font)[2] > max_width:
					word_parts.append(part)
					part = character
				else:
					part += character
			if part:
				word_parts.append(part)
			lines.extend(word_parts[:-1])
			current = word_parts[-1]
		if current:
			lines.append(current)
	return lines or [""]


def _font(size: int) -> ImageFont.ImageFont:
	font_paths = (
		Path("C:/Windows/Fonts/arial.ttf"),
		Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
	)
	for font_path in font_paths:
		if font_path.is_file():
			return ImageFont.truetype(str(font_path), size=size)
	return ImageFont.load_default()


def _display_text(value: str) -> str:
	return value.encode("latin-1", errors="replace").decode("latin-1")


def _wrap_text(
	draw: ImageDraw.ImageDraw,
	text: str,
	font: ImageFont.ImageFont,
	max_width: int,
	max_lines: int,
) -> list[str]:
	lines: list[str] = []
	current = ""
	for word in textwrap.shorten(text, width=500, placeholder="...").split():
		candidate = f"{current} {word}".strip()
		if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
			lines.append(current)
			current = word
		else:
			current = candidate
	if current:
		lines.append(current)
	if len(lines) > max_lines:
		lines = lines[:max_lines]
		lines[-1] = textwrap.shorten(lines[-1], width=max(8, len(lines[-1]) - 3), placeholder="...")
	return lines
