from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

from services.classifier import ARCHETYPE_COLORS


CARD_WIDTH = 1200
CARD_HEIGHT = 630
BACKGROUND = "#0d1117"
TEXT_PRIMARY = "#f0f6fc"
TEXT_MUTED = "#8b949e"
BAR_BG = "#21262d"
OUTPUT_DIR = Path("/tmp")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/Library/Fonts/FiraCode-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Menlo Bold.ttf",
            "/Library/Fonts/FiraCode-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ] + candidates
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _fetch_avatar(avatar_url: str) -> Image.Image:
    if not avatar_url:
        return Image.new("RGBA", (160, 160), "#30363d")

    try:
        response = httpx.get(avatar_url, timeout=10.0)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGBA")
    except Exception:
        return Image.new("RGBA", (160, 160), "#30363d")

    image = ImageOps.fit(image, (160, 160), centering=(0.5, 0.5))
    mask = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 160, 160), fill=255)
    output = Image.new("RGBA", (160, 160))
    output.paste(image, (0, 0), mask)
    return output


def _bar(draw: ImageDraw.ImageDraw, label: str, value: float, x: int, y: int, width: int, accent: str, font: ImageFont.ImageFont) -> None:
    draw.text((x, y), label, font=font, fill=TEXT_PRIMARY)
    bar_top = y + 28
    draw.rounded_rectangle((x, bar_top, x + width, bar_top + 18), radius=9, fill=BAR_BG)
    filled = int(width * (max(0.0, min(100.0, value)) / 100.0))
    draw.rounded_rectangle((x, bar_top, x + max(filled, 18), bar_top + 18), radius=9, fill=accent)
    draw.text((x + width + 16, y + 4), f"{int(round(value))}", font=font, fill=TEXT_MUTED)


def _topics_text(topics: Iterable[str]) -> str:
    cleaned = [topic for topic in topics if topic]
    return " · ".join(cleaned[:3]) if cleaned else "open source · automation · tooling"


def generate_profile_card(raw_data: Dict[str, Any], feature_data: Dict[str, Any], classification: Dict[str, Any]) -> str:
    username = raw_data["user"]["login"]
    archetype = classification["archetype"]
    accent = ARCHETYPE_COLORS.get(archetype, "#58a6ff")
    radar_scores = classification["radar_scores"]

    image = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(52, bold=True)
    subtitle_font = _load_font(26, bold=True)
    body_font = _load_font(22)
    small_font = _load_font(18)

    avatar = _fetch_avatar(raw_data["user"].get("avatar_url", ""))
    image.paste(avatar, (70, 70), avatar)

    draw.text((260, 90), f"@{username}", font=title_font, fill=TEXT_PRIMARY)

    badge_x, badge_y = 260, 165
    badge_width = 220
    draw.rounded_rectangle((badge_x, badge_y, badge_x + badge_width, badge_y + 48), radius=24, fill=accent)
    draw.text((badge_x + 24, badge_y + 11), archetype, font=subtitle_font, fill=BACKGROUND)

    percentile = int(round(classification["technical_depth_percentile"]))
    draw.text((70, 285), f"{percentile}th percentile technical depth", font=subtitle_font, fill=TEXT_PRIMARY)
    draw.text(
        (70, 330),
        raw_data["user"].get("bio", "")[:90] or "GitHub profile analyzed from repositories, commits, comments, and pull requests.",
        font=body_font,
        fill=TEXT_MUTED,
    )

    start_x = 700
    start_y = 90
    for index, (label, value) in enumerate(radar_scores.items()):
        _bar(draw, label, value, start_x, start_y + (index * 76), 340, accent, body_font)

    draw.text((70, 520), "Top topics", font=small_font, fill=TEXT_MUTED)
    draw.text((70, 548), _topics_text(feature_data.get("topics", [])), font=body_font, fill=TEXT_PRIMARY)

    draw.text((900, 585), "github-profiler.app", font=small_font, fill=TEXT_MUTED)

    output_path = OUTPUT_DIR / f"{username}_card.png"
    image.convert("RGB").save(output_path, format="PNG")
    return str(output_path)
