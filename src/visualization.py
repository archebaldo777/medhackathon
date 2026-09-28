from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from .constants import DEFAULT_PIXEL_SPACING_X_MM, DEFAULT_PIXEL_SPACING_Y_MM, SPINE
from .features import spine_axis_line
from .model import ModelResult, QualityModel


def _display_image(
    normalized: np.ndarray,
    pixel_spacing_y_mm: float | None = None,
    pixel_spacing_x_mm: float | None = None,
    target_height: int = 640,
) -> Image.Image:
    """Grayscale image resampled to its physical aspect ratio.

    The scanner pixel is 1.05 mm tall and 0.60 mm wide, so drawing it square
    stretches the anatomy sideways by 75 % and makes a tilted axis look less
    tilted than it is. The display grid is therefore square in millimetres.
    """
    image = Image.fromarray((np.clip(normalized, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    rows, columns = normalized.shape
    spacing_y = float(pixel_spacing_y_mm or DEFAULT_PIXEL_SPACING_Y_MM)
    spacing_x = float(pixel_spacing_x_mm or DEFAULT_PIXEL_SPACING_X_MM)
    height = min(target_height, max(2 * rows, 320))
    width = max(1, round(height * (columns * spacing_x) / (rows * spacing_y)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def source_png(path: str | Path) -> bytes:
    """PNG of a DICOM image at its physical aspect ratio, without overlays."""
    import io

    from .dicom_io import normalize_pixels, read_dicom

    dicom = read_dicom(path)
    image = _display_image(
        normalize_pixels(dicom.pixels), dicom.pixel_spacing_y_mm, dicom.pixel_spacing_x_mm
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


IMAGE_BOX_KEY = "evectio_image_box"

MARKUP_COLOURS = {
    "edge": (236, 222, 64),
    "roi": (95, 215, 232),
    "landmark": (255, 168, 90),
}


def draw_markup(image: Image.Image, markup: dict | None, shape: tuple[int, int]) -> None:
    """Draw markup given in source pixels, styled like the densitometer screen.

    Bone edges are yellow and measurement regions cyan, as on the densitometer
    the specialist works with, so the proposal reads the way the markup it
    replaces does.
    """
    if not markup:
        return
    rows, columns = shape
    scale_x = image.width / columns
    scale_y = image.height / rows
    draw = ImageDraw.Draw(image)
    for item in list(markup.get("derived", [])) + list(markup.get("shapes", [])):
        points = [(x * scale_x, y * scale_y) for x, y in item.get("points", [])]
        if len(points) < 2:
            continue
        colour = MARKUP_COLOURS.get(item.get("style", "roi"), (255, 255, 255))
        width = 1 if item.get("style") == "edge" else 2
        if item.get("type") == "ELLIPSE":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            draw.ellipse((min(xs), min(ys), max(xs), max(ys)), outline=colour, width=1)
            continue
        draw.line(points + ([points[0]] if item.get("type") == "POLYGON" else []), fill=colour, width=width)
        label = str(item.get("label", ""))
        if label in {"L1", "L2", "L3", "L4"}:
            anchor_x = min(p[0] for p in points) + 4
            anchor_y = sum(p[1] for p in points) / len(points) - 6
            draw.text((anchor_x, anchor_y), label, fill=colour)


def _font(size: int):
    """Pillow's built-in scalable font; the bitmap default on older Pillow."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _importance_overlay(
    model: QualityModel,
    normalized: np.ndarray,
    result: ModelResult,
    size: tuple[int, int],
    pixel_spacing_y_mm: float | None = None,
    pixel_spacing_x_mm: float | None = None,
) -> Image.Image:
    heatmap = model.heatmap(
        normalized,
        result.region,
        result,
        pixel_spacing_y_mm or DEFAULT_PIXEL_SPACING_Y_MM,
        pixel_spacing_x_mm or DEFAULT_PIXEL_SPACING_X_MM,
    )
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    value = np.asarray(heat, dtype=np.float32) / 255.0
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (145 * (1.0 - value)).astype(np.uint8)
    rgba[..., 3] = (value * 92).astype(np.uint8)
    return Image.fromarray(rgba)


def _spine_axis(
    image: Image.Image,
    normalized: np.ndarray,
    pixel_spacing_y_mm: float | None = None,
    pixel_spacing_x_mm: float | None = None,
    show_reference: bool = False,
) -> float | None:
    """Draw the fitted lumbar axis and return its physical tilt in degrees.

    The trace comes from the same function the axis descriptor uses, so the line
    and the angle printed above it are what the model measured. When the axis
    head fires, the vertical it should follow is drawn as a dashed reference —
    the proposed correction the specialist confirms or rejects.
    """
    axis = spine_axis_line(
        normalized,
        pixel_spacing_y_mm or DEFAULT_PIXEL_SPACING_Y_MM,
        pixel_spacing_x_mm or DEFAULT_PIXEL_SPACING_X_MM,
    )
    if axis is None:
        return None
    height, width = normalized.shape
    scale_x = image.width / width
    scale_y = image.height / height
    y0, y1 = axis["row_start"] * scale_y, axis["row_end"] * scale_y
    x0, x1 = axis["column_start"] * scale_x, axis["column_end"] * scale_x
    draw = ImageDraw.Draw(image)
    if show_reference:
        centre = (x0 + x1) / 2.0
        dash = max(6, int(image.height / 40))
        y = y0
        while y < y1:
            draw.line((centre, y, centre, min(y + dash, y1)), fill=(255, 220, 120), width=2)
            y += dash * 2
    draw.line((x0, y0, x1, y1), fill=(59, 224, 189), width=3)
    return float(axis["angle_deg"])


def save_visualization(
    model: QualityModel,
    normalized: np.ndarray,
    result: ModelResult,
    output: str | Path,
    pixel_spacing_y_mm: float | None = None,
    pixel_spacing_x_mm: float | None = None,
    markup: dict | None = None,
) -> None:
    image = _display_image(normalized, pixel_spacing_y_mm, pixel_spacing_x_mm).convert("RGBA")
    image = Image.alpha_composite(
        image,
        _importance_overlay(
            model, normalized, result, image.size, pixel_spacing_y_mm, pixel_spacing_x_mm
        ),
    )
    rgb = image.convert("RGB")
    draw_markup(rgb, markup, normalized.shape)
    axis = (
        _spine_axis(
            rgb,
            normalized,
            pixel_spacing_y_mm,
            pixel_spacing_x_mm,
            show_reference="spine_axis_deviation" in result.violation_keys,
        )
        if result.region == SPINE
        else None
    )

    # Result on the image itself, the way a second reader reports it: a status
    # bar above the image (solid = issue, dashed = doubt, grey = no issue), a
    # short header and a footer with the number of findings.
    issue = bool(result.quality_class)
    doubt = not issue and 0.85 * result.decision_threshold <= result.quality_probability < result.decision_threshold
    accent, navy, grey = (244, 236, 106), (14, 23, 38), (150, 160, 172)
    small, regular = _font(12), _font(14)
    margin, bar, header, footer = 16, 8, 64, 34
    width = max(rgb.width + 2 * margin, 360)
    canvas = Image.new("RGB", (width, bar + header + rgb.height + footer), navy)
    draw = ImageDraw.Draw(canvas)
    if issue:
        draw.rectangle((0, 0, width, bar - 1), fill=accent)
    elif doubt:
        for x in range(0, width, 22):
            draw.rectangle((x, 0, x + 13, bar - 1), fill=accent)
    else:
        draw.rectangle((0, 0, width, bar - 1), fill=(60, 70, 84))
    draw.text((margin, bar + 12), "EVECTIO", fill=grey, font=small)
    label = "QUALITY ISSUE" if issue else "DOUBT" if doubt else "NO ISSUE"
    label_width = int(draw.textlength(label, font=small)) + 20
    box = (width - margin - label_width, bar + 8, width - margin, bar + 30)
    if issue:
        draw.rounded_rectangle(box, radius=5, fill=accent)
        draw.text((box[0] + 10, box[1] + 4), label, fill=(27, 27, 10), font=small)
    elif doubt:
        draw.rounded_rectangle(box, radius=5, outline=accent, width=2)
        draw.text((box[0] + 10, box[1] + 4), label, fill=accent, font=small)
    else:
        draw.rounded_rectangle(box, radius=5, fill=(40, 50, 64))
        draw.text((box[0] + 10, box[1] + 4), label, fill=grey, font=small)
    summary = f"risk {result.quality_probability:.2f}  |  threshold {result.decision_threshold:.2f}"
    if axis is not None:
        summary += f"  |  axis {axis:+.1f} deg (limit 5)"
    draw.text((margin, bar + 36), summary, fill=(230, 234, 240), font=regular)
    top = bar + header
    canvas.paste(rgb, ((width - rgb.width) // 2, top))
    foot = top + rgb.height
    draw.line((0, foot, width, foot), fill=(34, 44, 58))
    region_name = "LUMBAR SPINE" if result.region == SPINE else "PROXIMAL FEMUR"
    if markup and markup.get("shapes"):
        region_name += "  |  MARKUP: AUTO"
    draw.text((margin, foot + 10), region_name, fill=grey, font=small)
    count = f"{len(result.violation_keys):02d}"
    count_width = int(draw.textlength(count, font=regular))
    draw.text((width - margin - count_width, foot + 8), count, fill=accent, font=regular)
    caption = "ISSUES"
    draw.text((width - margin - count_width - 8 - int(draw.textlength(caption, font=small)), foot + 10), caption, fill=grey, font=small)
    # Where the image sits inside the report picture, so the web viewer can show
    # the same frame as the plain image and the markup editor without jumping.
    info = PngImagePlugin.PngInfo()
    info.add_text(IMAGE_BOX_KEY, f"{(width - rgb.width) // 2},{top},{rgb.width},{rgb.height}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", pnginfo=info)


def overlay_png(path: str | Path) -> bytes:
    """The image area of a saved report picture, without its header and footer.

    It has the same size as ``source_png`` of the same study, so the web viewer
    switches between the plain image, the AI overlay and the markup editor in
    one frame.
    """
    import io

    with Image.open(path) as picture:
        box = picture.info.get(IMAGE_BOX_KEY)
        image = picture.convert("RGB")
        if box:
            x, y, w, h = (int(value) for value in box.split(","))
            image = image.crop((x, y, x + w, y + h))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
