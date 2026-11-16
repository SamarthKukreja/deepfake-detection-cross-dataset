"""Generate the vector end-to-end pipeline used as Fig. 1."""

from pathlib import Path

import reportlab
from reportlab.lib.colors import Color, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


font_dir = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("Vera", str(font_dir / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Vera-Bold", str(font_dir / "VeraBd.ttf")))

width, height = 520, 230
box_w, box_h = 160, 78
xs = [0, 180, 360]
top_y, bottom_y = 138, 14

out = Path("Fig1.pdf")
c = canvas.Canvas(
    str(out), pagesize=(width, height), pageCompression=1,
    initialFontName="Vera",
)
c.setTitle("Shared deepfake-detection pipeline")

outline = Color(0.18, 0.29, 0.72)
fill = Color(0.97, 0.98, 1.00)
header = Color(0.10, 0.20, 0.58)


def draw_box(x, y, number, title, lines):
    c.setFillColor(fill)
    c.setStrokeColor(outline)
    c.setLineWidth(1.1)
    c.roundRect(x, y, box_w, box_h, 5, fill=1, stroke=1)
    c.setFillColor(header)
    c.setFont("Vera-Bold", 10.5)
    c.drawString(x + 8, y + box_h - 16, f"{number}. {title}")
    c.setFillColor(black)
    c.setFont("Vera", 10)
    text_y = y + box_h - 32
    for line in lines:
        c.drawCentredString(x + box_w / 2, text_y, line)
        text_y -= 13


def arrow(x1, y1, x2, y2):
    c.setStrokeColor(outline)
    c.setFillColor(outline)
    c.setLineWidth(1.2)
    c.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        c.line(x2, y2, x2 - 6 * direction, y2 + 3)
        c.line(x2, y2, x2 - 6 * direction, y2 - 3)
    else:
        direction = 1 if y2 > y1 else -1
        c.line(x2, y2, x2 + 3, y2 - 6 * direction)
        c.line(x2, y2, x2 - 3, y2 - 6 * direction)


draw_box(xs[0], top_y, 1, "Source videos", [
    "FaceForensics++,",
    "Celeb-DF (v2),",
    "DeepFake Detection (DFD)",
])
draw_box(xs[1], top_y, 2, "Frame sampling", [
    "Every 10th frame",
    "Maximum 20 crops per video",
])
draw_box(xs[2], top_y, 3, "Face extraction", [
    "MTCNN confidence >= 0.95",
    "Side >= 60 px; 20% padding",
    "Blur filtering",
])
draw_box(xs[2], bottom_y, 4, "Dataset splits", [
    "Source-video-disjoint splits",
    "Source-domain validation",
    "Held-out target videos",
])
draw_box(xs[1], bottom_y, 5, "Fine-tuning", [
    "Five ImageNet-pretrained",
    "backbones; common protocol",
    "and fixed seed",
])
draw_box(xs[0], bottom_y, 6, "Video evaluation", [
    "Five-view TTA; mean score",
    "AUC, F1, EER, CI, paired tests",
])

mid_top = top_y + box_h / 2
mid_bottom = bottom_y + box_h / 2
arrow(xs[0] + box_w, mid_top, xs[1] - 8, mid_top)
arrow(xs[1] + box_w, mid_top, xs[2] - 8, mid_top)
arrow(xs[2] + box_w / 2, top_y - 4, xs[2] + box_w / 2, bottom_y + box_h + 4)
arrow(xs[2] - 8, mid_bottom, xs[1] + box_w + 8, mid_bottom)
arrow(xs[1] - 8, mid_bottom, xs[0] + box_w + 8, mid_bottom)

c.showPage()
c.save()
