"""Generate the vector cross-dataset summary used as Fig. 2."""

from pathlib import Path
from statistics import mean

import reportlab
from reportlab.lib.colors import Color, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

font_dir = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("Vera", str(font_dir / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Vera-Bold", str(font_dir / "VeraBd.ttf")))

values = [
    [0.900, 0.736, 0.924, 0.957, 0.867],
    [0.713, 0.573, 0.799, 0.806, 0.668],
    [0.506, 0.543, 0.589, 0.576, 0.588],
    [0.788, 0.735, 0.812, 0.915, 0.713],
]
labels = [("Exp1", "FF++ to FF++"), ("Exp2", "FF++ to Celeb-DF"),
          ("Exp3", "Celeb-DF to FF++"), ("Exp4", "Mixed to DFD")]
grays = [0.82, 0.67, 0.88, 0.55]
width, height = 480, 230
left, bottom, top = 52, 42, 212
plot_w, plot_h = 408, top - bottom
y_min, y_max = 0.45, 1.0


def y(value):
    return bottom + (value - y_min) / (y_max - y_min) * plot_h


out = Path("Fig2.pdf")
c = canvas.Canvas(
    str(out), pagesize=(width, height), pageCompression=1,
    initialFontName="Vera",
)
c.setTitle("Cross-dataset generalization summary")
c.setStrokeColor(Color(0.82, 0.82, 0.82))
c.setLineWidth(0.5)
for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    c.line(left, y(tick), left + plot_w, y(tick))
    c.setFillColor(black)
    c.setFont("Vera", 9.5)
    c.drawRightString(left - 5, y(tick) - 2.5, f"{tick:.1f}")

c.setStrokeColor(black)
c.setLineWidth(0.8)
c.line(left, bottom, left + plot_w, bottom)
c.line(left, bottom, left, top)
c.setDash(4, 3)
c.line(left, y(0.5), left + plot_w, y(0.5))
c.setDash()

centres = [100, 200, 300, 400]
bar_w = 46
for index, (series, centre, gray, label) in enumerate(zip(values, centres, grays, labels)):
    avg, lo, hi = mean(series), min(series), max(series)
    x0, y0, y1 = centre - bar_w / 2, y(0.5), y(avg)
    c.setFillColor(Color(gray, gray, gray))
    c.setStrokeColor(black)
    c.rect(x0, y0, bar_w, y1 - y0, fill=1, stroke=1)
    c.saveState()
    path = c.beginPath()
    path.rect(x0, y0, bar_w, y1 - y0)
    c.clipPath(path, stroke=0, fill=0)
    c.setStrokeColor(Color(0.30, 0.30, 0.30))
    c.setLineWidth(0.45)
    if index == 0:
        for xx in range(int(x0) - 30, int(x0 + bar_w) + 30, 8):
            c.line(xx, y0, xx + 40, y1)
    elif index == 1:
        for xx in range(int(x0), int(x0 + bar_w) + 1, 7):
            c.line(xx, y0, xx, y1)
    elif index == 2:
        for yy in range(int(y0) + 4, int(y1), 7):
            for xx in range(int(x0) + 4, int(x0 + bar_w), 8):
                c.circle(xx, yy, 0.65, fill=1, stroke=0)
    else:
        for yy in range(int(y0) + 5, int(y1), 8):
            c.line(x0, yy, x0 + bar_w, yy)
    c.restoreState()
    c.setStrokeColor(black)
    c.setLineWidth(0.9)
    c.line(centre, y(lo), centre, y(hi))
    c.line(centre - 5, y(lo), centre + 5, y(lo))
    c.line(centre - 5, y(hi), centre + 5, y(hi))
    c.setFillColor(black)
    c.setFont("Vera-Bold", 10.5)
    c.drawCentredString(centre, y(hi) + 6, f"{avg:.2f}")
    c.drawCentredString(centre, 27, label[0])
    c.setFillColor(black)
    c.setFont("Vera", 9.5)
    c.drawCentredString(centre, 16, label[1])

c.saveState()
c.translate(13, 105)
c.rotate(90)
c.setFillColor(black)
c.setFont("Vera-Bold", 10.5)
c.drawCentredString(0, 0, "Video-level AUC")
c.restoreState()
c.showPage()
c.save()
