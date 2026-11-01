"""Render slides.pdf pages to PNGs for visual review."""
import fitz
import os

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides.pdf")
out = os.path.join(os.path.dirname(src), "pages")
os.makedirs(out, exist_ok=True)
doc = fitz.open(src)
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=110)
    pix.save(os.path.join(out, f"page{i:02d}.png"))
print(f"rendered {len(doc)} pages to {out}")
