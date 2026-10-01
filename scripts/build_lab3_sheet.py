"""Build Lab Sheet 3.pdf from its editable Markdown source.

Run from any directory after installing reportlab:
    python -m pip install reportlab
    python scripts/build_lab3_sheet.py
"""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Lab Sheet 3.md"
OUTPUT = ROOT / "Lab Sheet 3.pdf"


def inline(text):
    """Render the small set of Markdown used by this lab sheet."""
    parts = []
    pattern = re.compile(r"\[([^]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*")
    start = 0
    for match in pattern.finditer(text):
        parts.append(escape(text[start:match.start()]))
        label, url, code, bold = match.groups()
        if label is not None:
            parts.append(f'<link href="{escape(url, quote=True)}" color="#205b93">{escape(label)}</link>')
        elif code is not None:
            parts.append(f'<font name="Courier" size="9">{escape(code)}</font>')
        else:
            parts.append(f"<b>{escape(bold)}</b>")
        start = match.end()
    parts.append(escape(text[start:]))
    return "".join(parts).replace("  \n", "<br/>")


def main():
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Arial", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="LabTitle", fontName="Arial-Bold", fontSize=15, leading=19,
        alignment=TA_CENTER, spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        name="LabSubtitle", fontName="Arial-Bold", fontSize=12, leading=16,
        alignment=TA_CENTER, spaceAfter=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="LabHeading", fontName="Arial-Bold", fontSize=11.5, leading=15,
        spaceBefore=4 * mm, spaceAfter=2 * mm, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="LabBody", fontName="Arial", fontSize=10.5, leading=15,
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="LabCode", fontName="Courier", fontSize=8.5, leading=12,
        leftIndent=5 * mm, spaceBefore=1 * mm, spaceAfter=3 * mm,
        backColor=colors.HexColor("#f2f4f7"), borderPadding=5,
    ))

    story = []
    paragraph = []
    code = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["LabBody"]))
            paragraph.clear()

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code), styles["LabCode"]))
                code.clear()
            in_code = not in_code
            continue
        if in_code:
            code.append(raw_line)
        elif not line:
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), styles["LabTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline(line[3:]), styles["LabSubtitle"]))
        elif line.startswith("### "):
            flush_paragraph()
            if line.startswith("### Task 2:"):
                story.append(PageBreak())
            story.append(Paragraph(inline(line[4:]), styles["LabHeading"]))
        else:
            paragraph.append(line)
    flush_paragraph()

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="COMP3222 Lab Sheet 3",
        author="COMP3222 Machine Learning Technologies",
    )
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
