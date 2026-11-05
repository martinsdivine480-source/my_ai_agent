"""Turns simple Markdown-style text into real Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
and PDF files.

Needs:  pip install python-docx openpyxl python-pptx reportlab

Word and PDF: # headings, **bold**, *italic*, `code`, - bullets (indent for sub-bullets),
1. numbered lists, | tables |, > quotes, ``` code blocks ```, and [[pagebreak]].
Excel: one Markdown table per sheet, each optionally under a '## Sheet name' heading.
PowerPoint: '# Slide title' starts a slide; '- bullets', plain lines, a | table |, 'Notes: ...'."""
import os
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

INLINE = re.compile(r"(\*\*[^*\n]+\*\*|\*[^*\s][^*\n]*\*|`[^`\n]+`)")
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
BULLET = re.compile(r"^(\s*)[-*\u2022]\s+(.*)$")
NUMBER = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")


def _is_separator(line: str) -> bool:
    cleaned = line.replace("|", "").replace(":", "").replace(" ", "")
    return bool(cleaned) and set(cleaned) <= {"-"}


def _cells(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(text: str) -> str:
    return re.sub(r"[*`]", "", text).strip()


def _temp_name(target: Path) -> Path:
    return target.with_name(target.stem + ".tmp" + target.suffix)


def _atomic_save(target: Path, writer) -> None:
    """Write to a temporary file first so a failed build never damages an existing file."""
    tmp = _temp_name(target)
    try:
        writer(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------
# Word
# ---------------------------------------------------------------
def _add_runs(paragraph, text: str, bold: bool = False) -> None:
    from docx.shared import Pt

    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.italic = True
            if bold:
                run.bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            if bold:
                run.bold = True
        else:
            run = paragraph.add_run(part)
            if bold:
                run.bold = True


def _add_table(doc, rows: list) -> None:
    if not rows:
        return
    columns = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=columns)
    table.style = "Table Grid"
    for r, cells in enumerate(rows):
        for c in range(columns):
            text = cells[c] if c < len(cells) else ""
            paragraph = table.cell(r, c).paragraphs[0]
            _add_runs(paragraph, text, bold=(r == 0))
    doc.add_paragraph()


def _add_code(doc, code_lines: list) -> None:
    from docx.shared import Pt

    for line in code_lines:
        paragraph = doc.add_paragraph(style="No Spacing")
        run = paragraph.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
    doc.add_paragraph()


def build_docx(target: Path, content: str, title: str | None = None) -> None:
    try:
        from docx import Document
        from docx.shared import Cm, Pt
    except ImportError:
        raise RuntimeError("The python-docx package is not installed. Run: pip install python-docx")

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    if title and title.strip():
        doc.add_heading(_plain(title), 0)
        doc.core_properties.title = _plain(title)

    lines = content.replace("\r\n", "\n").split("\n")
    buffer: list = []
    in_code = False
    code_lines: list = []

    def flush() -> None:
        if not buffer:
            return
        paragraph = doc.add_paragraph()
        for index, text in enumerate(buffer):
            if index:
                paragraph.add_run().add_break()
            _add_runs(paragraph, text)
        buffer.clear()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()

        if stripped.startswith("```"):
            flush()
            if in_code:
                _add_code(doc, code_lines)
                code_lines = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not stripped:
            flush()
            i += 1
            continue
        if stripped == "[[pagebreak]]":
            flush()
            doc.add_page_break()
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            flush()
            i += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            flush()
            doc.add_heading(_plain(heading.group(2)), len(heading.group(1)))
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not _is_separator(lines[i]):
                    rows.append(_cells(lines[i]))
                i += 1
            _add_table(doc, rows)
            continue

        bullet = BULLET.match(raw)
        if bullet:
            flush()
            level = min(len(bullet.group(1).replace("\t", "  ")) // 2, 2)
            style = "List Bullet" if level == 0 else f"List Bullet {level + 1}"
            _add_runs(doc.add_paragraph(style=style), bullet.group(2))
            i += 1
            continue

        number = NUMBER.match(raw)
        if number:
            flush()
            level = min(len(number.group(1).replace("\t", "  ")) // 2, 2)
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.63 + 0.63 * level)
            paragraph.paragraph_format.first_line_indent = Cm(-0.63)
            _add_runs(paragraph, f"{number.group(2)}.\t{number.group(3)}")
            i += 1
            continue

        if stripped.startswith(">"):
            flush()
            _add_runs(doc.add_paragraph(style="Quote"), stripped.lstrip("> ").strip())
            i += 1
            continue

        buffer.append(stripped)
        i += 1

    flush()
    if in_code and code_lines:
        _add_code(doc, code_lines)

    tmp = target.with_name(target.name + ".tmp")
    try:
        doc.save(str(tmp))
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------
# Excel
# ---------------------------------------------------------------
def _parse_sheets(content: str) -> list:
    """[(sheet name, [row, row, ...]), ...] from Markdown tables."""
    sheets: list = []
    name = None
    rows: list = []
    for raw in content.replace("\r\n", "\n").split("\n"):
        stripped = raw.strip()
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading:
            if rows:
                sheets.append((name or f"Sheet{len(sheets) + 1}", rows))
            name, rows = _plain(heading.group(1)), []
        elif stripped.startswith("|"):
            if not _is_separator(stripped):
                rows.append(_cells(stripped))
    if rows:
        sheets.append((name or f"Sheet{len(sheets) + 1}", rows))
    return sheets


def _sheet_name(name: str, used: set) -> str:
    clean = re.sub(r"[\[\]:*?/\\]", " ", name).strip()[:31].strip() or "Sheet"
    candidate, counter = clean, 2
    while candidate.lower() in used:
        suffix = f" {counter}"
        candidate = clean[: 31 - len(suffix)] + suffix
        counter += 1
    used.add(candidate.lower())
    return candidate


def _cell_value(text: str):
    value = text.strip()
    if re.fullmatch(r"-?\d{1,15}", value) and not (len(value.lstrip("-")) > 1 and value.lstrip("-").startswith("0")):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value) and not value.lstrip("-").startswith("00"):
        return float(value)
    if value.startswith("**") and value.endswith("**") and len(value) > 4:
        return value[2:-2]
    return value


def build_xlsx(target: Path, content: str, title: str | None = None) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError("The openpyxl package is not installed. Run: pip install openpyxl")

    sheets = _parse_sheets(content)
    if not sheets:
        raise ValueError("No table was found. Write each sheet as a Markdown table, "
                         "for example: | Name | Score | then | Ann | 90 |.")
    workbook = Workbook()
    workbook.remove(workbook.active)
    if title and title.strip():
        workbook.properties.title = _plain(title)
    used: set = set()
    header_fill = PatternFill("solid", fgColor="2F5BEA")

    for name, rows in sheets:
        sheet = workbook.create_sheet(title=_sheet_name(name, used))
        columns = max(len(r) for r in rows)
        for r, cells in enumerate(rows, 1):
            for c in range(1, columns + 1):
                text = cells[c - 1] if c - 1 < len(cells) else ""
                cell = sheet.cell(row=r, column=c, value=_cell_value(text))
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if r == 1:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = header_fill
        for c in range(1, columns + 1):
            longest = max(len(str(sheet.cell(row=r, column=c).value or "")) for r in range(1, len(rows) + 1))
            sheet.column_dimensions[get_column_letter(c)].width = max(8, min(60, longest + 2))
        if len(rows) > 1:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

    _atomic_save(target, lambda tmp: workbook.save(str(tmp)))


# ---------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------
def _parse_slides(content: str) -> list:
    slides: list = []
    current = None
    for raw in content.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        heading = re.match(r"^#{1,2}\s+(.*)$", stripped)
        if heading:
            current = {"title": _plain(heading.group(1)), "items": [], "table": [], "notes": []}
            slides.append(current)
            continue
        if not stripped:
            continue
        if current is None:
            current = {"title": "", "items": [], "table": [], "notes": []}
            slides.append(current)
        if stripped.lower().startswith("notes:"):
            current["notes"].append(stripped[6:].strip())
        elif stripped.startswith("|"):
            if not _is_separator(stripped):
                current["table"].append(_cells(stripped))
        else:
            bullet = BULLET.match(line)
            number = NUMBER.match(line)
            if bullet:
                level = min(len(bullet.group(1).replace("\t", "  ")) // 2, 3)
                current["items"].append((level, _plain(bullet.group(2)), True))
            elif number:
                level = min(len(number.group(1).replace("\t", "  ")) // 2, 3)
                current["items"].append((level, f"{number.group(2)}. {_plain(number.group(3))}", True))
            else:
                current["items"].append((0, _plain(stripped.lstrip("#").strip()), False))
    return slides


def _set_notes(slide, spec: dict) -> None:
    if spec["notes"]:
        slide.notes_slide.notes_text_frame.text = "\n".join(spec["notes"])


def build_pptx(target: Path, content: str, title: str | None = None) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError:
        raise RuntimeError("The python-pptx package is not installed. Run: pip install python-pptx")

    slides = _parse_slides(content)
    if not slides:
        raise ValueError("The presentation is empty. Start each slide with a '# Slide title' line.")
    prs = Presentation()
    if title and title.strip():
        prs.core_properties.title = _plain(title)

    for index, spec in enumerate(slides):
        items = spec["items"]
        plain = [text for (_, text, is_bullet) in items if not is_bullet]
        has_bullets = any(is_bullet for (_, _, is_bullet) in items)

        if index == 0 and not has_bullets and not spec["table"]:
            slide = prs.slides.add_slide(prs.slide_layouts[0])  # title slide
            slide.shapes.title.text = spec["title"] or _plain(title or "")
            if len(slide.placeholders) > 1:
                slide.placeholders[1].text = "\n".join(plain[:3])
            _set_notes(slide, spec)
            continue

        base_title = spec["title"] or _plain(title or "")
        chunks = [items[i:i + 7] for i in range(0, len(items), 7)] or ([[]] if not spec["table"] else [])
        for number, chunk in enumerate(chunks):
            slide = prs.slides.add_slide(prs.slide_layouts[1])  # title and content
            slide.shapes.title.text = base_title + (" (cont.)" if number else "")
            frame = slide.placeholders[1].text_frame
            frame.clear()
            first = True
            for level, text, _ in chunk:
                paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
                first = False
                paragraph.text = text
                paragraph.level = level
                for run in paragraph.runs:
                    run.font.size = Pt(22 if level == 0 else 18)
            if number == 0:
                _set_notes(slide, spec)

        if spec["table"]:
            rows = spec["table"][:12]
            columns = max(len(r) for r in rows)
            slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
            slide.shapes.title.text = base_title + (" (table)" if chunks else "")
            shape = slide.shapes.add_table(len(rows), columns, Inches(0.5), Inches(1.7),
                                           Inches(9), Inches(0.4) * len(rows))
            for r, cells in enumerate(rows):
                for c in range(columns):
                    cell = shape.table.cell(r, c)
                    cell.text = _plain(cells[c]) if c < len(cells) else ""
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(14)
            if not chunks:
                _set_notes(slide, spec)

    _atomic_save(target, lambda tmp: prs.save(str(tmp)))


# ---------------------------------------------------------------
# PDF
# ---------------------------------------------------------------
_FONTS: dict = {}


def _pdf_fonts() -> dict:
    """Use real fonts from this computer when possible (they cover accents and many symbols);
    otherwise fall back to the built-in Helvetica, which only covers Western European text."""
    if _FONTS:
        return _FONTS
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    families = [
        (windows_fonts, ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu"),
         ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")),
    ]
    fonts = {"body": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique",
             "mono": "Courier", "latin_only": True}
    for folder, files in families:
        paths = [folder / name for name in files]
        if all(p.exists() for p in paths):
            try:
                for face, path in zip(("BodyFont", "BodyFont-Bold", "BodyFont-Italic", "BodyFont-BoldItalic"), paths):
                    pdfmetrics.registerFont(TTFont(face, str(path)))
                pdfmetrics.registerFontFamily("BodyFont", normal="BodyFont", bold="BodyFont-Bold",
                                              italic="BodyFont-Italic", boldItalic="BodyFont-BoldItalic")
                fonts.update(body="BodyFont", bold="BodyFont-Bold", italic="BodyFont-Italic", latin_only=False)
                break
            except Exception:
                continue
    mono_path = windows_fonts / "consola.ttf"
    if mono_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("MonoFont", str(mono_path)))
            fonts["mono"] = "MonoFont"
        except Exception:
            pass
    _FONTS.update(fonts)
    return _FONTS


def _rl_inline(text: str, mono_font: str) -> str:
    """Markdown inline formatting -> ReportLab paragraph markup (everything else is escaped)."""
    out = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append("<b>" + _xml_escape(part[2:-2]) + "</b>")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append("<i>" + _xml_escape(part[1:-1]) + "</i>")
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(f'<font face="{mono_font}">' + _xml_escape(part[1:-1]) + "</font>")
        else:
            out.append(_xml_escape(part))
    return "".join(out)


def build_pdf(target: Path, content: str, title: str | None = None) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except ImportError:
        raise RuntimeError("The reportlab package is not installed. Run: pip install reportlab")

    fonts = _pdf_fonts()
    available = A4[0] - 4 * cm

    def fix(text: str) -> str:
        return text.encode("cp1252", "replace").decode("cp1252") if fonts["latin_only"] else text

    def rl(text: str) -> str:
        return _rl_inline(fix(text), fonts["mono"])

    body = ParagraphStyle("body", fontName=fonts["body"], fontSize=10.5, leading=14.5, spaceAfter=6)
    styles = {
        "title": ParagraphStyle("title", parent=body, fontName=fonts["bold"], fontSize=24, leading=28, spaceAfter=14),
        1: ParagraphStyle("h1", parent=body, fontName=fonts["bold"], fontSize=19, leading=23, spaceBefore=12, spaceAfter=8),
        2: ParagraphStyle("h2", parent=body, fontName=fonts["bold"], fontSize=15.5, leading=19, spaceBefore=10, spaceAfter=6),
        3: ParagraphStyle("h3", parent=body, fontName=fonts["bold"], fontSize=13, leading=16, spaceBefore=8, spaceAfter=4),
        4: ParagraphStyle("h4", parent=body, fontName=fonts["bold"], fontSize=11.5, leading=14, spaceBefore=6, spaceAfter=3),
        "quote": ParagraphStyle("quote", parent=body, fontName=fonts["italic"], leftIndent=18, textColor=colors.HexColor("#555555")),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9, leading=11.5, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=body, fontName=fonts["bold"], fontSize=9, leading=11.5, spaceAfter=0),
        "code": ParagraphStyle("code", fontName=fonts["mono"], fontSize=8.5, leading=11),
    }

    story: list = []
    if title and title.strip():
        story.append(Paragraph(rl(_plain(title)), styles["title"]))

    def add_table(rows: list) -> None:
        columns = max(len(r) for r in rows)
        data = []
        for r, cells in enumerate(rows):
            style = styles["cellhead"] if r == 0 else styles["cell"]
            data.append([Paragraph(rl(cells[c]) if c < len(cells) else "", style) for c in range(columns)])
        table = Table(data, colWidths=[available / columns] * columns, repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8BECB")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def add_code(code_lines: list) -> None:
        wrapped = []
        for line in code_lines:
            wrapped.extend(textwrap.wrap(fix(line.expandtabs(4)), width=88, drop_whitespace=False,
                                         replace_whitespace=False) or [""])
        block = Table([[Preformatted("\n".join(wrapped), styles["code"])]], colWidths=[available])
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F3F7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DFE3EA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(block)
        story.append(Spacer(1, 8))

    lines = content.replace("\r\n", "\n").split("\n")
    buffer: list = []
    in_code = False
    code_lines: list = []

    def flush() -> None:
        if buffer:
            story.append(Paragraph("<br/>".join(rl(t) for t in buffer), body))
            buffer.clear()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()

        if stripped.startswith("```"):
            flush()
            if in_code:
                add_code(code_lines)
                code_lines = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        if stripped == "[[pagebreak]]":
            flush()
            story.append(PageBreak())
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            flush()
            story.append(Spacer(1, 6))
            i += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            flush()
            story.append(Paragraph(rl(_plain(heading.group(2))), styles[len(heading.group(1))]))
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not _is_separator(lines[i]):
                    rows.append(_cells(lines[i]))
                i += 1
            if rows:
                add_table(rows)
            continue

        bullet = BULLET.match(raw)
        number = NUMBER.match(raw)
        if bullet or number:
            flush()
            match = bullet or number
            level = min(len(match.group(1).replace("\t", "  ")) // 2, 3)
            item_style = ParagraphStyle("item", parent=body, leftIndent=18 + level * 14,
                                        bulletIndent=4 + level * 14, spaceAfter=3, bulletFontName=fonts["body"])
            if bullet:
                story.append(Paragraph(rl(bullet.group(2)), item_style, bulletText="\u2022"))
            else:
                story.append(Paragraph(rl(number.group(3)), item_style, bulletText=f"{number.group(2)}."))
            i += 1
            continue

        if stripped.startswith(">"):
            flush()
            story.append(Paragraph(rl(stripped.lstrip("> ").strip()), styles["quote"]))
            i += 1
            continue

        buffer.append(stripped)
        i += 1

    flush()
    if in_code and code_lines:
        add_code(code_lines)

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(fonts["body"], 9)
        canvas.setFillGray(0.5)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))
        canvas.restoreState()

    def write(tmp: Path) -> None:
        pdf = SimpleDocTemplate(str(tmp), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm, title=_plain(title or ""))
        pdf.build(story, onFirstPage=footer, onLaterPages=footer)

    _atomic_save(target, write)