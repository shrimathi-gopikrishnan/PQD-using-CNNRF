"""Build paper.docx from main.tex content.

IEEE-style two-column layout (Times New Roman 10pt body).
The header (title, authors, abstract, index terms) and full-width
figures/tables use single-column section breaks; the body flows in two
columns. The algorithm block in the LaTeX source is rendered as a
numbered table for readability.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
FIG_DIR = PAPER_DIR / "figures"
OUT = PAPER_DIR / "paper.docx"

FONT = "Times New Roman"


# ----------------------------- column / section helpers --------------------

def _set_section_columns(section, num=1, space_pt=24):
    """Set the column count on a python-docx Section."""
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    # Strip prior attributes so we don't end up with stale equalWidth/sep
    # combinations that Word treats as malformed.
    for k in list(cols.attrib.keys()):
        del cols.attrib[k]
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_pt * 20)))  # twentieths of a pt


def _set_section_continuous(section):
    """Mark this section as a continuous (no page break) section."""
    sectPr = section._sectPr
    typ = sectPr.find(qn("w:type"))
    if typ is None:
        typ = OxmlElement("w:type")
        sectPr.append(typ)
    typ.set(qn("w:val"), "continuous")


def add_continuous_section(doc, num_cols):
    """Insert a continuous section break and switch to ``num_cols`` columns."""
    new_section = doc.add_section(WD_SECTION.CONTINUOUS)
    new_section.page_height = Inches(11)
    new_section.page_width = Inches(8.5)
    new_section.top_margin = Inches(0.75)
    new_section.bottom_margin = Inches(1.0)
    new_section.left_margin = Inches(0.75)
    new_section.right_margin = Inches(0.75)
    _set_section_columns(new_section, num=num_cols)
    _set_section_continuous(new_section)
    return new_section


# ----------------------------- helpers --------------------------------------

def set_run(run, *, bold=False, italic=False, size=10, font=FONT, color=None,
            superscript=False, subscript=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    if superscript:
        run.font.superscript = True
    if subscript:
        run.font.subscript = True
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
        rFonts.set(qn(f"w:{attr}"), font)


def add_paragraph(doc, text="", *, style=None, align=None, size=10,
                  bold=False, italic=False, space_before=0, space_after=4,
                  first_line_indent=None, line_spacing=1.15):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        set_run(run, size=size, bold=bold, italic=italic)
    return p


def add_runs(p, segments, *, size=10):
    """segments is a list of (text, kwargs) — kwargs passed to set_run."""
    for text, kw in segments:
        run = p.add_run(text)
        set_run(run, size=size, **kw)


def justified_paragraph(doc, text=None, *, size=10, first_line_indent=Cm(0.5),
                        space_after=4, line_spacing=1.15, segments=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    if segments is not None:
        add_runs(p, segments, size=size)
    elif text is not None:
        run = p.add_run(text)
        set_run(run, size=size)
    return p


def section_heading(doc, num, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(f"{num}. {title.upper()}")
    set_run(run, bold=True, size=10)


def subsection_heading(doc, letter, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(f"{letter}. ")
    set_run(run, italic=True, size=10)
    run = p.add_run(title)
    set_run(run, italic=True, size=10)


def subsubsection_heading(doc, num, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(f"{num}) ")
    set_run(run, italic=True, size=10)
    run = p.add_run(title + ":")
    set_run(run, italic=True, size=10)


def equation_line(doc, segments, number=None):
    """segments list passed to add_runs; number is the (n) tag."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, segments, size=10)
    if number is not None:
        # right-align the number using a tab stop at the right margin
        pPr = p._element.get_or_add_pPr()
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "right")
        tab.set(qn("w:pos"), "9000")  # ~6.25 inches
        tabs.append(tab)
        pPr.append(tabs)
        run = p.add_run("\t" + f"({number})")
        set_run(run, size=10)
    return p


def caption(doc, text, *, before=4, after=8):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.15
    run = p.add_run(text)
    set_run(run, size=9, italic=False)
    return p


def figure_image(doc, path, width_inches=6.3, caption_text=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_inches))
    if caption_text:
        caption(doc, caption_text)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def style_cell(cell, text, *, bold=False, italic=False, size=9, align=None,
               va=WD_ALIGN_VERTICAL.CENTER, fill=None):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    set_run(run, bold=bold, italic=italic, size=size)
    cell.vertical_alignment = va
    if fill:
        shade_cell(cell, fill)


def add_horizontal_border(cell, position, sz=8, color="000000"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    border = OxmlElement(f"w:{position}")
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), str(sz))
    border.set(qn("w:space"), "0")
    border.set(qn("w:color"), color)
    tcBorders.append(border)


def remove_all_borders(table):
    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for pos in ("top", "left", "bottom", "right", "insideH", "insideV"):
                b = OxmlElement(f"w:{pos}")
                b.set(qn("w:val"), "nil")
                tcBorders.append(b)
            tcPr.append(tcBorders)


def booktab_table(doc, headers, rows, col_widths=None, table_caption=None,
                  table_label=None):
    """Render a table in IEEE booktabs style with caption above."""
    if table_label or table_caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        if table_label:
            run = p.add_run(table_label.upper())
            set_run(run, size=9, bold=False)
            run = p.add_run("\n")
            set_run(run, size=9)
        if table_caption:
            run = p.add_run(table_caption)
            set_run(run, size=9, bold=False, italic=False)

    n_cols = len(headers)
    n_rows = 1 + len(rows)
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    remove_all_borders(table)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = w

    # header row
    for i, h in enumerate(headers):
        style_cell(table.rows[0].cells[i], h, bold=True, size=9,
                   align=WD_ALIGN_PARAGRAPH.CENTER)
        add_horizontal_border(table.rows[0].cells[i], "top", sz=12)
        add_horizontal_border(table.rows[0].cells[i], "bottom", sz=8)

    # body rows
    for r, row in enumerate(rows, start=1):
        for i, val in enumerate(row):
            is_bold = isinstance(val, tuple) and len(val) > 1 and val[1].get("bold")
            text = val[0] if isinstance(val, tuple) else str(val)
            style_cell(
                table.rows[r].cells[i], text,
                bold=is_bold, size=9,
                align=WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT,
            )
        if r == n_rows - 1:
            for i in range(n_cols):
                add_horizontal_border(table.rows[r].cells[i], "bottom", sz=12)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    return table


def add_horizontal_rule(p):
    pPr = p._element.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single")
    bot.set(qn("w:sz"), "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "000000")
    pBdr.append(bot)
    pPr.append(pBdr)


# ----------------------------- math glyphs ----------------------------------

# Use Unicode math characters. Variables that should be italic are wrapped
# in italic runs at call sites.

SQRT = "√"     # √
SUM = "∑"      # Σ
SIGMA = "σ"    # σ
SIGMA_CAP = "Σ"
MU = "μ"       # μ
DELTA = "Δ"    # Δ
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃",
       "4": "₄", "5": "₅", "6": "₆", "7": "₇",
       "8": "₈", "9": "₉", "n": "ₙ", "x": "ₓ",
       "j": "ⱼ", "k": "ₖ", "h": "ₕ", "max": "max"}
SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³",
       "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
       "8": "⁸", "9": "⁹", "-": "⁻", "T": "ᵀ"}
APPROX = "≈"
LEQ = "≤"
GEQ = "≥"
TIMES = "×"
PLUSMN = "±"
MINUS = "−"
RARR = "→"


# ----------------------------- main builder ---------------------------------

def build():
    doc = Document()

    # Page setup: US Letter, narrow IEEE-ish margins
    section = doc.sections[0]
    section.page_height = Inches(11)
    section.page_width = Inches(8.5)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    # Header section is single-column (title, authors, abstract, index terms).
    _set_section_columns(section, num=1)

    # Default style
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(10)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts")) or OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
        rfonts.set(qn(f"w:{attr}"), FONT)
    if rfonts.getparent() is None:
        rpr.append(rfonts)

    # ============================ TITLE =====================================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(
        "Two-Stage Hybrid CNN and Random Forest for Real-Time "
        "Power Quality Disturbance Classification"
    )
    set_run(run, bold=True, size=20)

    # ============================ AUTHORS ===================================
    authors = [
        ("Shrimathi G",
         "Department of Electrical and\nElectronics Engineering\n"
         "Sri Venkateswara College of Engineering\nPennalur 602117\n"
         "2022ee0135@svce.ac.in"),
        ("Sanjay Kumar V",
         "Department of Electrical and\nElectronics Engineering\n"
         "Sri Venkateswara College of Engineering\nPennalur 602117\n"
         "2022ee0214@svce.ac.in"),
        ("Suresh N",
         "Department of Electrical and\nElectronics Engineering\n"
         "Sri Venkateswara College of Engineering\nPennalur 602117\n"
         "sureshn@svce.ac.in"),
    ]
    auth_table = doc.add_table(rows=1, cols=3)
    auth_table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    remove_all_borders(auth_table)
    for i, (name, aff) in enumerate(authors):
        cell = auth_table.rows[0].cells[i]
        cell.text = ""
        # name
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(name)
        set_run(run, bold=True, size=11)
        # affiliation lines
        for line in aff.split("\n"):
            ap = cell.add_paragraph()
            ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ap.paragraph_format.space_before = Pt(0)
            ap.paragraph_format.space_after = Pt(0)
            ap.paragraph_format.line_spacing = 1.1
            run = ap.add_run(line)
            set_run(run, size=10, italic=False)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ============================ ABSTRACT ==================================
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_after = Pt(6)
    pf.left_indent = Inches(0.25)
    pf.right_indent = Inches(0.25)
    run = p.add_run("Abstract—")
    set_run(run, bold=True, italic=True, size=9)
    abstract_text = (
        "This paper presents a two-stage hybrid pipeline for real-time "
        "classification of power quality disturbances (PQDs) defined under "
        "IEEE Standard 1159. Single-cycle voltage windows of 100 samples at "
        "5 kHz are first screened by an O(N) threshold detector using five "
        "physical features: root mean square, low-order total harmonic "
        "distortion, excess kurtosis, maximum consecutive-sample difference, "
        "and quarter-window RMS dispersion. Windows flagged as Abnormal are "
        "escalated to a hybrid Stage 2 classifier in which a compact "
        "one-dimensional convolutional neural network learns a 64-dimensional "
        "feature representation from the raw waveform, and a Random Forest "
        "with 200 trees classifies seventeen classes (one normal, eight single "
        "disturbances, eight pairwise compound disturbances). Each prediction "
        "is enriched at run time with a severity grade, cause, "
        "equipment-at-risk, and three to five operator actions from a 17-entry "
        "IEEE 1159 knowledge base, and is streamed to a browser dashboard "
        "through a Flask-SocketIO backend. On a stratified 80-to-20 split of "
        "the 17,000-signal XPQRS dataset, Stage 2 attains 98.85% accuracy "
        "(thirteen of seventeen classes at 100% recall), the end-to-end "
        "pipeline reaches 96.67% on a balanced mixed-traffic test, Stage 1 "
        "detects 100% of Normal windows at zero false alarms, and inference "
        "takes approximately 67 ms on a single-thread laptop CPU without GPU "
        "or cloud. A preprocessing analysis shows that per-sample "
        "maximum-absolute normalization, common in deep-learning PQD work, "
        "collapses accuracy to 52% by discarding amplitude information that "
        "separates Sag, Swell, and Interruption from Pure Sinusoidal; a single "
        "global-scale normalization recovers the missing 47 percentage points. "
        "The system is suitable for edge deployment on distribution feeders."
    )
    run = p.add_run(abstract_text)
    set_run(run, bold=True, italic=False, size=9)
    # Make whole abstract bold-italic per IEEE convention is too heavy;
    # standard is bold body. Override: keep italic abstract per IEEEtran.
    for r in p.runs[1:]:
        r.bold = True
        r.italic = False

    # Index Terms
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_after = Pt(10)
    pf.left_indent = Inches(0.25)
    pf.right_indent = Inches(0.25)
    run = p.add_run("Index Terms—")
    set_run(run, bold=True, italic=True, size=9)
    run = p.add_run(
        "Power quality disturbance classification, IEEE Standard 1159, "
        "two-stage pipeline, hybrid CNN-Random Forest, real-time monitoring, "
        "edge intelligence, knowledge-based reasoning."
    )
    set_run(run, bold=True, italic=False, size=9)

    # Switch to two-column body for the rest of the paper.
    add_continuous_section(doc, num_cols=2)

    # ============================ I. INTRODUCTION ===========================
    section_heading(doc, "I", "Introduction")
    justified_paragraph(doc, segments=[
        ("Power quality (PQ) has emerged as a critical concern in contemporary "
         "power systems owing to the increasing penetration of renewable energy "
         "resources, power-electronic converters, electric-vehicle charging "
         "infrastructure, and non-linear industrial loads. The injection of "
         "non-sinusoidal currents into distribution networks produces voltage "
         "waveforms whose deviation from the nominal sinusoid is frequently the "
         "limiting factor for sensitive end-use equipment. The categorization of "
         "such deviations (voltage sag, voltage swell, interruption, harmonic "
         "distortion, flicker, impulsive transient, oscillatory transient, "
         "notching, and their pairwise compound combinations) is formally "
         "defined in IEEE Standard 1159 [1]. A deployable PQ monitor is "
         "required to identify the specific disturbance class within an "
         "operationally relevant time window [2].", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Conventional PQ classification approaches extract handcrafted "
         "statistical or spectral features (root mean square, total harmonic "
         "distortion, wavelet energies, higher-order statistics) from each "
         "waveform window and apply a shallow classifier such as a support "
         "vector machine, an extreme learning machine, or a decision-tree "
         "ensemble [3], [4]. Recent literature has demonstrated that "
         "one-dimensional convolutional neural networks (1D-CNNs) [5], [6], "
         "[7] can achieve competitive classification accuracy directly from "
         "the raw voltage cycle without explicit feature engineering. Hybrid "
         "configurations [8], [9], in which a CNN serves as a learned feature "
         "extractor followed by a classical classifier, represent an active "
         "line of investigation aimed at obtaining the representational "
         "capacity of deep models while retaining the deployment "
         "characteristics of classical methods.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("This paper presents a real-time PQ classification system that "
         "integrates these two paradigms within a unified two-stage architecture "
         "and contributes three system-level components that, to the best of "
         "the authors' knowledge, have not been reported together in a single "
         "published pipeline. First, a five-feature ", {}),
        ("O(N)", {"italic": True}),
        (" rule-based screen serves as Stage 1 and resolves clean traffic in "
         "less than one millisecond, restricting the heavier classifier to "
         "flagged windows. Second, a hybrid 1D-CNN and Random Forest "
         "classifier serves as Stage 2 and performs 17-class classification "
         "on the windows escalated by Stage 1. Third, an IEEE 1159-derived "
         "knowledge base augments every classified output with a severity "
         "grade, physical cause, equipment-at-risk, and three to five "
         "immediate operator actions. The full pipeline executes on a "
         "single-thread laptop-class CPU without graphics-processing-unit "
         "(GPU) acceleration or external network connectivity, and is shown "
         "to satisfy operational latency requirements that cloud-deployed "
         "deep PQ classifiers cannot meet under typical wide-area "
         "round-trip-time constraints.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The remainder of the paper is organized as follows. Section II "
         "reviews related work. Section III formalizes the problem statement, "
         "summarizes the contributions, and articulates the novelties. "
         "Section IV describes the dataset and the underlying signal model. "
         "Section V details the proposed methodology and the supporting "
         "mathematical formulation. Section VI describes the real-time "
         "backend and the dashboard. Section VII presents the experimental "
         "setup. Section VIII reports the model-selection comparison. "
         "Section IX presents results and discussion. Section X enumerates "
         "the limitations of the present study, and Section XI concludes "
         "the paper.", {}),
    ])

    # ============================ II. RELATED WORK ==========================
    section_heading(doc, "II", "Related Work")
    justified_paragraph(doc, segments=[
        ("PQ disturbance classification has been studied extensively. "
         "Khetarpal and Tripathi [2] provide a comprehensive review of the "
         "field, surveying signal-processing, machine-learning, and hybrid "
         "approaches and identifying the trade-offs between feature-engineering "
         "depth and classification accuracy. Thirumala ", {}),
        ("et al.", {"italic": True}),
        (" [3] apply empirical wavelet transform-based adaptive filtering with "
         "a multiclass support vector machine. Sahani and Dash [4] introduce a "
         "weighted bidirectional extreme learning machine in conjunction with "
         "the Hilbert–Huang transform.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Deep-learning approaches have been investigated more recently. "
         "Wang ", {}),
        ("et al.", {"italic": True}),
        (" [5] employ a deep convolutional neural network on "
         "compressed-sensing reconstructions of voltage cycles. Cai ", {}),
        ("et al.", {"italic": True}),
        (" [6] feed Wigner–Ville distribution images to a deep CNN. "
         "Eristi ", {}),
        ("et al.", {"italic": True}),
        (" [7] combine a one-dimensional local binary pattern operator with a "
         "CNN. Garcia ", {}),
        ("et al.", {"italic": True}),
        (" [9] perform a comparative study of CNN, long short-term memory, "
         "and CNN-LSTM architectures applied to the same PQ disturbance set. "
         "The principal trade-off across these systems is model size: typical "
         "published deep PQ classifiers carry between 10⁵ and 10⁶ "
         "parameters, which is acceptable for server-side execution but is "
         "unsuitable for embedded deployment.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The work most closely related to the present study is that of "
         "Cai ", {}),
        ("et al.", {"italic": True}),
        (" [8], which integrates a CNN feature extractor with an extreme "
         "gradient boosting classifier for PQ disturbance recognition. The "
         "present work adopts the broader hybrid template (a CNN feature "
         "extractor combined with an ensemble classifier head) and extends it "
         "along three axes: (i) the introduction of an ", {}),
        ("O(N)", {"italic": True}),
        (" rule-based screening stage that precedes the heavier model; (ii) "
         "the extension of the class set to seventeen, including eight "
         "pairwise compound disturbances; and (iii) the integration of a "
         "deterministic IEEE 1159-grounded knowledge base that emits "
         "severity, cause, equipment-at-risk, and immediate-action metadata "
         "for every classified window.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Two-stage architectures that screen with a fast rule and invoke a "
         "heavier model only on suspicious windows are common in time-series "
         "anomaly detection in general. However, no published PQ system known "
         "to the authors has reported the screen behaviour with both detection "
         "rate and false-alarm rate, employed a hybrid CNN and tree-ensemble "
         "classifier as the second stage, and emitted operator-actionable "
         "knowledge-base metadata as part of the deployed pipeline.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Interpretability of PQ classifiers has historically received limited "
         "attention. The TreeSHAP attribution method of Lundberg ", {}),
        ("et al.", {"italic": True}),
        (" [10] enables tractable per-prediction attribution for tree-based "
         "models, and is applied in the present work as a post-hoc explainer "
         "on the legacy 36-feature Random Forest baseline. Broader surveys of "
         "explainable artificial intelligence are provided in [11].", {}),
    ])

    # =========== III. PROBLEM STATEMENT, CONTRIBUTIONS, NOVELTIES ===========
    section_heading(doc, "III", "Problem Statement, Contributions, and Novelties")

    subsection_heading(doc, "A", "Problem Statement")
    justified_paragraph(doc, segments=[
        ("Modern power systems are increasingly subject to PQ disturbances "
         "arising from the proliferation of renewable energy sources, "
         "power-electronic converters, and non-linear loads. These disturbances "
         "introduce non-stationary and non-linear characteristics into the "
         "voltage waveform, and their reliable identification under realistic "
         "noise conditions remains technically demanding. Three constraints "
         "define the operational solution space.", {}),
    ])
    subsubsection_heading(doc, "1", "Latency")
    justified_paragraph(doc, segments=[
        ("A real-time PQ classifier must complete inference within an "
         "operationally meaningful time-scale; for monitoring and diagnostic "
         "applications this is one to a few cycles of the fundamental.", {}),
    ])
    subsubsection_heading(doc, "2", "Class confusability")
    justified_paragraph(doc, segments=[
        ("Several IEEE 1159 classes differ from the normal reference solely "
         "in amplitude (sag, swell, interruption), and several others are "
         "pairwise compounds of two single disturbances. Both characteristics "
         "challenge feature- and representation-learning methods.", {}),
    ])
    subsubsection_heading(doc, "3", "Deployability")
    justified_paragraph(doc, segments=[
        ("A significant fraction of the published deep-learning PQ literature "
         "presupposes cloud or GPU compute, which is incompatible with "
         "edge-relay or substation-class hardware. Cloud deployment further "
         "introduces a network round-trip-time penalty that conventional "
         "links cannot absorb within a one-cycle decision deadline.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("A classifier is therefore required that simultaneously (a) achieves "
         "high accuracy across a large class set, (b) sustains low "
         "single-thread CPU latency, (c) provides per-prediction "
         "interpretability, (d) emits an output more actionable than a bare "
         "class label, and (e) operates without external infrastructure.", {}),
    ])

    subsection_heading(doc, "B", "Contributions")
    justified_paragraph(doc, segments=[
        ("The principal contributions of this work are as follows.", {}),
    ])
    contribs = [
        ("A two-stage classification pipeline comprising a five-feature O(N) "
         "threshold screen as Stage 1 and a hybrid 1D-CNN and Random Forest "
         "classifier as Stage 2."),
        ("A hybrid CNN-RF classifier that achieves 98.85% accuracy across "
         "seventeen IEEE 1159 classes, with thirteen of seventeen classes at "
         "100% recall."),
        ("A knowledge-based output layer that augments every prediction with "
         "severity, cause, equipment-at-risk, and immediate-action metadata "
         "derived from IEEE Standard 1159."),
        ("A real-time backend and browser dashboard that streams predictions "
         "to all connected clients with sub-cycle Stage 1 detection and "
         "approximately three-cycle Stage 2 classification on a laptop-class "
         "CPU."),
        ("An input-normalization analysis that documents and explains a "
         "47-percentage-point accuracy collapse caused by per-sample "
         "maximum-absolute scaling, together with a single-line global-scale "
         "fix."),
        ("A post-hoc SHAP explainer applied to the legacy 36-feature Random "
         "Forest baseline, returning the top three contributing physical "
         "features for any single prediction."),
    ]
    for i, c in enumerate(contribs, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Inches(0.4)
        pf.first_line_indent = Inches(-0.25)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.15
        run = p.add_run(f"{i}) ")
        set_run(run, size=10)
        run = p.add_run(c)
        set_run(run, size=10)

    subsection_heading(doc, "C", "Novelties")
    justified_paragraph(doc, segments=[
        ("The novelties of the proposed system, taken together, define an "
         "integrated pipeline whose combination of features has not been "
         "reported in prior PQ literature.", {}),
    ])

    novelties = [
        ("N1. Two-stage architecture with rule-based screening and hybrid "
         "deep classification.",
         "The majority of published PQ classifiers operate as single-stage "
         "end-to-end models. The proposed system introduces a five-feature "
         "O(N) rule-based screen that resolves the clean fraction of traffic "
         "in under one millisecond and reserves the heavier hybrid model for "
         "windows that cannot be classified at the rule level. The screen is "
         "characterized by both a 100% detection rate and a 0% false-alarm "
         "rate, a combination not jointly reported in prior PQ literature."),
        ("N2. Hybrid 1D-CNN and Random Forest with frozen-feature retuning.",
         "The 1D-CNN learns a 64-dimensional representation directly from the "
         "raw cycle, and the Random Forest classifies on this representation. "
         "The hybrid arrangement combines deep representational learning with "
         "the calibrated top-K probability outputs of a tree ensemble, and "
         "permits the Random Forest head to be re-tuned on the frozen feature "
         "space within seconds, without re-training the CNN."),
        ("N3. Knowledge-based severity-aware output layer.",
         "Every prediction is enriched at inference time with a severity "
         "grade, two-sentence physical cause, equipment-at-risk, and three to "
         "five operator-level actions, drawn from a 17-entry knowledge base "
         "derived from IEEE Standard 1159. No prior PQ classifier known to "
         "the authors emits this level of actionable diagnosis as part of the "
         "deployed pipeline."),
        ("N4. Seventeen-class coverage including eight pairwise compound "
         "disturbances.",
         "The proposed system extends the class set beyond the typical nine "
         "to fourteen classes covered in prior deep PQ classifiers, and "
         "demonstrates thirteen of seventeen classes at 100% recall."),
        ("N5. Global-scale normalization for amplitude-preserving deep "
         "learning on PQ signals.",
         "A 47-percentage-point accuracy collapse caused by the per-sample "
         "maximum-absolute scaling commonly employed in the deep-learning PQ "
         "literature is documented, and a single-line global-scale "
         "normalization that restores the missing accuracy is presented. The "
         "result is transferable to any deep PQ pipeline that includes "
         "amplitude-distinguished classes."),
        ("N6. Real-time multi-source backend and offline-tolerant dashboard.",
         "The Flask-SocketIO and TCP backend ingests live signals from "
         "MATLAB, Python, and JSON-over-HTTP senders concurrently, and "
         "pushes predictions to a single-file browser dashboard with "
         "severity-coded panels, top-three confidence display, latency "
         "timeline, and an offline demo-mode fallback. The complete system "
         "executes on a laptop-class CPU without GPU acceleration or cloud "
         "connectivity."),
        ("N7. Edge-deployable inference profile within the cloud "
         "round-trip-time floor.",
         "Stage 1 detection is completed in less than one millisecond, and "
         "Stage 2 classification in approximately three cycles of the 50 Hz "
         "fundamental. Wide-area cloud round-trip times are typically "
         "100–300 ms before any inference is initiated, which is several "
         "multiples of one PQ cycle. The proposed system therefore satisfies "
         "an operational deadline that cloud-deployed deep PQ classifiers "
         "cannot satisfy under realistic network conditions."),
        ("N8. Post-hoc SHAP attribution with physical-feature semantics.",
         "A TreeSHAP explainer is integrated on the legacy 36-feature Random "
         "Forest baseline, returning the top three contributing physical "
         "features (root mean square, total harmonic distortion, harmonic "
         "magnitudes, wavelet energies) for any single prediction. Operators "
         "are therefore able to review borderline classifications using "
         "named physical quantities rather than abstract activation maps."),
    ]
    for label, body in novelties:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Cm(0.5)
        pf.space_before = Pt(4)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.15
        run = p.add_run(label + " ")
        set_run(run, italic=True, size=10)
        run = p.add_run(body)
        set_run(run, size=10)

    # ============================ IV. DATASET ===============================
    section_heading(doc, "IV", "Dataset and Signal Model")
    justified_paragraph(doc, segments=[
        ("Each input to the proposed system is a one-cycle voltage window of "
         "length ", {}),
        ("N", {"italic": True}),
        (" = 100 samples acquired at a sampling rate ", {}),
        ("f", {"italic": True}),
        ("s", {"italic": True, "subscript": True}),
        (" = 5 kHz, corresponding to one period of the 50 Hz fundamental. "
         "Amplitudes are expressed in per-unit (pu) with the nominal peak "
         "value normalized to 1.0. The underlying signal-generation model "
         "conforms to the disturbance definitions of IEEE Standard 1159 [1]. "
         "The publicly available XPQRS-style synthetic dataset is employed "
         "for training and evaluation. The dataset comprises 17,000 signals "
         "balanced across seventeen classes (1,000 signals per class). The "
         "seventeen classes are: Pure Sinusoidal (the normal reference), "
         "voltage sag, voltage swell, interruption, impulsive transient, "
         "oscillatory transient, harmonics, notch, flicker, and eight "
         "pairwise compound disturbances formed by combinations of sag, "
         "swell, harmonics, flicker, oscillatory transient, and notch.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Additive white Gaussian noise (AWGN) at a signal-to-noise ratio of "
         "40 dB is applied during training to discourage memorization of "
         "artefacts of the synthetic generator. A stratified 80-to-20 split "
         "with random seed 42 is used, yielding 13,600 training samples and "
         "3,400 test samples (200 per class).", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The signal characteristics merit explicit description. A pure "
         "sinusoid is a unit-amplitude sine wave at 50 Hz. A voltage sag "
         "corresponds to multiplication of the sinusoid by a constant in the "
         "range 0.1–0.9 pu over a stretch of cycles, while a voltage "
         "swell corresponds to multiplication by a constant in the range "
         "1.1–1.8 pu. An interruption is approximated by a near-zero "
         "amplitude window. Harmonic disturbances superimpose integer-multiple "
         "sinusoids on the fundamental. Flicker corresponds to slow amplitude "
         "modulation of the fundamental. Several classes (sag, swell, and "
         "interruption) therefore differ from the normal reference solely in "
         "amplitude. As demonstrated in Section IX, this characteristic "
         "interacts critically with the choice of input normalization.", {}),
    ])

    # ============================ V. METHODOLOGY ============================
    section_heading(doc, "V", "Methodology")

    subsection_heading(doc, "A", "System Overview")
    justified_paragraph(doc, segments=[
        ("The proposed framework follows a structured pipeline: signal "
         "acquisition, preprocessing, two-stage classification, knowledge-base "
         "enrichment, and real-time push to the dashboard. Fig. 1 illustrates "
         "the overall architecture. A signal arrives via TCP, HTTP, or a "
         "SocketIO event. Stage 1 produces a binary Normal/Abnormal decision "
         "in under one millisecond. If the decision is Normal, the prediction "
         "is reported as Pure Sinusoidal and the call returns. If the "
         "decision is Abnormal, the same window is forwarded to Stage 2, "
         "which produces a 17-class label and a confidence vector. The "
         "result is enriched with knowledge-base metadata and pushed to all "
         "connected dashboard clients.", {}),
    ])

    figure_image(
        doc, FIG_DIR / "architecture.png", width_inches=3.3,
        caption_text=(
            "Fig. 1. Two-stage classification architecture. Stage 1 is a "
            "rule-based threshold detector with a sub-millisecond budget; "
            "Stage 2 is invoked only on Abnormal windows and is a hybrid "
            "one-dimensional convolutional neural network feature extractor "
            "followed by a Random Forest classifier. Each prediction is "
            "enriched online from an IEEE 1159 knowledge base before being "
            "pushed to the dashboard."
        ),
    )

    subsection_heading(doc, "B", "Signal Acquisition")
    justified_paragraph(doc, segments=[
        ("The input to the system is a discrete-time voltage signal", {}),
    ])
    equation_line(doc, segments=[
        ("x(n),", {"italic": True}),
        ("   ", {}),
        ("n", {"italic": True}),
        (" = 0, 1, 2, …, ", {}),
        ("N", {"italic": True}),
        (" − 1,", {}),
    ], number=1)
    justified_paragraph(doc, segments=[
        ("where ", {}),
        ("N", {"italic": True}),
        (" = 100 samples represents one cycle of the 50 Hz fundamental at ", {}),
        ("f", {"italic": True}),
        ("s", {"italic": True, "subscript": True}),
        (" = 5 kHz. Signals can be sourced from MATLAB, a Python sender, a "
         "JSON HTTP POST request, or a SocketIO event.", {}),
    ])

    subsection_heading(doc, "C", "Preprocessing and Global Normalization")
    justified_paragraph(doc, segments=[
        ("In contrast to the conventional per-sample scaling ", {}),
        ("x̃(n)", {"italic": True}),
        (" = (", {}),
        ("x(n)", {"italic": True}),
        (" − μ", {}),
        ("x", {"italic": True, "subscript": True}),
        (")/σ", {}),
        ("x", {"italic": True, "subscript": True}),
        (", which removes amplitude information from each window, the proposed "
         "system applies a single global scale that is computed once on the "
         "training set and stored with the model:", {}),
    ])
    equation_line(doc, segments=[
        ("x̃(n)", {"italic": True}),
        (" = ", {}),
        ("x(n)", {"italic": True}),
        (" / ", {}),
        ("s", {"italic": True}),
        (",   ", {}),
        ("s", {"italic": True}),
        (" = Q₉₉ ( { |", {}),
        ("x", {"italic": True}),
        ("i", {"italic": True, "subscript": True}),
        ("(train)", {"superscript": True}),
        ("| } ),", {}),
    ], number=2)
    justified_paragraph(doc, segments=[
        ("where Q₉₉(·) denotes the 99th percentile of the "
         "absolute amplitudes of the entire training set. In the present "
         "implementation ", {}),
        ("s", {"italic": True}),
        (" = 2.0535. The reason per-sample scaling is unsuitable for "
         "amplitude-distinguished PQD classes, and the recovery achieved by "
         "global scaling, are reported in Section IX-C.", {}),
    ])

    subsection_heading(doc, "D", "Stage 1: Threshold Detector")
    justified_paragraph(doc, segments=[
        ("Stage 1 computes five features and returns Abnormal if any one of "
         "them exceeds its threshold.", {}),
    ])

    subsubsection_heading(doc, "1", "Root mean square (RMS)")
    equation_line(doc, segments=[
        ("V", {"italic": True}),
        ("RMS", {"italic": True, "subscript": True}),
        (" = √ ( (1/", {}),
        ("N", {"italic": True}),
        (") ∑", {}),
        ("n", {"italic": True, "subscript": True}),
        ("=0", {"subscript": True}),
        ("N", {"italic": True, "superscript": True}),
        ("−1", {"superscript": True}),
        (" ", {}),
        ("x(n)", {"italic": True}),
        ("² ).", {}),
    ], number=3)
    justified_paragraph(doc, segments=[
        ("The detector trips if ", {}),
        ("V", {"italic": True}),
        ("RMS", {"italic": True, "subscript": True}),
        (" < 0.65 pu (sag, interruption) or ", {}),
        ("V", {"italic": True}),
        ("RMS", {"italic": True, "subscript": True}),
        (" > 0.78 pu (swell).", {}),
    ])

    subsubsection_heading(doc, "2", "Low-order total harmonic distortion approximation")
    justified_paragraph(doc, segments=[
        ("Using the magnitudes ", {}),
        ("V", {"italic": True}),
        ("h", {"italic": True, "subscript": True}),
        (" of the ", {}),
        ("h", {"italic": True}),
        ("-th harmonic bin obtained from the FFT of the window,", {}),
    ])
    equation_line(doc, segments=[
        ("THD", {}),
        ("lo", {"subscript": True}),
        (" = √(", {}),
        ("V", {"italic": True}),
        ("2", {"italic": True, "subscript": True}),
        ("² + ", {}),
        ("V", {"italic": True}),
        ("3", {"italic": True, "subscript": True}),
        ("² + ", {}),
        ("V", {"italic": True}),
        ("5", {"italic": True, "subscript": True}),
        ("²) / ", {}),
        ("V", {"italic": True}),
        ("1", {"italic": True, "subscript": True}),
        (".", {}),
    ], number=4)
    justified_paragraph(doc, segments=[
        ("The detector trips if THD", {}),
        ("lo", {"subscript": True}),
        (" > 0.025 (harmonics, notch).", {}),
    ])

    subsubsection_heading(doc, "3", "Kurtosis")
    equation_line(doc, segments=[
        ("K", {"italic": True}),
        (" = [ (1/", {}),
        ("N", {"italic": True}),
        (") ∑", {}),
        ("n", {"italic": True, "subscript": True}),
        (" (", {}),
        ("x(n)", {"italic": True}),
        (" − μ", {}),
        ("x", {"italic": True, "subscript": True}),
        (")⁴ ] / [ (1/", {}),
        ("N", {"italic": True}),
        (") ∑", {}),
        ("n", {"italic": True, "subscript": True}),
        (" (", {}),
        ("x(n)", {"italic": True}),
        (" − μ", {}),
        ("x", {"italic": True, "subscript": True}),
        (")² ]².", {}),
    ], number=5)
    justified_paragraph(doc, segments=[
        ("This is the Pearson form of kurtosis, for which a clean "
         "unit-amplitude sinusoid yields ", {}),
        ("K", {"italic": True}),
        (" ≈ 1.5. The detector trips if ", {}),
        ("K", {"italic": True}),
        (" > 2.5 (impulsive transients).", {}),
    ])

    subsubsection_heading(doc, "4", "Maximum consecutive-sample difference")
    equation_line(doc, segments=[
        ("Δ", {}),
        ("max", {"subscript": True}),
        (" = max", {}),
        ("n", {"italic": True, "subscript": True}),
        ("=1,…,", {"subscript": True}),
        ("N", {"italic": True, "subscript": True}),
        ("−1", {"subscript": True}),
        (" |", {}),
        ("x(n)", {"italic": True}),
        (" − ", {}),
        ("x(n", {"italic": True}),
        ("−1)", {"italic": True}),
        ("|.", {}),
    ], number=6)
    justified_paragraph(doc, segments=[
        ("The detector trips if Δ", {}),
        ("max", {"subscript": True}),
        (" > 0.15 pu (notch, oscillatory burst).", {}),
    ])

    subsubsection_heading(doc, "5", "Quarter-window RMS dispersion")
    justified_paragraph(doc, segments=[
        ("The window is partitioned into four quarters of length ", {}),
        ("N", {"italic": True}),
        ("/4 = 25, and the RMS ", {}),
        ("r", {"italic": True}),
        ("q", {"italic": True, "subscript": True}),
        (" of each quarter is computed; the dispersion is", {}),
    ])
    equation_line(doc, segments=[
        ("σ", {"italic": True}),
        ("R", {"italic": True, "subscript": True}),
        (" = std(", {}),
        ("r", {"italic": True}),
        ("1", {"italic": True, "subscript": True}),
        (", ", {}),
        ("r", {"italic": True}),
        ("2", {"italic": True, "subscript": True}),
        (", ", {}),
        ("r", {"italic": True}),
        ("3", {"italic": True, "subscript": True}),
        (", ", {}),
        ("r", {"italic": True}),
        ("4", {"italic": True, "subscript": True}),
        (").", {}),
    ], number=7)
    justified_paragraph(doc, segments=[
        ("The detector trips if σ", {}),
        ("R", {"italic": True, "subscript": True}),
        (" > 0.020 (flicker).", {}),
    ])

    justified_paragraph(doc, segments=[
        ("All five features are O(", {}),
        ("N", {"italic": True}),
        (") on the 100-sample window, yielding a Stage 1 budget below 1 ms on "
         "a generic CPU. The complete decision logic is summarized in "
         "Table I. The thresholds are deliberately set on the conservative "
         "side. The role of Stage 1 is not to perform fine-grained "
         "classification but to filter clean traffic out of the heavier "
         "inference path while maintaining a zero Abnormal-as-Normal "
         "misclassification rate.", {}),
    ])

    # Table I: Stage 1 decision logic (replaces algorithm)
    booktab_table(
        doc,
        headers=["Step", "Feature", "Eqn.", "Trip condition", "Flagged"],
        rows=[
            ["1", "RMS",               "(3)", "V_RMS ∉ [0.65, 0.78] pu", "Sag / Swell / Intr."],
            ["2", "Low-order THD",     "(4)", "THD_lo > 0.025",          "Harmonics / Notch"],
            ["3", "Kurtosis",          "(5)", "K > 2.5",                 "Impulsive transient"],
            ["4", "Max sample diff.",  "(6)", "Δ_max > 0.15 pu",        "Notch / Osc. burst"],
            ["5", "Quarter-RMS disp.", "(7)", "σ_R > 0.020",            "Flicker"],
        ],
        col_widths=[Inches(0.32), Inches(0.95), Inches(0.32), Inches(0.95), Inches(0.85)],
        table_label="Table I",
        table_caption=(
            "Stage 1 threshold-detector decision logic. The five features "
            "are evaluated in order; if none trips, the window is Normal."
        ),
    )

    subsection_heading(doc, "E", "Stage 2: Hybrid 1D-CNN and Random Forest")
    justified_paragraph(doc, segments=[
        ("Stage 2 produces the seventeen-class decision. The 1D-CNN does not "
         "output class logits at inference time; instead, its penultimate "
         "layer provides a 64-dimensional feature vector that is consumed by "
         "a Random Forest classifier.", {}),
    ])

    subsubsection_heading(doc, "1", "Multi-domain feature foundation (legacy reference)")
    justified_paragraph(doc, segments=[
        ("For the legacy 36-feature baseline employed in the model-comparison "
         "study (Section VIII), features are extracted across three domains.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Time domain (14 features): ", {"italic": True}),
        ("including RMS (Eq. 3), mean", {}),
    ])
    equation_line(doc, segments=[
        ("μ", {}),
        ("x", {"italic": True, "subscript": True}),
        (" = (1/", {}),
        ("N", {"italic": True}),
        (") ∑", {}),
        ("n", {"italic": True, "subscript": True}),
        ("=0", {"subscript": True}),
        ("N", {"italic": True, "superscript": True}),
        ("−1", {"superscript": True}),
        (" ", {}),
        ("x(n),", {"italic": True}),
    ], number=8)
    justified_paragraph(doc, segments=[("crest factor", {})])
    equation_line(doc, segments=[
        ("CF = ", {}),
        ("V", {"italic": True}),
        ("peak", {"italic": True, "subscript": True}),
        (" / ", {}),
        ("V", {"italic": True}),
        ("RMS", {"italic": True, "subscript": True}),
        (",", {}),
    ], number=9)
    justified_paragraph(doc, segments=[
        ("peak amplitude, kurtosis, skewness, zero-crossing rate, and "
         "quarter-window statistics.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Frequency domain (10 features): ", {"italic": True}),
        ("including the full total harmonic distortion", {}),
    ])
    equation_line(doc, segments=[
        ("THD = √( ∑", {}),
        ("h", {"italic": True, "subscript": True}),
        ("=2", {"subscript": True}),
        ("H", {"italic": True, "superscript": True}),
        (" ", {}),
        ("V", {"italic": True}),
        ("h", {"italic": True, "subscript": True}),
        ("² ) / ", {}),
        ("V", {"italic": True}),
        ("1", {"italic": True, "subscript": True}),
        (",", {}),
    ], number=10)
    justified_paragraph(doc, segments=[
        ("the magnitudes of the first ten harmonic bins of the FFT, and the "
         "fundamental amplitude ", {}),
        ("V", {"italic": True}),
        ("1", {"italic": True, "subscript": True}),
        (".", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Wavelet domain (12 features): ", {"italic": True}),
        ("obtained from a three-level Daubechies-4 discrete wavelet transform. "
         "The energy at decomposition level ", {}),
        ("j", {"italic": True}),
        (" is", {}),
    ])
    equation_line(doc, segments=[
        ("E", {"italic": True}),
        ("j", {"italic": True, "subscript": True}),
        (" = ∑", {}),
        ("k", {"italic": True, "subscript": True}),
        ("=1", {"subscript": True}),
        ("M", {"italic": True, "superscript": True}),
        ("j", {"italic": True, "superscript": True}),
        (" |", {}),
        ("d", {"italic": True}),
        ("j,k", {"italic": True, "subscript": True}),
        ("|²,", {}),
    ], number=11)
    justified_paragraph(doc, segments=[
        ("where ", {}),
        ("d", {"italic": True}),
        ("j,k", {"italic": True, "subscript": True}),
        (" is the ", {}),
        ("k", {"italic": True}),
        ("-th detail coefficient at level ", {}),
        ("j", {"italic": True}),
        (" and ", {}),
        ("M", {"italic": True}),
        ("j", {"italic": True, "subscript": True}),
        (" is the number of coefficients at that level.", {}),
    ])

    subsubsection_heading(doc, "2", "1D-CNN feature extractor")
    justified_paragraph(doc, segments=[
        ("The globally normalized window x̃ (Eq. 2) is supplied directly "
         "to a compact 1D-CNN whose architecture is specified as follows:", {}),
    ])
    cnn_arch = [
        ("Three convolutional blocks of the form Conv1D(k=5, c) → "
         "BatchNorm → ReLU → MaxPool(2), with channel widths "
         "c ∈ {16, 32, 64}."),
        ("A global average pooling layer that collapses the temporal axis."),
        ("A fully connected hidden layer with 128 units and ReLU activation."),
        ("A 64-unit feature layer; its activation forms the feature vector "
         "f ∈ ℝ⁶⁴."),
        ("A softmax output head used during training only."),
    ]
    for i, c in enumerate(cnn_arch, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Inches(0.4)
        pf.first_line_indent = Inches(-0.25)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.15
        run = p.add_run(f"{i}) ")
        set_run(run, size=10)
        run = p.add_run(c)
        set_run(run, size=10)
    justified_paragraph(doc, segments=[
        ("The CNN is trained for at most 80 epochs using Adam with an initial "
         "learning rate of 10⁻³ and an early-stopping criterion on "
         "validation accuracy with patience 10 and best-weights restoration. "
         "Additive white Gaussian noise at 40 dB SNR is applied online to each "
         "training batch. After training, the softmax head is removed and the "
         "network is retained as a fixed feature extractor.", {}),
    ])

    subsubsection_heading(doc, "3", "Random Forest head")
    justified_paragraph(doc, segments=[
        ("A Random Forest with ", {}),
        ("T", {"italic": True}),
        (" = 200 decision trees and max_features = log₂(", {}),
        ("d", {"italic": True}),
        (") where ", {}),
        ("d", {"italic": True}),
        (" = 64 is trained on the CNN feature vectors. At inference, the "
         "prediction is the majority vote", {}),
    ])
    equation_line(doc, segments=[
        ("ŷ = mode( ŷ", {}),
        ("1", {"subscript": True}),
        (", ŷ", {}),
        ("2", {"subscript": True}),
        (", …, ŷ", {}),
        ("T", {"italic": True, "subscript": True}),
        (" ),", {}),
    ], number=12)
    justified_paragraph(doc, segments=[
        ("and the class probability is the empirical frequency of each label "
         "across the trees. This formulation provides a calibrated top-K "
         "distribution, which the dashboard uses to display the top three "
         "candidate classes.", {}),
    ])

    subsubsection_heading(doc, "4", "Rationale for the hybrid configuration")
    justified_paragraph(doc, segments=[
        ("A CNN with a softmax output head converges to comparable accuracy on "
         "this dataset. The hybrid configuration is preferred for two reasons. "
         "First, the Random Forest provides a calibrated top-K probability "
         "distribution without further post-hoc calibration steps such as "
         "temperature scaling or Platt scaling. Second, the modular separation "
         "of the feature extractor and the classifier permits the Random "
         "Forest head to be re-tuned on the frozen feature space within "
         "seconds, without re-training the CNN. This property is operationally "
         "advantageous for field deployments that require periodic drift "
         "adaptation or rapid re-validation.", {}),
    ])

    subsection_heading(doc, "F", "Knowledge-Base Enrichment")
    justified_paragraph(doc, segments=[
        ("Each Stage 2 prediction ŷ is augmented at run time with five "
         "fields drawn from a 17-entry knowledge base derived from IEEE "
         "Standard 1159: ", {}),
        ("display name", {"italic": True}),
        (" (human-readable label), ", {}),
        ("severity", {"italic": True}),
        (" ∈ {none, low, medium, high, critical}, ", {}),
        ("cause", {"italic": True}),
        (" (a two-sentence physical explanation), ", {}),
        ("equipment-at-risk", {"italic": True}),
        (" (the loads typically affected), and ", {}),
        ("immediate actions", {"italic": True}),
        (" (three to five operator-level steps). The enrichment is a "
         "deterministic lookup rather than a model output, but it transforms "
         "the bare class label into actionable diagnostic content.", {}),
    ])

    subsection_heading(doc, "G", "Hyperparameter Optimization")
    justified_paragraph(doc, segments=[
        ("The Stage 2 Random Forest hyperparameters are optimized using a "
         "five-fold stratified cross-validation grid search over the search "
         "space summarized in Table II. Twenty-four parameter combinations "
         "were evaluated, yielding 120 fits in total (24 × 5). The "
         "deployed configuration uses 200 trees with log₂ feature "
         "selection and reaches a held-out test accuracy of 98.85%. The "
         "grid-search optimum (500 trees with sqrt feature selection) reaches "
         "98.88%, an absolute improvement of 0.03 percentage points. The top "
         "thirteen of the 24 configurations are observed to lie within "
         "±0.05 percentage points of one another, indicating that "
         "classification accuracy is saturated on the deployed 64-dimensional "
         "CNN feature space.", {}),
    ])

    booktab_table(
        doc,
        headers=["Hyperparameter", "Search space", "Deployed value"],
        rows=[
            ["n_estimators",      "{100, 200, 500}", "200"],
            ["max_depth",         "{None, 20}",      "None"],
            ["max_features",      "{sqrt, log2}",    "log2"],
            ["min_samples_leaf",  "{1, 2}",          "1"],
        ],
        col_widths=[Inches(1.3), Inches(1.0), Inches(0.85)],
        table_label="Table II",
        table_caption=(
            "Stage 2 Random Forest hyperparameter grid search. The deployed "
            "configuration is retained because the grid-search optimum exceeds "
            "it by less than the test-set noise floor."
        ),
    )

    subsection_heading(doc, "H", "Computational Complexity and Real-Time Considerations")
    justified_paragraph(doc, segments=[
        ("For an ", {}),
        ("N", {"italic": True}),
        ("-sample window the computational cost of each stage is as follows.", {}),
    ])
    cc = [
        ("Stage 1.", "Each of the five features is O(N), with the FFT-based "
         "total harmonic distortion approximation requiring O(N log N) on a "
         "fixed-size buffer. The total measured latency is below 1 ms for "
         "N = 100 on a single-thread CPU."),
        ("Stage 2 CNN forward pass.", "The forward pass is dominated by the "
         "three Conv1D layers. The measured mean latency is approximately "
         "40 ms."),
        ("Stage 2 Random Forest predict.", "The cost is O(T · d_max) for "
         "T = 200 trees on the 64-dimensional feature vector. The measured "
         "mean latency is approximately 26 ms."),
        ("End-to-end Abnormal path.", "Approximately 67 ms per window, "
         "equivalent to approximately three cycles of the 50 Hz fundamental."),
    ]
    for label, body in cc:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Inches(0.3)
        pf.first_line_indent = Inches(-0.2)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.15
        run = p.add_run("• ")
        set_run(run, size=10)
        run = p.add_run(label + " ")
        set_run(run, bold=True, size=10)
        run = p.add_run(body)
        set_run(run, size=10)

    justified_paragraph(doc, segments=[
        ("The pipeline is engineered for single-thread CPU inference. The deep "
         "learning framework's graph caches are warmed at backend start-up "
         "using 50 dummy windows, ensuring that the first live inference "
         "matches steady-state latency. The dashboard maintains a "
         "synthetic-mode fallback so that user-interface demonstration is "
         "independent of backend reachability. The intended deployment "
         "surface is illustrated in Fig. 2: voltage and current signals from "
         "the grid, solar inverters, and distributed generation are acquired "
         "through a sensing front-end (e.g., ZMPT101B and ACS712 modules), "
         "routed to an edge compute node (Raspberry Pi, ESP32, or ARM-class "
         "microcontroller) that executes the trained model, and forwarded "
         "over RS485/Modbus or SocketIO to a panel meter or a SCADA "
         "human–machine interface for operator review.", {}),
    ])

    figure_image(
        doc, FIG_DIR / "deployment.png", width_inches=3.3,
        caption_text=(
            "Fig. 2. Practical deployment architecture. Multi-source signal "
            "acquisition is followed by a sensing front-end, edge-compute "
            "inference using the proposed pipeline, and an RS485 or SocketIO "
            "link to the monitoring human–machine interface."
        ),
    )

    # ============================ VI. REAL-TIME BACKEND =====================
    section_heading(doc, "VI", "Real-Time Backend and Dashboard")
    justified_paragraph(doc, segments=[
        ("The classifier is encapsulated within a Flask and Flask-SocketIO "
         "server. Three input paths are exposed: a JSON POST /api/predict "
         "endpoint for one-shot HTTP queries, a SocketIO predict event for "
         "low-latency clients, and a TCP listener on port 5555 that accepts "
         "newline-delimited JSON frames of the form "
         "{label, signal:[v0..v99], window}. The TCP listener broadcasts "
         "each prediction as a SocketIO pqd_result event to all connected "
         "dashboards. Both the included MATLAB sender and the Python sender, "
         "the latter implementing realistic synthetic IEEE 1159 signal "
         "generation, employ this path.", {}),
    ])

    figure_image(
        doc, FIG_DIR / "dashboard.png", width_inches=3.3,
        caption_text=(
            "Fig. 3. Live dashboard. Six panels are rendered: waveform and "
            "spectrum, signal parameters, the classification card with "
            "top-three candidates, cause and equipment-at-risk, recommended "
            "operator actions, and a system-statistics strip with a rolling "
            "latency timeline. The dashboard automatically falls back to a "
            "built-in demonstration mode if the backend is unreachable for "
            "three seconds."
        ),
    )

    justified_paragraph(doc, segments=[
        ("The dashboard, illustrated in Fig. 3, is implemented as a "
         "single-file HTML/CSS/JavaScript page using Chart.js and Socket.IO. "
         "Six panels are rendered: the live waveform with a tabbed FFT "
         "spectrum view, a gauge cluster for RMS, total harmonic distortion, "
         "crest factor, and peak amplitude, a classification card displaying "
         "a severity badge and top-three confidence bars, a card containing "
         "the cause and equipment-at-risk fields, a list of the recommended "
         "operator actions, and a system-statistics strip showing the number "
         "of frames received, both stage latencies, and a rolling latency "
         "timeline.", {}),
    ])

    # ============================ VII. EXPERIMENTAL SETUP ===================
    section_heading(doc, "VII", "Experimental Setup")

    subsection_heading(doc, "A", "Hardware")
    justified_paragraph(doc, segments=[
        ("All measurements are conducted on a single laptop-class workstation: "
         "Windows on AMD64 with an Intel 8-thread CPU. Single-thread "
         "evaluation is enforced. No GPU is used, and no cloud service is "
         "contacted at inference time.", {}),
    ])

    subsection_heading(doc, "B", "Test Workload")
    justified_paragraph(doc, segments=[
        ("Inference latency is measured on the full 3,400-sample test set, "
         "with one sample per call. A 50-iteration warm-up is applied to all "
         "neural-network models to eliminate graph-caching effects from the "
         "timed window. The Random Forest classifier is warmed by a single "
         "forward pass.", {}),
    ])

    subsection_heading(doc, "C", "Performance Metrics")
    justified_paragraph(doc, segments=[
        ("Reported metrics include overall accuracy, per-class precision, "
         "recall, F1 score, the confusion matrix, latency at the 50th and "
         "95th percentiles and at the mean, parameter count, and on-disk "
         "model size. End-to-end pipeline accuracy is reported on a "
         "510-signal balanced mixed-traffic test composed of 30% Normal and "
         "70% Abnormal traffic.", {}),
    ])

    # ============================ VIII. MODEL COMPARISON ====================
    section_heading(doc, "VIII", "Model Comparison and Selection")
    justified_paragraph(doc, segments=[
        ("Prior to adoption of the hybrid CNN-RF design, six classical "
         "machine-learning algorithms were trained on the same 36-feature "
         "multi-domain vector under identical conditions. The algorithms "
         "considered were Logistic Regression, Support Vector Machine (SVM), "
         "k-Nearest Neighbours (kNN), Decision Tree, Gradient Boosting, and "
         "Random Forest. Performance on the XPQRS dataset and on a second PQ "
         "dataset (the PQ Disturbances Dataset, with pre-extracted features) "
         "is reported in Table III.", {}),
    ])

    booktab_table(
        doc,
        headers=["Model", "XPQRS Acc.", "PQ Acc.", "XPQRS F1", "PQ F1"],
        rows=[
            ["Gradient Boosting",   "0.9112", "0.9837", "0.9107", "0.9836"],
            ["Random Forest",       "0.9062", "0.9812", "0.9056", "0.9816"],
            ["Decision Tree",       "0.8650", "0.9625", "0.8646", "0.9620"],
            ["Logistic Regression", "0.8524", "0.9375", "0.8489", "0.9372"],
            ["SVM",                 "0.8335", "0.8812", "0.8298", "0.8798"],
            ["kNN",                 "0.8006", "0.9062", "0.7955", "0.9046"],
        ],
        col_widths=[Inches(1.0), Inches(0.6), Inches(0.55), Inches(0.6), Inches(0.55)],
        table_label="Table III",
        table_caption=(
            "Comparison of classical machine-learning models on two datasets. "
            "Accuracy and macro F1 are reported on the held-out test split."
        ),
    )

    justified_paragraph(doc, segments=[
        ("The Random Forest classifier provides the most favourable "
         "combination of accuracy, robustness, and deployment simplicity "
         "within this comparison and is therefore adopted as the basis for "
         "both the legacy 36-feature baseline and the Stage 2 head of the "
         "hybrid model. When combined with the learned 64-dimensional CNN "
         "feature representation in place of the 36 hand-engineered features, "
         "the Random Forest head attains 98.85% accuracy on the XPQRS test "
         "set, an absolute improvement of 8.23 percentage points over the "
         "legacy configuration. This improvement is attributed to the richer "
         "learned feature space.", {}),
    ])

    # ============================ IX. RESULTS ===============================
    section_heading(doc, "IX", "Results and Discussion")

    subsection_heading(doc, "A", "Stage 2 Classification Performance")
    justified_paragraph(doc, segments=[
        ("The hybrid 1D-CNN and Random Forest classifier achieves an overall "
         "test accuracy of 98.85% on the held-out 3,400-sample test set. The "
         "test set is exactly balanced; macro-averaged precision, recall, and "
         "F1 are therefore equal at 98.85%. Thirteen of the seventeen classes "
         "attain 100% recall. The four non-perfect classes are Sag (90.50%), "
         "Sag-Flicker (91.00%), Swell (99.50%), and Swell-Flicker (99.50%). "
         "The complete per-class breakdown is reported in Table IV.", {}),
    ])

    perclass_rows = [
        ["Pure_Sinusoidal",       "100.00", "100.00", "100.00"],
        ["Flicker",               "100.00", "100.00", "100.00"],
        ["Harmonics",             "100.00", "100.00", "100.00"],
        ["Harmonics_Flicker",     "100.00", "100.00", "100.00"],
        ["Harmonics_Notch",       "100.00", "100.00", "100.00"],
        ["Interruption",          "100.00", "100.00", "100.00"],
        ["Notch",                 "100.00", "100.00", "100.00"],
        ["Oscillatory_Transient", "100.00", "100.00", "100.00"],
        ["Sag_Harmonics",         "100.00", "100.00", "100.00"],
        ["Sag_Oscillatory",       "100.00", "100.00", "100.00"],
        ["Swell_Harmonics",       "100.00", "100.00", "100.00"],
        ["Swell_Oscillatory",     "100.00", "100.00", "100.00"],
        ["Transient",             "100.00", "100.00", "100.00"],
        ["Swell",                 "99.50",  "99.50",  "99.50"],
        ["Swell_Flicker",         "99.50",  "99.50",  "99.50"],
        ["Sag_Flicker",           "91.00",  "90.55",  "90.77"],
        ["Sag",                   "90.50",  "90.95",  "90.73"],
        [("Macro avg", {"bold": True}),
         ("98.85", {"bold": True}),
         ("98.85", {"bold": True}),
         ("98.85", {"bold": True})],
    ]
    booktab_table(
        doc,
        headers=["Class", "Recall", "Precision", "F1"],
        rows=perclass_rows,
        col_widths=[Inches(1.5), Inches(0.55), Inches(0.6), Inches(0.55)],
        table_label="Table IV",
        table_caption=(
            "Per-class classification performance on the 3,400-sample test "
            "set, with 200 samples per class."
        ),
    )

    subsection_heading(doc, "B", "Confusion Patterns")
    justified_paragraph(doc, segments=[
        ("The misclassification errors are not uniformly distributed across "
         "the off-diagonal of the confusion matrix shown in Fig. 4. The errors "
         "concentrate in a single 2 × 2 block: nineteen Sag windows are "
         "predicted as Sag_Flicker, and eighteen Sag_Flicker windows are "
         "predicted as Sag. Every other off-diagonal entry contains at most "
         "one misclassification. The discriminating feature between Sag and "
         "Sag-with-Flicker is the presence of envelope drift across the "
         "cycle. With ", {}),
        ("N", {"italic": True}),
        (" = 100 samples at 50 Hz, only one full cycle is available, and a "
         "slow flicker modulation may not have produced an observable envelope "
         "variation within that cycle. A multi-cycle context window for these "
         "classes is identified as a natural extension and is reserved for "
         "future work.", {}),
    ])

    figure_image(
        doc, FIG_DIR / "confusion_matrix_hybrid.png", width_inches=3.3,
        caption_text=(
            "Fig. 4. Confusion matrix of the hybrid 1D-CNN and Random Forest "
            "classifier on the 3,400-sample test set. Thirteen of seventeen "
            "classes attain 100% recall, and the residual errors are confined "
            "to a single 2 × 2 block between Sag and Sag_Flicker."
        ),
    )

    subsection_heading(doc, "C", "Input-Normalization Analysis")
    justified_paragraph(doc, segments=[
        ("An initial implementation employed per-sample maximum-absolute "
         "normalization, in which each window is divided by its own peak "
         "amplitude before being supplied to the CNN. This scheme is the "
         "canonical preprocessing step in much of the published deep-learning "
         "PQ literature. Under per-sample normalization, the hybrid model's "
         "test accuracy was observed to be 52.0%, substantially below the "
         "legacy 36-feature Random Forest baseline.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The mechanism underlying this collapse is straightforward. After "
         "per-sample maximum-absolute normalization, a 0.5 pu sag and a 1.0 pu "
         "pure sinusoid both reduce to unit-amplitude sine waves of identical "
         "shape; voltage swells and interruptions are subject to the same "
         "collapse. The amplitude information that distinguishes these "
         "classes is therefore eliminated by the normalizer.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The remedy is to apply a single global scale, computed once on the "
         "training set and stored with the model, in accordance with Eq. (2). "
         "Under this scheme, the test accuracy recovered from 52.0% to "
         "98.85%, an absolute improvement of 47 percentage points obtained "
         "from a single preprocessing modification. This finding is "
         "transferable to any deep-learning PQ pipeline that includes "
         "amplitude-distinguished classes.", {}),
    ])

    subsection_heading(doc, "D", "End-to-End Pipeline Performance")
    justified_paragraph(doc, segments=[
        ("On the 510-signal mixed-traffic test (30% Normal, 70% Abnormal, "
         "balanced across the sixteen Abnormal classes), the end-to-end "
         "pipeline attains an accuracy of 96.67%. Stage 1 detects 100% of "
         "Normal windows correctly and produces zero false alarms; no "
         "Abnormal window is misclassified as Normal. The accuracy gap "
         "between the Stage 2 result on its own test set (98.85%) and the "
         "end-to-end pipeline (96.67%) is therefore attributable to the "
         "Stage 2 misclassifications on Sag and Sag-Flicker, evaluated over "
         "a smaller and less favourable mix.", {}),
    ])

    subsection_heading(doc, "E", "Latency Budget for Edge Deployment")
    justified_paragraph(doc, segments=[
        ("A practical PQ monitor must deliver a decision before the next "
         "decision-relevant window arrives. Stage 1 of the proposed pipeline "
         "satisfies a sub-cycle deadline for 100% of Normal traffic, "
         "completing in less than one millisecond per window. Stage 2 "
         "requires approximately 67 ms per Abnormal window, equivalent to "
         "approximately three cycles of the 50 Hz fundamental, and is invoked "
         "only on windows that have already been screened as anomalous, "
         "ensuring that the heavy inference path is never exercised on clean "
         "traffic. Table V reports the per-stage latency budget on a "
         "single-thread laptop CPU.", {}),
    ])

    booktab_table(
        doc,
        headers=["Stage", "p50 (ms)", "Mean (ms)", "p95 (ms)"],
        rows=[
            ["Stage 1 (5 features + decision)",   "0.4", "0.4", "0.7"],
            ["Stage 2 — CNN forward pass",   "38",  "40",  "55"],
            ["Stage 2 — Random Forest predict", "24", "26", "38"],
            ["Stage 2 — total (Abnormal path)", "65", "67", "88"],
            [("End-to-end — Normal path",    {"bold": True}),
             ("0.4", {"bold": True}),
             ("0.4", {"bold": True}),
             ("0.7", {"bold": True})],
            [("End-to-end — Abnormal path",  {"bold": True}),
             ("65",  {"bold": True}),
             ("67",  {"bold": True}),
             ("88",  {"bold": True})],
        ],
        col_widths=[Inches(1.7), Inches(0.5), Inches(0.55), Inches(0.5)],
        table_label="Table V",
        table_caption=(
            "Per-stage latency budget on a single-thread laptop CPU. "
            "Measurements are steady-state, post warm-up, without GPU "
            "acceleration or cloud connectivity."
        ),
    )

    justified_paragraph(doc, segments=[
        ("A growing portion of the deep-learning PQ literature relies on "
         "cloud inference in order to accommodate model size. Cloud "
         "deployment, however, introduces a network round-trip-time overhead "
         "before any inference is initiated. Wide-area round-trip times from "
         "a distribution feeder to a regional data centre are typically "
         "100–300 ms even on a clean wired link, and substantially "
         "higher on congested or wireless paths, which is already several "
         "multiples of one PQ cycle. The proposed pipeline performs "
         "detection in less than one millisecond and classification within "
         "approximately three cycles, entirely on a single CPU thread, "
         "without GPU acceleration or external service connectivity. The "
         "proposed system therefore satisfies an operational deadline that "
         "cloud-deployed deep PQ classifiers cannot satisfy under realistic "
         "network conditions, while remaining a deeply learned classifier in "
         "Stage 2.", {}),
    ])

    subsection_heading(doc, "F", "Hyperparameter Sensitivity")
    justified_paragraph(doc, segments=[
        ("A 24-combination, five-fold cross-validation grid search was "
         "conducted over the Stage 2 Random Forest on the frozen CNN feature "
         "space. The top thirteen configurations clustered within ±0.05 "
         "percentage points of one another. The deployed configuration (200 "
         "trees, log₂ feature selection) lies inside this cluster. The "
         "result indicates that classification accuracy is saturated on the "
         "deployed feature space, and that further hyperparameter tuning of "
         "the Random Forest head is unlikely to yield improvements that "
         "exceed the statistical noise of the test split.", {}),
    ])

    subsection_heading(doc, "G", "Per-Prediction Interpretability via SHAP")
    justified_paragraph(doc, segments=[
        ("A TreeSHAP [10] explainer is integrated as a post-hoc attribution "
         "layer on the legacy 36-feature Random Forest baseline. For any "
         "single window, the system returns the top three contributing "
         "features along with their physical units (RMS in pu, total harmonic "
         "distortion in percent, harmonic magnitudes, and wavelet energies) "
         "together with the SHAP magnitude and sign of each contribution. A "
         "representative explanation for a Sag prediction identifies RMS as "
         "the dominant feature with a negative SHAP value, together with two "
         "amplitude-correlated wavelet energies. Operators are therefore able "
         "to review borderline classifications using named physical quantities "
         "rather than abstract activation maps. Broader surveys of explainable "
         "artificial intelligence are provided in [11].", {}),
    ])

    subsection_heading(doc, "H", "Discussion")
    justified_paragraph(doc, segments=[
        ("Three observations summarize the principal results. First, the "
         "separation of detection and classification through the Stage 1 and "
         "Stage 2 architecture provides a real-time monitor that satisfies a "
         "sub-cycle deadline for clean traffic and applies a heavier hybrid "
         "classifier only when warranted, avoiding the engineering trade-offs "
         "of either purely shallow or purely deep designs. Second, the "
         "learned 64-dimensional feature representation produced by the "
         "1D-CNN closes the accuracy gap that hand-engineered features leave "
         "open on compound-disturbance classes; an absolute gain of 8.23 "
         "percentage points is observed when the learned feature space is "
         "substituted for the 36 hand-engineered features under the same "
         "Random Forest head, while the Random Forest itself preserves "
         "calibrated top-K outputs and supports rapid retunability. Third, "
         "the proposed system executes end-to-end on a laptop-class CPU "
         "without GPU acceleration or external service connectivity, at a "
         "latency below the round-trip-time floor of typical cloud links; "
         "the deployability claim is therefore an experimental result rather "
         "than an aspirational one.", {}),
    ])

    # ============================ X. LIMITATIONS ============================
    section_heading(doc, "X", "Limitations")
    justified_paragraph(doc, segments=[
        ("The limitations of the present study are noted below.", {}),
    ])
    limits = [
        ("All measurements are obtained on a Windows AMD64 laptop using "
         "single-thread, single-sample evaluation. Embedded-hardware "
         "deployment (ARM, FPGA, microcontroller) is identified as a planned "
         "extension and is not claimed in the present study."),
        ("The dataset is synthetic, generated under the IEEE 1159 class "
         "definitions with additive white Gaussian noise at 40 dB SNR. Real "
         "grid waveforms contain additional noise structure (sub-harmonic "
         "content, inter-harmonics, and higher-order non-stationarities) that "
         "is absent from the training set."),
        ("The Stage 1 thresholds are hand-set rather than learned. The "
         "thresholds are configured conservatively, admitting some Normal "
         "windows into Stage 2 in exchange for a 0% Abnormal-as-Normal "
         "misclassification rate. A learned screen is identified as a natural "
         "extension."),
        ("The knowledge-base entries are derived from the IEEE 1159 reference "
         "text and do not currently incorporate utility-specific operational "
         "practice. Per-deployment customization would require additional "
         "curation."),
    ]
    for i, c in enumerate(limits, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Inches(0.4)
        pf.first_line_indent = Inches(-0.25)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.15
        run = p.add_run(f"{i}) ")
        set_run(run, size=10)
        run = p.add_run(c)
        set_run(run, size=10)

    # ============================ XI. CONCLUSION ============================
    section_heading(doc, "XI", "Conclusion and Future Work")
    justified_paragraph(doc, segments=[
        ("This paper has presented a real-time PQ disturbance classification "
         "system based on a two-stage pipeline. A five-feature O(N) threshold "
         "detector (Stage 1) handles clean traffic in less than one "
         "millisecond and forwards only flagged windows to a hybrid 1D-CNN "
         "and Random Forest classifier (Stage 2), which attains 98.85% "
         "accuracy across seventeen IEEE 1159 classes (thirteen of seventeen "
         "at 100% recall) and operates in approximately 67 ms per call. The "
         "end-to-end pipeline attains 96.67% accuracy on a balanced "
         "mixed-traffic test, and the system streams its predictions, each "
         "enriched online with severity, cause, equipment-at-risk, and "
         "recommended operator actions, to a browser dashboard via a "
         "Flask-SocketIO backend. The full pipeline executes on a laptop CPU "
         "without GPU acceleration or external network connectivity, "
         "completing detection in less than one millisecond and "
         "classification in approximately three cycles of the 50 Hz "
         "fundamental, below the round-trip-time floor of cloud-deployed "
         "deep PQ classifiers. The proposed system is therefore directly "
         "suitable for edge deployment on distribution feeders.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("The most actionable single finding of the present study is the "
         "input normalization result: per-sample maximum-absolute scaling, "
         "which is the default in much of the deep-learning PQ literature, "
         "eliminates the amplitude information that distinguishes Sag, "
         "Swell, and Interruption from Pure Sinusoidal, and incurs a "
         "47-percentage-point accuracy penalty on the seventeen-class "
         "problem. A single global-scale normalization derived from the "
         "training distribution restores the missing accuracy.", {}),
    ])
    justified_paragraph(doc, segments=[
        ("Several directions for future work are immediate: (i) replacement "
         "of the hand-set Stage 1 thresholds with a learned screening rule; "
         "(ii) introduction of a multi-cycle context window targeted at the "
         "Sag-versus-Sag-Flicker confusability; (iii) embedded-hardware "
         "benchmarks of the Stage 2 model on ARM, ESP32, and Raspberry-Pi "
         "class targets; and (iv) integration of the SocketIO event stream "
         "with SCADA and Modbus systems to close the loop between prediction "
         "and operator action.", {}),
    ])

    # ============================ REFERENCES ================================
    section_heading(doc, "", "References")
    refs = [
        ("IEEE Recommended Practice for Monitoring Electric Power Quality, "
         "IEEE Std 1159-2019 (Revision of IEEE Std 1159-2009), Institute of "
         "Electrical and Electronics Engineers, 2019, pp. 1–98. "
         "doi: 10.1109/IEEESTD.2019.8796486."),
        ("P. Khetarpal and M. M. Tripathi, “A critical and comprehensive "
         "review on power quality disturbance detection and classification,” "
         "Sustainable Computing: Informatics and Systems, vol. 28, p. 100417, "
         "2020. doi: 10.1016/j.suscom.2020.100417."),
        ("K. Thirumala, S. Pal, T. Jain, and A. C. Umarikar, “A "
         "classification method for multiple power quality disturbances using "
         "EWT based adaptive filtering and multiclass SVM,” "
         "Neurocomputing, vol. 334, pp. 265–274, 2019. "
         "doi: 10.1016/j.neucom.2019.01.038."),
        ("M. Sahani and P. K. Dash, “Automatic power quality events "
         "recognition based on Hilbert–Huang transform and weighted "
         "bidirectional extreme learning machine,” IEEE Trans. Ind. "
         "Informat., vol. 17, no. 2, pp. 1090–1100, 2021. "
         "doi: 10.1109/TII.2020.2989128."),
        ("J. Wang, Z. Xu, and Y. Che, “Power quality disturbance "
         "classification based on compressed sensing and deep convolutional "
         "neural networks,” IEEE Access, vol. 8, pp. 78336–78346, "
         "2020. doi: 10.1109/ACCESS.2020.2992803."),
        ("K. Cai, W. Cao, L. Aarniovuori, H. Pang, Y. Lin, and G. Li, "
         "“Classification of power quality disturbances using "
         "Wigner–Ville distribution and deep convolutional neural "
         "networks,” IEEE Access, vol. 8, pp. 117400–117409, 2020. "
         "doi: 10.1109/ACCESS.2020.3004984."),
        ("B. Eristi, O. Yildirim, H. Eristi, and Y. Demir, “A robust "
         "power quality disturbance classification method using convolutional "
         "neural networks and 1D local binary pattern,” Electric Power "
         "Systems Research, vol. 193, p. 107015, 2021. "
         "doi: 10.1016/j.epsr.2020.107015."),
        ("K. Cai, T. Hu, W. Cao, T. D. Memon, Y. Wang, and W. Liu, “A "
         "new hybrid convolutional neural network and extreme gradient "
         "boosting classifier for recognizing power quality disturbances,” "
         "IEEE Access, vol. 9, pp. 91017–91030, 2021. "
         "doi: 10.1109/ACCESS.2021.3091267."),
        ("C. I. Garcia, F. Grasso, A. Luchetta, M. C. Piccirilli, "
         "L. Paolucci, and G. Talluri, “A comparison of power quality "
         "disturbance detection and classification methods using CNN, LSTM "
         "and CNN-LSTM,” Applied Sciences, vol. 10, no. 19, p. 6755, "
         "2020. doi: 10.3390/app10196755."),
        ("S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, "
         "B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "
         "“From local explanations to global understanding with "
         "explainable AI for trees,” Nature Machine Intelligence, "
         "vol. 2, no. 1, pp. 56–67, 2020. "
         "doi: 10.1038/s42256-019-0138-9."),
        ("P. Linardatos, V. Papastefanopoulos, and S. Kotsiantis, "
         "“Explainable AI: A review of machine learning interpretability "
         "methods,” Entropy, vol. 23, no. 1, p. 18, 2021. "
         "doi: 10.3390/e23010018."),
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Inches(0.30)
        pf.first_line_indent = Inches(-0.30)
        pf.space_after = Pt(3)
        pf.line_spacing = 1.15
        run = p.add_run(f"[{i}] ")
        set_run(run, size=9)
        run = p.add_run(r)
        set_run(run, size=9)

    # Save
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
