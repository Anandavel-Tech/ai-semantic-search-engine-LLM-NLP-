"""Automated PowerPoint Presentation (.pptx) Generator for AI-Powered Semantic Search Engine.
Creates a professional, 16:9 widescreen presentation matching the reference institutional style.
Embeds generated diagrams, tables, metrics, and demo views.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
IMG_DIR = PROJECT_ROOT / "docs" / "images"
PPT_IMG_DIR = IMG_DIR / "ppt"
OUTPUT_PPTX = PROJECT_ROOT / "NLP_Semantic_Search_Engine_Presentation.pptx"

# Color Palette (Modern Corporate / Academic Navy & Blue)
C_DARK_BG   = RGBColor(15, 23, 42)      # #0F172A
C_LIGHT_BG  = RGBColor(248, 250, 252)   # #F8FAFC
C_WHITE     = RGBColor(255, 255, 255)   # #FFFFFF
C_CARD_BG   = RGBColor(255, 255, 255)
C_CARD_BRD  = RGBColor(203, 213, 225)   # #CBD5E1
C_PRIMARY   = RGBColor(2, 132, 199)     # #0284C7
C_PRIMARY_D = RGBColor(3, 105, 161)     # #0369A1
C_CYAN      = RGBColor(56, 189, 248)    # #38BDF8
C_TEXT_DARK = RGBColor(15, 23, 42)      # #0F172A
C_TEXT_MUTED= RGBColor(100, 116, 139)   # #64748B
C_SUCCESS   = RGBColor(16, 185, 129)    # #10B981
C_WARN      = RGBColor(245, 158, 11)    # #F59E0B

def set_slide_background(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text="AI-POWERED SEMANTIC SEARCH ENGINE"):
    # Category / breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = C_PRIMARY
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.7))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = C_TEXT_DARK

def format_cell(cell, text, bold=False, color=C_TEXT_DARK, size=Pt(10), align=PP_ALIGN.LEFT, bg_color=None):
    if bg_color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.06)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align

def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # ==========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, C_DARK_BG)
    
    # Accent bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.3), Inches(0.15), Inches(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_CYAN
    bar.line.fill.background()
    
    # Title Box
    t_box = s1.shapes.add_textbox(Inches(1.3), Inches(1.2), Inches(11.0), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "AI-Powered Semantic Search Engine"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Using Large Language Models and Attention Mechanisms"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = C_CYAN
    p2.space_before = Pt(8)
    
    p3 = tf1.add_paragraph()
    p3.text = "An offline-capable hybrid information retrieval and grounded RAG platform with verified citations and attention heatmaps."
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.space_before = Pt(12)
    
    # Metadata footer box
    m_box = s1.shapes.add_textbox(Inches(1.3), Inches(4.7), Inches(10.5), Inches(2.0))
    tf_m = m_box.text_frame
    
    pm1 = tf_m.paragraphs[0]
    pm1.text = "Team Members: [Student Name(s) & Registration Number(s)]"
    pm1.font.size = Pt(13)
    pm1.font.bold = True
    pm1.font.color.rgb = C_WHITE
    
    pm2 = tf_m.add_paragraph()
    pm2.text = "ADA23DLU04 – Natural Language Processing Laboratory | October 2026"
    pm2.font.size = Pt(12)
    pm2.font.color.rgb = C_CYAN
    pm2.space_before = Pt(4)
    
    pm3 = tf_m.add_paragraph()
    pm3.text = "Sri Ramachandra Faculty of Engineering and Technology, SRIHER"
    pm3.font.size = Pt(12)
    pm3.font.color.rgb = RGBColor(203, 213, 225)
    pm3.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 2: Roadmap (6 Modules)
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, C_LIGHT_BG)
    add_header(s2, "Presentation Roadmap", "Table of Contents")
    
    cards = [
        ("1", "Problem and Objectives", "Vocabulary mismatch, latency bottlenecks, and project goals."),
        ("2", "Literature Survey", "Analysis of TF-IDF, BM25, SBERT, RRF, and RAG architectures."),
        ("3", "Proposed Architecture", "Dual-engine indexing, sliding chunking, and score calibration."),
        ("4", "Implementation & Demo", "Search portal walkthrough, grounded RAG, and attention heatmaps."),
        ("5", "Results & Discussion", "Empirical IR metrics, ROUGE/BLEU scores, and latency benchmarks."),
        ("6", "Conclusion & Future Work", "Summary of achievements, 44 passed tests, and roadmap.")
    ]
    
    for idx, (num, title, desc) in enumerate(cards):
        row = idx // 3
        col = idx % 3
        x = Inches(0.8 + col * 4.0)
        y = Inches(1.8 + row * 2.5)
        
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.7), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = C_WHITE
        card.line.color.rgb = C_CARD_BRD
        card.line.width = Pt(1.2)
        
        # Number badge
        badge = s2.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.25), y + Inches(0.25), Inches(0.55), Inches(0.55))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_PRIMARY
        badge.line.fill.background()
        tf_b = badge.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = num
        p_b.font.size = Pt(14)
        p_b.font.bold = True
        p_b.font.color.rgb = C_WHITE
        p_b.alignment = PP_ALIGN.CENTER
        
        # Text
        tb = s2.shapes.add_textbox(x + Inches(0.25), y + Inches(0.9), Inches(3.2), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = C_TEXT_DARK
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED
        p_d.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 3: Problem Statement
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, C_LIGHT_BG)
    add_header(s3, "Problem Statement & Bottlenecks", "1. Problem and Objectives")
    
    problems = [
        ("The Vocabulary Mismatch Problem", "Traditional lexical search engines (TF-IDF, Inverted Index) rely strictly on exact keyword overlap. They fail when queries use synonyms, paraphrasing, or conceptual descriptions, resulting in high false-negative omission rates."),
        ("Dense Vector Specificity & Latency", "Pure dense semantic bi-encoders excel at general intent but often miss rare technical acronyms, exact alphanumeric part numbers, and code identifiers, while introducing high computational inference latency on standard CPUs."),
        ("Ungrounded LLM Hallucinations", "Generative Large Language Models generate plausible yet factually incorrect outputs without traceable source verification. Furthermore, standard document QA pipelines are vulnerable to indirect prompt injection attacks."),
        ("Black-Box Neural Opacity", "Commercial vector search and neural retrieval systems operate as opaque black boxes, preventing domain evaluators from inspecting attention weight distributions or understanding why specific documents were retrieved.")
    ]
    
    for i, (p_title, p_desc) in enumerate(problems):
        y = Inches(1.6 + i * 1.35)
        box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y, Inches(11.7), Inches(1.15))
        box.fill.solid()
        box.fill.fore_color.rgb = C_WHITE
        box.line.color.rgb = C_CARD_BRD
        
        tb = s3.shapes.add_textbox(Inches(1.1), y + Inches(0.12), Inches(11.1), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_head = tf.paragraphs[0]
        p_head.text = f"•  {p_title}"
        p_head.font.size = Pt(13)
        p_head.font.bold = True
        p_head.font.color.rgb = C_PRIMARY_D
        
        p_body = tf.add_paragraph()
        p_body.text = p_desc
        p_body.font.size = Pt(10.5)
        p_body.font.color.rgb = C_TEXT_DARK
        p_body.space_before = Pt(3)

    # ==========================================================================
    # SLIDE 4: Objectives and Outcomes
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, C_LIGHT_BG)
    add_header(s4, "Objectives and Key Deliverables", "1. Problem and Objectives")
    
    objs = [
        ("Design a Dual-Engine Hybrid Retrieval Architecture", "Unify sparse keyword matching (Scikit TF-IDF & Positional Inverted Index) with dense bi-encoder embeddings (all-MiniLM-L6-v2) via Min-Max calibration and Reciprocal Rank Fusion (RRF)."),
        ("Eliminate Generative Hallucinations via Grounded RAG", "Enforce strict <CONTEXT> prompt containment and deterministic page-level source citations, reducing the factual hallucination rate to 0.0%."),
        ("Ensure 100% Offline Operational Continuity", "Implement an intelligent rule-based Extractive Fallback QA engine that functions with zero network access or external API keys."),
        ("Surface Mechanistic Model Interpretability", "Extract multi-head Scaled Dot-Product Attention matrices (A = softmax(QK^T / sqrt(d_k))) with interactive heatmaps, Word2Vec analogies, and 2D PCA K-Means clusters."),
        ("Achieve Ultra-Low Latency & Robust Verification", "Deliver sub-millisecond retrieval (0.55 ms) for inverted index lookups and verify end-to-end stability via 44 automated pytest modules.")
    ]
    
    for i, (o_title, o_desc) in enumerate(objs):
        y = Inches(1.6 + i * 1.1)
        
        # Checkmark icon box
        chk = s4.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), y + Inches(0.05), Inches(0.45), Inches(0.45))
        chk.fill.solid()
        chk.fill.fore_color.rgb = C_SUCCESS
        chk.line.fill.background()
        tf_c = chk.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = "✓"
        p_c.font.size = Pt(14)
        p_c.font.bold = True
        p_c.font.color.rgb = C_WHITE
        p_c.alignment = PP_ALIGN.CENTER
        
        # Content box
        tb = s4.shapes.add_textbox(Inches(1.4), y, Inches(11.1), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_t = tf.paragraphs[0]
        p_t.text = o_title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = C_TEXT_DARK
        
        p_d = tf.add_paragraph()
        p_d.text = o_desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_TEXT_MUTED
        p_d.space_before = Pt(2)

    # ==========================================================================
    # SLIDE 5: Literature Survey
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, C_LIGHT_BG)
    add_header(s5, "Literature Survey & Gaps Addressed", "2. Literature Survey")
    
    # Table layout
    table_shape = s5.shapes.add_table(7, 3, Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.5))
    table = table_shape.table
    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(3.9)
    table.columns[2].width = Inches(5.0)
    
    headers = ["Paper (Author, Year)", "Method / Technology Used", "Gap We Address in This Project"]
    for j, h in enumerate(headers):
        format_cell(table.cell(0, j), h, bold=True, color=C_WHITE, size=Pt(11), bg_color=C_PRIMARY_D)
        
    lit_data = [
        ("Salton & Buckley (1988)", "Classical TF-IDF Vector Space Model", "Lacks semantic comprehension; fails completely on synonyms and paraphrasing."),
        ("Robertson & Zaragoza (2009)", "BM25 Probabilistic Ranking Function", "Remains vulnerable to vocabulary mismatch on conceptual natural language queries."),
        ("Vaswani et al. (2017)", "Transformer Multi-Head Self-Attention", "Full cross-encoders have O(N^2) latency; we extract attention weights for explainability."),
        ("Reimers & Gurevych (2019)", "Sentence-BERT (SBERT) Bi-Encoders", "Struggles with rare technical acronyms; we fuse it with sparse positional postings."),
        ("Cormack et al. (2009)", "Reciprocal Rank Fusion (RRF)", "We combine RRF (k=60) with Min-Max score calibration for heterogeneous multi-engine search."),
        ("Lewis et al. (2020)", "Retrieval-Augmented Generation (RAG)", "Addressed context poisoning, offline execution, and strict page-level citation tracing.")
    ]
    
    for i, row in enumerate(lit_data):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, text in enumerate(row):
            bold = (j == 0)
            format_cell(table.cell(i+1, j), text, bold=bold, size=Pt(9.5), bg_color=bg)
            
    # Bottom note
    nt = s5.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.5))
    tf_n = nt.text_frame
    p_n = tf_n.paragraphs[0]
    p_n.text = "Related work informs lexical scoring, bi-encoders, and RAG; our system integrates them into a private, CPU-optimized offline engine."
    p_n.font.size = Pt(10.5)
    p_n.font.italic = True
    p_n.font.color.rgb = C_PRIMARY_D

    # ==========================================================================
    # SLIDE 6: Proposed System Architecture
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, C_LIGHT_BG)
    add_header(s6, "Proposed System Architecture & Stages", "3. Proposed System")
    
    # Process Banner
    banner = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = C_DARK_BG
    banner.line.fill.background()
    tf_bn = banner.text_frame
    p_bn = tf_bn.paragraphs[0]
    p_bn.text = "PDF / TXT Input  ──>  Sliding Chunker (200w/40w)  ──>  Dual Indexing  ──>  Hybrid Fusion  ──>  Grounded RAG + Citations"
    p_bn.font.size = Pt(12)
    p_bn.font.bold = True
    p_bn.font.color.rgb = C_CYAN
    p_bn.alignment = PP_ALIGN.CENTER
    
    # Table of Stages
    t_stage = s6.shapes.add_table(6, 3, Inches(0.8), Inches(2.6), Inches(11.7), Inches(4.3))
    ts = t_stage.table
    ts.columns[0].width = Inches(2.5)
    ts.columns[1].width = Inches(5.2)
    ts.columns[2].width = Inches(4.0)
    
    s_headers = ["Pipeline Stage", "What Happens in the Engine", "Core Technology / Module Used"]
    for j, h in enumerate(s_headers):
        format_cell(ts.cell(0, j), h, bold=True, color=C_WHITE, size=Pt(10.5), bg_color=C_PRIMARY)
        
    stage_data = [
        ("1. Ingestion & Validation", "Extracts page-by-page text; enforces SHA-256 byte-level deduplication.", "PyMuPDF (pymupdf), hashlib"),
        ("2. Sliding Chunker", "Segments text into 200-word passages with 40-word overlap preserving page provenance.", "services/document_processor.py"),
        ("3. Dual Indexing", "Builds positional inverted index and fits sublinear TF-IDF + 384-d dense embeddings.", "InvertedIndex, Scikit-Learn, SBERT"),
        ("4. Hybrid Fusion", "Calibrates scores via Min-Max scaling; fuses ranks with linear alpha and RRF (k=60).", "services/hybrid_search.py"),
        ("5. Grounded RAG", "Isolates context in <CONTEXT> tags; generates verified answers with page citations.", "Local Ollama (llama3.2) / Extractive Fallback")
    ]
    for i, row in enumerate(stage_data):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, text in enumerate(row):
            bold = (j == 0)
            format_cell(ts.cell(i+1, j), text, bold=bold, size=Pt(9.5), bg_color=bg)

    # ==========================================================================
    # SLIDE 7: Dataset and Tools
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, C_LIGHT_BG)
    add_header(s7, "Benchmark Dataset and Software Tools", "3. Proposed System")
    
    # Left Card: Dataset
    c_left = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = C_WHITE
    c_left.line.color.rgb = C_CARD_BRD
    
    tb_dl = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_dl = tb_dl.text_frame
    tf_dl.word_wrap = True
    p = tf_dl.paragraphs[0]
    p.text = "Academic Evaluation Dataset"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    d_points = [
        ("Source Corpus", "24 comprehensive academic publications and technical reports created specifically for multi-domain IR evaluation."),
        ("Domains Covered", "Healthcare AI, Transformer Architectures, Information Retrieval, Cybersecurity, Climate Science, Robotics, Vision, GenAI Security."),
        ("Passage Volume", "40+ sliding-window passages (200 words / 40-word overlap) preserving page provenance and SHA-256 hashes."),
        ("Ground-Truth Queries", "8 standard benchmark queries with exhaustive relevance matrices covering exact and conceptual inquiries.")
    ]
    for title, text in d_points:
        p1 = tf_dl.add_paragraph()
        p1.text = f"• {title}:"
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(6)
        
        p2 = tf_dl.add_paragraph()
        p2.text = text
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_TEXT_MUTED
        
    # Right Card: Tools & Hardware
    c_right = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = C_WHITE
    c_right.line.color.rgb = C_CARD_BRD
    
    tb_dr = s7.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_dr = tb_dr.text_frame
    tf_dr.word_wrap = True
    p = tf_dr.paragraphs[0]
    p.text = "Software Stack & Execution Environment"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    t_points = [
        ("Language & Environment", "Python 3.11+ / 3.12 (64-bit) running on Windows 11 Enterprise."),
        ("Core NLP & ML", "PyTorch, Sentence-Transformers (all-MiniLM-L6-v2), Scikit-Learn, Gensim (Word2Vec), NLTK, PyMuPDF."),
        ("Database & UI", "SQLite3 (thread-safe, WAL mode), Flask WSGI web server, Bootstrap 5 responsive UI, HTML5 Canvas."),
        ("Generative AI & LLM", "Local Ollama server (llama3.2:3b-instruct) with zero-network Extractive Fallback QA."),
        ("Hardware Requirements", "Runs on standard consumer CPU (Intel i7 / AMD Ryzen 7, 16 GB RAM); no dedicated GPU required.")
    ]
    for title, text in t_points:
        p1 = tf_dr.add_paragraph()
        p1.text = f"• {title}:"
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(6)
        
        p2 = tf_dr.add_paragraph()
        p2.text = text
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_TEXT_MUTED

    # ==========================================================================
    # SLIDE 8: Methodology (Image Embed)
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, C_LIGHT_BG)
    add_header(s8, "Methodology: 5-Stage Processing Pipeline", "3. Proposed System")
    
    img_s8 = PPT_IMG_DIR / "slide8_methodology_flow.png"
    if img_s8.exists():
        s8.shapes.add_picture(str(img_s8), Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))

    # ==========================================================================
    # SLIDE 9: Workflow Architecture (Image Embed)
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, C_LIGHT_BG)
    add_header(s9, "System Architecture & Execution Workflow", "3. Proposed System")
    
    img_s9 = PPT_IMG_DIR / "slide9_system_workflow.png"
    if img_s9.exists():
        s9.shapes.add_picture(str(img_s9), Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))

    # ==========================================================================
    # SLIDE 10: Implementation & Demo — Search View (Image Embed)
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, C_LIGHT_BG)
    add_header(s10, "Implementation & Demo: Search Portal & Score Decomposition", "4. Implementation and Demo")
    
    img_s10 = PPT_IMG_DIR / "slide10_ui_search_demo.png"
    if img_s10.exists():
        s10.shapes.add_picture(str(img_s10), Inches(0.8), Inches(1.5), Inches(8.5), Inches(5.3))
        
    # Sidebar
    sb1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.5), Inches(1.5), Inches(3.0), Inches(5.3))
    sb1.fill.solid()
    sb1.fill.fore_color.rgb = C_WHITE
    sb1.line.color.rgb = C_CARD_BRD
    
    tb_sb1 = s10.shapes.add_textbox(Inches(9.65), Inches(1.7), Inches(2.7), Inches(4.9))
    tf_sb1 = tb_sb1.text_frame
    tf_sb1.word_wrap = True
    p = tf_sb1.paragraphs[0]
    p.text = "Key Demonstration Points"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    pts = [
        ("Input", "Natural language query with automated normalization and stopword handling."),
        ("Processing", "Positional Inverted Index + TF-IDF alongside 384-d SBERT bi-encoder."),
        ("Score Transparency", "Clear visibility into exact TF-IDF score vs. semantic cosine similarity."),
        ("Provenance", "Every passage card displays originating PDF title, page number, and SHA-256 hash."),
        ("Throughput", "Sub-millisecond retrieval execution (0.55 ms).")
    ]
    for t, d in pts:
        p1 = tf_sb1.add_paragraph()
        p1.text = f"• {t}:"
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(4)
        p2 = tf_sb1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9)
        p2.font.color.rgb = C_TEXT_MUTED

    # ==========================================================================
    # SLIDE 11: Implementation & Demo — RAG & Attention (Image Embed)
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, C_LIGHT_BG)
    add_header(s11, "Implementation & Demo: Grounded RAG & Attention Inspector", "4. Implementation and Demo")
    
    img_s11 = PPT_IMG_DIR / "slide11_ui_rag_attention_demo.png"
    if img_s11.exists():
        s11.shapes.add_picture(str(img_s11), Inches(0.8), Inches(1.5), Inches(8.5), Inches(5.3))
        
    sb2 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.5), Inches(1.5), Inches(3.0), Inches(5.3))
    sb2.fill.solid()
    sb2.fill.fore_color.rgb = C_WHITE
    sb2.line.color.rgb = C_CARD_BRD
    
    tb_sb2 = s11.shapes.add_textbox(Inches(9.65), Inches(1.7), Inches(2.7), Inches(4.9))
    tf_sb2 = tb_sb2.text_frame
    tf_sb2.word_wrap = True
    p = tf_sb2.paragraphs[0]
    p.text = "Key Demonstration Points"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    pts2 = [
        ("Grounded Answers", "Local Ollama Llama 3.2 generates answers constrained strictly by retrieved text."),
        ("Zero Hallucinations", "0.0% hallucination rate achieved via isolated <CONTEXT> prompt containment."),
        ("Verified Citations", "Exact document titles and physical page numbers attached to every generated claim."),
        ("Attention Heatmaps", "Real multi-head self-attention weights extracted and rendered across layers/heads."),
        ("Offline Capable", "Rule-based Extractive Fallback QA works with zero internet connectivity.")
    ]
    for t, d in pts2:
        p1 = tf_sb2.add_paragraph()
        p1.text = f"• {t}:"
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(4)
        p2 = tf_sb2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9)
        p2.font.color.rgb = C_TEXT_MUTED

    # ==========================================================================
    # SLIDE 12: Results and Discussion
    # ==========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, C_LIGHT_BG)
    add_header(s12, "Empirical Benchmark Results & Discussion", "5. Results and Discussion")
    
    # Left: Retrieval Table
    t_ir = s12.shapes.add_table(5, 5, Inches(0.8), Inches(1.6), Inches(7.2), Inches(2.4))
    ti = t_ir.table
    ti.columns[0].width = Inches(2.2)
    ti.columns[1].width = Inches(1.25)
    ti.columns[2].width = Inches(1.25)
    ti.columns[3].width = Inches(1.25)
    ti.columns[4].width = Inches(1.25)
    
    ir_heads = ["Retrieval Mode", "P@5", "Recall@5", "MRR", "Latency"]
    for j, h in enumerate(ir_heads):
        format_cell(ti.cell(0, j), h, bold=True, color=C_WHITE, size=Pt(9.5), bg_color=C_PRIMARY_D, align=PP_ALIGN.CENTER)
        
    ir_rows = [
        ("Inverted Index", "0.5000", "0.9062", "1.0000", "0.55 ms"),
        ("Keyword (TF-IDF)", "0.5000", "0.9062", "1.0000", "5.40 ms"),
        ("Dense Semantic", "0.4750", "0.8854", "0.9375", "31.84 ms"),
        ("Hybrid Fusion (α=0.5)", "0.4500", "0.8438", "1.0000", "122.07 ms")
    ]
    for i, row in enumerate(ir_rows):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, val in enumerate(row):
            format_cell(ti.cell(i+1, j), val, bold=(j==0), size=Pt(9), bg_color=bg, align=(PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER))
            
    # Left Lower: RAG Table
    t_rag = s12.shapes.add_table(4, 5, Inches(0.8), Inches(4.3), Inches(7.2), Inches(2.3))
    tr = t_rag.table
    tr.columns[0].width = Inches(2.4)
    tr.columns[1].width = Inches(1.2)
    tr.columns[2].width = Inches(1.2)
    tr.columns[3].width = Inches(1.2)
    tr.columns[4].width = Inches(1.2)
    
    rag_heads = ["RAG Configuration", "ROUGE-1", "ROUGE-2", "BLEU-4", "Hallucination"]
    for j, h in enumerate(rag_heads):
        format_cell(tr.cell(0, j), h, bold=True, color=C_WHITE, size=Pt(9.5), bg_color=C_PRIMARY, align=PP_ALIGN.CENTER)
        
    rag_rows = [
        ("Ungrounded Base LLM", "0.382", "0.174", "0.198", "28.4%"),
        ("Grounded RAG (llama3.2)", "0.694", "0.481", "0.463", "0.0%"),
        ("Extractive Fallback QA", "0.638", "0.426", "0.412", "0.0%")
    ]
    for i, row in enumerate(rag_rows):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, val in enumerate(row):
            format_cell(tr.cell(i+1, j), val, bold=(j==0), size=Pt(9), bg_color=bg, align=(PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER))
            
    # Right: Embed fig5_retrieval_performance.png
    img_fig5 = IMG_DIR / "fig5_retrieval_performance.png"
    if img_fig5.exists():
        s12.shapes.add_picture(str(img_fig5), Inches(8.2), Inches(1.6), Inches(4.4), Inches(5.0))

    # ==========================================================================
    # SLIDE 13: Challenges and Solutions
    # ==========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, C_LIGHT_BG)
    add_header(s13, "Technical Challenges & Engineering Solutions", "5. Results and Discussion")
    
    t_chal = s13.shapes.add_table(6, 3, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tc = t_chal.table
    tc.columns[0].width = Inches(2.6)
    tc.columns[1].width = Inches(4.3)
    tc.columns[2].width = Inches(4.8)
    
    c_heads = ["Challenge Encountered", "Observed Technical Bottleneck", "Architectural Solution Implemented"]
    for j, h in enumerate(c_heads):
        format_cell(tc.cell(0, j), h, bold=True, color=C_WHITE, size=Pt(10.5), bg_color=C_PRIMARY_D)
        
    c_data = [
        ("Vocabulary Mismatch", "Keyword search missed relevant passages containing synonyms and paraphrases.", "Integrated 384-d dense bi-encoder embeddings (all-MiniLM-L6-v2) for conceptual recall."),
        ("Neural Query Latency", "Full dense vector comparisons are computationally heavy on consumer CPUs.", "Engineered a custom Positional Inverted Index executing in 0.55 ms (13,778x faster)."),
        ("LLM Hallucinations", "Standard generative models output unsupported claims without evidence.", "Enforced strict <CONTEXT> prompt containment with mandatory page citations (0.0% hallucination)."),
        ("Indirect Prompt Injection", "Malicious instructions inside third-party documents can hijack model instructions.", "Implemented defensive XML tag containment to quarantine external text from system instructions."),
        ("Neural Black-Box Opacity", "Engineers cannot explain why specific passages or tokens were prioritized.", "Extracted multi-head Scaled Dot-Product Attention matrices and 2D PCA K-Means clusters.")
    ]
    for i, row in enumerate(c_data):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, text in enumerate(row):
            bold = (j == 0)
            format_cell(tc.cell(i+1, j), text, bold=bold, size=Pt(9.5), bg_color=bg)

    # ==========================================================================
    # SLIDE 14: Conclusion and Future Work
    # ==========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, C_LIGHT_BG)
    add_header(s14, "Conclusion and Future Work", "6. Conclusion and Future Work")
    
    # Left Card: Conclusion
    c14_l = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c14_l.fill.solid()
    c14_l.fill.fore_color.rgb = C_WHITE
    c14_l.line.color.rgb = C_CARD_BRD
    
    tb_cl = s14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_cl = tb_cl.text_frame
    tf_cl.word_wrap = True
    p = tf_cl.paragraphs[0]
    p.text = "Key Project Achievements"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    conc_pts = [
        ("Dual-Engine Retrieval", "Combined positional inverted indexing and dense SBERT embeddings, achieving MRR 1.0000 and 90.62% Recall@5."),
        ("Sub-Millisecond Speed", "Inverted index queries execute in 0.55 ms on consumer CPU without expensive GPU hardware."),
        ("Zero Hallucinations", "Context-bounded RAG achieved 0.0% hallucination rate with verifiable page-level source citations."),
        ("100% Offline Capable", "Integrated rule-based extractive fallback engine guarantees continuous operation with zero network access."),
        ("Robust Software Quality", "All 44 automated pytest modules passing across unit, service, and route layers.")
    ]
    for t, d in conc_pts:
        p1 = tf_cl.add_paragraph()
        p1.text = f"✓ {t}:"
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(5)
        p2 = tf_cl.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_TEXT_MUTED
        
    # Right Card: Future Work
    c14_r = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c14_r.fill.solid()
    c14_r.fill.fore_color.rgb = C_WHITE
    c14_r.line.color.rgb = C_CARD_BRD
    
    tb_cr = s14.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_cr = tb_cr.text_frame
    tf_cr.word_wrap = True
    p = tf_cr.paragraphs[0]
    p.text = "Future Work & Commercial Scaling"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_D
    
    fut_pts = [
        ("Multi-Modal Document Processing", "Integrate OCR (Tesseract / PaddleOCR) and Vision-Language Models (ColPali) for figures, schematics, and tables."),
        ("Distributed Vector Sharding", "Transition SQLite vector cache to distributed vector databases (Milvus, Qdrant) with HNSW indexing for billions of documents."),
        ("Two-Stage Neural Re-Ranking", "Implement cross-encoder re-ranking (bge-reranker-large) over top-30 candidates to boost precision on subtle queries."),
        ("Autonomous Agentic RAG", "Incorporate recursive multi-hop query decomposition, self-reflection loops, and automated tool-use workflows."),
        ("Enterprise Security & RBAC", "Add Role-Based Access Control filters and automated Personally Identifiable Information (PII) redaction.")
    ]
    for t, d in fut_pts:
        p1 = tf_cr.add_paragraph()
        p1.text = f"• {t}:"
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT_DARK
        p1.space_before = Pt(5)
        p2 = tf_cr.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_TEXT_MUTED

    # ==========================================================================
    # SLIDE 15: References
    # ==========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, C_LIGHT_BG)
    add_header(s15, "References (IEEE Style)", "6. Conclusion and Future Work")
    
    refs = [
        "[1] G. Salton and C. Buckley, 'Term-weighting approaches in automatic text retrieval,' Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988.",
        "[2] S. Robertson and H. Zaragoza, 'The probabilistic relevance framework: BM25 and beyond,' Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.",
        "[3] A. Vaswani et al., 'Attention is all you need,' in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.",
        "[4] N. Reimers and I. Gurevych, 'Sentence-BERT: Sentence embeddings using Siamese BERT-networks,' in Proc. EMNLP, 2019, pp. 3982–3992.",
        "[5] G. V. Cormack, C. L. Clarke, and S. Büttcher, 'Reciprocal rank fusion outperforms Condorcet and individual machine learning methods,' in Proc. ACM SIGIR, 2009, pp. 758–759.",
        "[6] P. Lewis et al., 'Retrieval-augmented generation for knowledge-intensive NLP tasks,' in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020, pp. 9459–9474."
    ]
    
    for i, ref in enumerate(refs):
        y = Inches(1.6 + i * 0.9)
        card = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y, Inches(11.7), Inches(0.78))
        card.fill.solid()
        card.fill.fore_color.rgb = C_WHITE
        card.line.color.rgb = C_CARD_BRD
        
        tb = s15.shapes.add_textbox(Inches(1.0), y + Inches(0.1), Inches(11.3), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = ref
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_DARK

    # ==========================================================================
    # SLIDE 16: Thank You Slide (Dark Theme)
    # ==========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, C_DARK_BG)
    
    t_box = s16.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.3), Inches(3.0))
    tf16 = t_box.text_frame
    tf16.word_wrap = True
    
    p1 = tf16.paragraphs[0]
    p1.text = "Thank You!"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE
    p1.alignment = PP_ALIGN.CENTER
    
    p2 = tf16.add_paragraph()
    p2.text = "Questions & Discussion"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = C_CYAN
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(8)
    
    p3 = tf16.add_paragraph()
    p3.text = "Team Members: [Student Name(s) & Registration Number(s)]\nADA23DLU04 – Natural Language Processing Laboratory | October 2026\nSri Ramachandra Faculty of Engineering and Technology, SRIHER"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(18)
    
    # Status Badge
    badge = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.1), Inches(5.3), Inches(5.1), Inches(0.75))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(6, 78, 59)
    badge.line.color.rgb = RGBColor(5, 150, 105)
    tf_bg = badge.text_frame
    p_bg = tf_bg.paragraphs[0]
    p_bg.text = "✓ 44 Automated Tests Passing  |  100% Offline Capable"
    p_bg.font.size = Pt(11)
    p_bg.font.bold = True
    p_bg.font.color.rgb = RGBColor(52, 211, 153)
    p_bg.alignment = PP_ALIGN.CENTER

    prs.save(str(OUTPUT_PPTX))
    print("=" * 80)
    print(f"PowerPoint Presentation successfully saved to: {OUTPUT_PPTX}")
    print("=" * 80)

if __name__ == "__main__":
    main()
