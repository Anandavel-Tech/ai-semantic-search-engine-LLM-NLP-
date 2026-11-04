"""Generate an ultra-professional, high-impact LinkedIn showcase graphic for NeuralSearch."""
import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def draw_star(draw, center, size, fill=(250, 204, 21, 255)):
    """Draw a crisp 5-pointed star."""
    cx, cy = center
    points = []
    for i in range(10):
        r = size if i % 2 == 0 else size * 0.45
        angle = i * math.pi / 5 - math.pi / 2
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=fill)

def draw_lightning(draw, x, y, size=18, fill=(255, 255, 255, 255)):
    """Draw a sleek lightning bolt."""
    pts = [
        (x + size * 0.45, y),
        (x, y + size * 0.55),
        (x + size * 0.38, y + size * 0.55),
        (x + size * 0.25, y + size),
        (x + size * 0.85, y + size * 0.38),
        (x + size * 0.48, y + size * 0.38),
    ]
    draw.polygon(pts, fill=fill)

def create_linkedin_banner():
    # 2400 x 1350 (16:9 Ultra HD)
    W, H = 2400, 1350
    img = Image.new("RGBA", (W, H), (10, 14, 23, 255)) # Dark navy slate background
    draw = ImageDraw.Draw(img)

    # 1. Subtle background glow effects & grid
    glow_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_canvas)
    
    # Glow 1: Top-Left Deep Electric Blue
    glow_draw.ellipse([(-250, -250), (1100, 1100)], fill=(37, 99, 235, 60))
    # Glow 2: Center-Right Indigo
    glow_draw.ellipse([(1300, 100), (2600, 1300)], fill=(99, 102, 241, 50))
    # Glow 3: Bottom-Left Teal
    glow_draw.ellipse([(100, 700), (1350, 1700)], fill=(6, 182, 212, 35))
    
    glow_blurred = glow_canvas.filter(ImageFilter.GaussianBlur(150))
    img.paste(glow_blurred, (0, 0), glow_blurred)

    # Subtle modern grid pattern (very faint)
    grid_color = (255, 255, 255, 6)
    for x in range(0, W, 80):
        draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
    for y in range(0, H, 80):
        draw.line([(0, y), (W, y)], fill=grid_color, width=1)

    # Load system fonts
    font_dir = "C:/Windows/Fonts"
    f_bold = os.path.join(font_dir, "segoeuib.ttf")
    f_reg = os.path.join(font_dir, "segoeui.ttf")
    f_mono = os.path.join(font_dir, "consola.ttf")

    title_font = ImageFont.truetype(f_bold, 66)
    subtitle_font = ImageFont.truetype(f_reg, 28)
    tag_font = ImageFont.truetype(f_bold, 20)
    metric_font = ImageFont.truetype(f_bold, 30)
    caption_font = ImageFont.truetype(f_bold, 18)
    caption_sub = ImageFont.truetype(f_reg, 16)
    mono_font = ImageFont.truetype(f_mono, 22)
    url_font = ImageFont.truetype(f_reg, 15)

    # =========================================================================
    # HEADER SECTION
    # =========================================================================
    # Top Bar: Brand Pill
    pill_x, pill_y = 100, 55
    pill_w, pill_h = 350, 48
    draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=24, fill=(24, 34, 53, 230), outline=(59, 130, 246, 200), width=2)
    
    # Mini icon in pill
    draw.ellipse([pill_x + 9, pill_y + 9, pill_x + 39, pill_y + 39], fill=(37, 99, 235, 255))
    draw_lightning(draw, pill_x + 18, pill_y + 14, size=20, fill=(255, 255, 255, 255))
    draw.text((pill_x + 50, pill_y + 11), "ENTERPRISE NLP PLATFORM", font=tag_font, fill=(191, 219, 254, 255))

    # GitHub link badge on top right
    gh_w = 400
    gh_x = W - 100 - gh_w
    draw.rounded_rectangle([gh_x, pill_y, gh_x + gh_w, pill_y + pill_h], radius=24, fill=(15, 23, 42, 230), outline=(71, 85, 105, 255), width=2)
    draw_star(draw, (gh_x + 30, pill_y + 24), size=11, fill=(250, 204, 21, 255))
    draw.text((gh_x + 50, pill_y + 11), "github.com/Anandavel-Tech", font=tag_font, fill=(248, 250, 252, 255))

    # Main Project Title
    draw.text((100, 120), "NeuralSearch Engine", font=title_font, fill=(255, 255, 255, 255))
    
    # Subtitle / Core Description
    draw.text((100, 200), "AI-Powered Semantic Retrieval • Grounded RAG with Citations • Transformer Attention Inspection", font=subtitle_font, fill=(148, 163, 184, 255))

    # Tech Stack Badges
    tech_tags = [
        ("Python 3.12", (59, 130, 246)),
        ("Flask", (148, 163, 184)),
        ("SentenceTransformers (384D)", (16, 185, 129)),
        ("Scikit-Learn TF-IDF", (245, 158, 11)),
        ("PyMuPDF", (239, 68, 68)),
        ("NetworkX", (168, 85, 247)),
        ("Word2Vec", (6, 182, 212)),
        ("SQLite", (99, 102, 241)),
    ]
    tx = 100
    ty = 255
    for tag, color in tech_tags:
        tw = draw.textlength(tag, font=tag_font)
        pw = int(tw + 28)
        draw.rounded_rectangle([tx, ty, tx + pw, ty + 36], radius=8, fill=(15, 23, 42, 200), outline=(51, 65, 85, 220), width=1)
        # Small colored bullet dot
        draw.ellipse([tx + 10, ty + 14, tx + 18, ty + 22], fill=color)
        draw.text((tx + 24, ty + 6), tag, font=tag_font, fill=(226, 232, 240, 255))
        tx += pw + 12

    # =========================================================================
    # BROWSER MOCKUP FRAMES (3 Visual Hero Cards)
    # =========================================================================
    def make_window_mockup(img_path, width, height, title_text="https://neuralsearch.local", crop_top=0):
        src = Image.open(img_path)
        titlebar_h = 44
        content_h = height - titlebar_h
        
        # If crop_top is specified, crop navbar from screenshot
        if crop_top > 0 and src.height > crop_top:
            src = src.crop((0, crop_top, src.width, src.height))
            
        src_fitted = src.resize((width, content_h), Image.Resampling.LANCZOS)
        
        window = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        w_draw = ImageDraw.Draw(window)
        
        # Window background with rounded corners
        w_draw.rounded_rectangle([0, 0, width, height], radius=14, fill=(15, 23, 42, 255), outline=(51, 65, 85, 255), width=2)
        
        # Titlebar
        w_draw.rounded_rectangle([0, 0, width, titlebar_h], radius=14, fill=(30, 41, 59, 255))
        w_draw.rectangle([0, titlebar_h - 14, width, titlebar_h], fill=(30, 41, 59, 255))
        w_draw.line([(0, titlebar_h), (width, titlebar_h)], fill=(51, 65, 85, 255), width=1)
        
        # macOS-style Traffic light dots
        w_draw.ellipse([16, 16, 28, 28], fill=(239, 68, 68, 255))
        w_draw.ellipse([36, 16, 48, 28], fill=(245, 158, 11, 255))
        w_draw.ellipse([56, 16, 68, 28], fill=(16, 185, 129, 255))
        
        # Address bar pill
        addr_w = int(width * 0.50)
        addr_x = (width - addr_w) // 2
        w_draw.rounded_rectangle([addr_x, 8, addr_x + addr_w, 36], radius=6, fill=(15, 23, 42, 220), outline=(51, 65, 85, 200), width=1)
        w_draw.text((addr_x + 16, 11), title_text, font=url_font, fill=(148, 163, 184, 255))
        
        # Paste content into window
        window.paste(src_fitted, (0, titlebar_h))
        
        # Add drop shadow
        shadow_padding = 40
        shadow_w = width + shadow_padding * 2
        shadow_h = height + shadow_padding * 2
        shadow_layer = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(shadow_layer)
        s_draw.rounded_rectangle(
            [shadow_padding + 2, shadow_padding + 10, shadow_padding + width + 2, shadow_padding + height + 10],
            radius=18,
            fill=(0, 0, 0, 185)
        )
        shadow_blurred = shadow_layer.filter(ImageFilter.GaussianBlur(28))
        
        # Composite window on shadow
        composite = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
        composite.paste(shadow_blurred, (0, 0), shadow_blurred)
        composite.paste(window, (shadow_padding, shadow_padding), window)
        
        return composite, shadow_padding

    # Card 1: Main Search Portal (Left Hero Window)
    w1, pad1 = make_window_mockup(
        "docs/screenshots/01_search_portal.png",
        width=1160,
        height=680,
        title_text="neuralsearch.app/search • Dual-Engine Hybrid Retrieval",
        crop_top=75
    )
    img.paste(w1, (100 - pad1, 320 - pad1), w1)

    # Card 2: Grounded RAG Question Answering (Top Right Window)
    w2, pad2 = make_window_mockup(
        "docs/screenshots/03_rag_qa_answer.png",
        width=980,
        height=470,
        title_text="neuralsearch.app/qa • Grounded Answer with Citations",
        crop_top=75
    )
    img.paste(w2, (1320 - pad2, 320 - pad2), w2)

    # Card 3: Linguistic Word Cloud (Bottom Right Window)
    w3, pad3 = make_window_mockup(
        "docs/screenshots/05_visualizations_wordcloud.png",
        width=980,
        height=400,
        title_text="neuralsearch.app/visualizations • Linguistic NLP Lab",
        crop_top=75
    )
    img.paste(w3, (1320 - pad3, 825 - pad3), w3)

    # =========================================================================
    # BOTTOM KEY METRICS & FEATURE CALLOUTS
    # =========================================================================
    # Bottom banner card across bottom left
    bar_x, bar_y = 100, 1030
    bar_w, bar_h = 1160, 240
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=16, fill=(15, 23, 42, 240), outline=(51, 65, 85, 255), width=2)

    # 4 Key Metrics Columns
    metrics = [
        {"val": "Dual-Engine", "label": "Hybrid Retrieval", "sub": "TF-IDF + 384D Embeddings", "color": (59, 130, 246, 255)},
        {"val": "100% Offline", "label": "Grounded RAG", "sub": "Autonomous Fallback QA", "color": (16, 185, 129, 255)},
        {"val": "50 / 50", "label": "Unit & Route Tests", "sub": "100% Automated Pass Rate", "color": (245, 158, 11, 255)},
        {"val": "Multi-Head", "label": "Self-Attention Maps", "sub": "Real Weight Heatmaps", "color": (168, 85, 247, 255)},
    ]
    col_w = bar_w // 4
    for i, m in enumerate(metrics):
        cx = bar_x + i * col_w
        if i > 0:
            draw.line([(cx, bar_y + 25), (cx, bar_y + bar_h - 25)], fill=(51, 65, 85, 180), width=1)
        
        vw = draw.textlength(m["val"], font=metric_font)
        draw.text((cx + (col_w - vw) // 2, bar_y + 45), m["val"], font=metric_font, fill=m["color"])
        
        lw = draw.textlength(m["label"], font=caption_font)
        draw.text((cx + (col_w - lw) // 2, bar_y + 105), m["label"], font=caption_font, fill=(241, 245, 249, 255))

        sw = draw.textlength(m["sub"], font=caption_sub)
        draw.text((cx + (col_w - sw) // 2, bar_y + 145), m["sub"], font=caption_sub, fill=(148, 163, 184, 255))

    # Bottom banner on bottom right (call to action / repository)
    cta_x, cta_y = 1320, 1245
    cta_w, cta_h = 980, 58
    draw.rounded_rectangle([cta_x, cta_y, cta_x + cta_w, cta_y + cta_h], radius=10, fill=(30, 41, 59, 220), outline=(59, 130, 246, 200), width=1)
    
    # GitHub star icon in CTA
    draw_star(draw, (cta_x + 30, cta_y + 29), size=10, fill=(250, 204, 21, 255))
    draw.text((cta_x + 50, cta_y + 16), "github.com/Anandavel-Tech/ai-semantic-search-engine-LLM-NLP-", font=mono_font, fill=(248, 250, 252, 255))

    # Save output banner
    out_dir = "docs/images"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Ultra HD version (2400x1350)
    out_path_uhd = os.path.join(out_dir, "linkedin_project_showcase.png")
    img.save(out_path_uhd, format="PNG", quality=95)
    print(f"UHD Banner generated: {out_path_uhd} ({W}x{H})")

    # 2. LinkedIn Feed Standard (1200x675)
    web_img = img.resize((1200, 675), Image.Resampling.LANCZOS)
    out_path_std = os.path.join(out_dir, "linkedin_project_showcase_1200x675.png")
    web_img.save(out_path_std, format="PNG", quality=95)
    print(f"Standard Banner generated: {out_path_std} (1200x675)")

    # 3. Also save a copy directly in project root for instant access
    root_showcase = "linkedin_showcase.png"
    web_img.save(root_showcase, format="PNG", quality=95)
    print(f"Root copy generated: {root_showcase} (1200x675)")

if __name__ == "__main__":
    create_linkedin_banner()
