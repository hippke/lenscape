"""Generate lenscape logo SVGs (variant A: fine-ridge Δχ² pixel map). Text is outlined (Inter)."""
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
FONTS = "/usr/share/fonts/opentype/inter/"

AMBER = "#f0a032"
LIGHT = dict(bands=[AMBER, "#7fb0e6", "#3b6fb6", "#24477a", "#142a4a"], ink="#1d3a63", sub="#4a5f7d")
DARK = dict(bands=[AMBER, "#8fbcef", "#4a80c8", "#2f5d99", "#24477a"], ink="#e3ecf8", sub="#9fb2cc")


def band(v):
    return 0 if v < 0.12 else 1 if v < 1 else 2 if v < 3 else 3 if v < 7 else 4


def ridge(x, y):
    return ((x - y) / 0.32) ** 2 + ((x + y) / 1.5) ** 2


def icon(theme, size=256, x0=0, y0=0, n=11, gap=0.14):
    c = size / (n + (n - 1) * gap)
    out = []
    for j in range(n):
        for i in range(n):
            x, y = -1 + 2 * (i + 0.5) / n, 1 - 2 * (j + 0.5) / n
            col = theme["bands"][band(ridge(x, y))]
            px, py = x0 + i * c * (1 + gap), y0 + j * c * (1 + gap)
            out.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{c:.2f}" height="{c:.2f}" '
                       f'rx="{c * 0.18:.2f}" fill="{col}"/>')
    return "\n".join(out)


def text_path(s, font, size, x, y):
    """Outline string s at baseline (x, y); returns (svg path d, advance width)."""
    f = TTFont(FONTS + font)
    gs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
    scale = size / f["head"].unitsPerEm
    pen = SVGPathPen(gs)
    adv = 0
    for ch in s:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (scale, 0, 0, -scale, x + adv * scale, y)))
        adv += hmtx[g][0]
    return pen.getCommands(), adv * scale


def wordmark(theme, x, y, size):
    d1, w1 = text_path("lens", "Inter-SemiBold.otf", size, x, y)
    d2, w2 = text_path("cape", "Inter-SemiBold.otf", size, x + w1, y)
    return (f'<path d="{d1}" fill="{theme["ink"]}"/>\n<path d="{d2}" fill="{AMBER}"/>', w1 + w2)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" role="img" stroke="none"><title>{title}</title>\n{body}\n</svg>\n')


TAGLINE = "Fast Δχ²(log s, log q) maps for planetary microlensing"

for name, th in [("", LIGHT), ("-dark", DARK)]:
    # square icon, transparent
    (OUT / f"lenscape-icon{name}.svg").write_text(svg(256, 256, icon(th), "lenscape"))

    # horizontal logo: icon + wordmark
    H, S = 120, 120
    wm, ww = wordmark(th, S + 28, 86, 72)
    (OUT / f"lenscape-logo{name}.svg").write_text(
        svg(S + 28 + ww + 4, H, icon(th, S) + "\n" + wm, "lenscape"))

    # banner: logo + tagline
    S = 140
    wm, ww = wordmark(th, S + 32, 92, 84)
    tl, tw = text_path(TAGLINE, "Inter-Regular.otf", 22, S + 34, 130)
    W = S + 34 + max(ww, tw) + 8
    (OUT / f"lenscape-banner{name}.svg").write_text(
        svg(W, S, icon(th, S) + "\n" + wm + f'\n<path d="{tl}" fill="{th["sub"]}"/>', "lenscape"))

# avatar: icon on a rounded tile (GitHub / PyPI avatar, survives circular crop)
tile = '<rect width="512" height="512" rx="96" fill="#0f2240"/>\n' + icon(DARK, 300, 106, 106)
(OUT / "lenscape-avatar.svg").write_text(svg(512, 512, tile, "lenscape"))
print("\n".join(sorted(p.name for p in OUT.iterdir())))
