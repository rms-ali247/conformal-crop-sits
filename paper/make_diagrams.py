"""Redraw the two schematic figures at full text width with real point-size type.

The earlier versions came from the .d2 sources kept alongside in figs/. Those
exports laid their labels out at a few pixels against a canvas thousands of
pixels wide, so once scaled into the paper the text landed near 4pt and was not
readable. Here the canvas is the IEEE text width in inches and the fonts are
given in points, so what the script sets is what the page shows.

Writes figs/method_flow.png and figs/phenossm_arch.png.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figs"
TEXTW = 7.16          # \textwidth of an IEEEtran conference page, inches
MARGIN = 0.04
PAD_X, PAD_Y = 0.085, 0.075

BLUE_L = ("#eef2f7", "#5b7aa5")
BLUE_M = ("#dbe4f0", "#3f5b82")
BLUE_D = ("#c9d6ea", "#3f5b82")
GROUP = ("#f8fafd", "#3f5b82")
GREEN = ("#cfe8d6", "#3f7a52")
SAND = ("#f6f0e2", "#9a894f")
ROSE = ("#f6e2e2", "#a1595b")
INK = "#1a1a1a"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def canvas(height):
    fig = plt.figure(figsize=(TEXTW, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXTW)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    return fig, ax


def measure(fig, s, size, weight="normal"):
    """Width and height of a rendered string, in inches."""
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def box_size(fig, lines, size, lead=1.42, nbold=1):
    w = max(measure(fig, s, size, "bold" if i < nbold else "normal")[0]
            for i, s in enumerate(lines))
    h = len(lines) * size * lead / 72.0
    return w + 2 * PAD_X, h + 2 * PAD_Y


def draw_box(ax, cx, cy, w, h, lines, palette, size, dashed=False, lead=1.42,
             nbold=1):
    fill, stroke = palette
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.055",
        facecolor=fill, edgecolor=stroke, linewidth=0.9,
        linestyle=(0, (2.6, 1.6)) if dashed else "solid", zorder=2))
    step = size * lead / 72.0
    top = cy + (len(lines) - 1) * step / 2
    for i, s in enumerate(lines):
        ax.text(cx, top - i * step, s, ha="center", va="center", zorder=3,
                fontsize=size, fontweight="bold" if i < nbold else "normal")


def arrow(ax, p0, p1, dashed=False, rad=0.0, color="#2f4f79"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
        color=color, shrinkA=0, shrinkB=0, zorder=1,
        linestyle=(0, (2.6, 1.6)) if dashed else "solid",
        connectionstyle="arc3,rad=" + str(rad)))


def edge_label(ax, x, y, s, size=7.0):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, style="italic",
            color="#555555", zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def spread(widths, gap_min=0.18):
    """Left-to-right centres for a row of boxes filling the text width."""
    usable = TEXTW - 2 * MARGIN
    gap = (usable - sum(widths)) / (len(widths) - 1)
    assert gap >= gap_min, "row too wide: gap %.3f in" % gap
    xs, x = [], MARGIN
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    return xs, gap


# ----------------------------------------------------------------- pipeline
def method_flow():
    size, H = 9.0, 2.02
    fig, ax = canvas(H)

    chain = [
        (["Sentinel-2 SITS", "7 months × 28", "features per field"], BLUE_L, 1),
        (["PhenoSSM", "backbone", "multi-scale conv,", "S4D + attention"], BLUE_M, 2),
        (["Out-of-fold scores", "APS nonconformity"], BLUE_L, 1),
        (["Conformal", "calibration", "region-robust,", "class-conditional"], BLUE_M, 2),
        (["Guaranteed", "prediction sets"], GREEN, 1),
    ]
    sizes = [box_size(fig, lines, size, nbold=nb) for lines, _, nb in chain]
    hmax = max(h for _, h in sizes)
    xs, gap = spread([w for w, _ in sizes])
    cy = H - 0.10 - hmax / 2

    for (lines, pal, nb), (w, _), cx in zip(chain, sizes, xs):
        draw_box(ax, cx, cy, w, hmax, lines, pal, size, nbold=nb)
    for i in range(4):
        x0 = xs[i] + sizes[i][0] / 2
        arrow(ax, (x0, cy), (x0 + gap, cy))

    model_cx = xs[1]
    feeders = [
        (["Spatial protocol", "district-grouped CV,", "held-out districts"],
         SAND, "disjoint regions"),
        (["Conformal training", "set-size penalty"], ROSE, "tightens sets"),
    ]
    fsizes = [box_size(fig, lines, size) for lines, _, _ in feeders]
    fh = max(h for _, h in fsizes)
    fy = 0.12 + fh / 2
    fxs = [MARGIN + fsizes[0][0] / 2,
           MARGIN + fsizes[0][0] + 0.60 + fsizes[1][0] / 2]

    y0, y1 = fy + fh / 2, cy - hmax / 2
    for (lines, pal, lab), (w, _), cx in zip(feeders, fsizes, fxs):
        draw_box(ax, cx, fy, w, fh, lines, pal, size, dashed=True)
        arrow(ax, (cx, y0), (model_cx, y1), dashed=True, rad=-0.16)
        edge_label(ax, cx + 0.46 * (model_cx - cx), y0 + 0.34 * (y1 - y0), lab)

    fig.savefig(OUT / "method_flow.png", dpi=400)
    plt.close(fig)
    print("saved method_flow.png  %.2f x %.2f in" % (TEXTW, H))


# ------------------------------------------------------------------ PhenoSSM
def phenossm_arch():
    size, inner, tsize = 8.5, 7.8, 8.0
    gap_v, pad_g = 0.11, 0.12

    inp = ["Input", "B × T=7 × C=28"]
    att = ["Temporal", "attention pooling", "seq → (B, d)"]
    out = ["Dropout + linear", "crop logits:", "5 Rabi / 7 Kharif"]

    front = [["Parallel Conv1D", "k = 2, 3, 5"], ["concat + BN", "+ residual"]]
    s4d = [["diagonal SSM", "HiPPO-LegS init,", "FFT long conv"],
           ["GLU + residual", "+ LayerNorm"]]
    titles = [["Multi-scale", "front-end"], ["S4D backbone", "(× n layers)"]]

    # Size everything first, then make a canvas exactly tall enough.
    tmp = plt.figure(figsize=(TEXTW, 2.0))
    tmp.canvas.draw()
    plain = [box_size(tmp, l, size) for l in (inp, att, out)]
    fsz = [box_size(tmp, l, inner) for l in front]
    ssz = [box_size(tmp, l, inner) for l in s4d]
    tsz = [box_size(tmp, l, tsize) for l in titles]
    plt.close(tmp)

    groups = ((fsz, tsz[0]), (ssz, tsz[1]))
    gw = [max(max(w for w, _ in grp), tw) + 0.18 for grp, (tw, _) in groups]
    gh = [sum(h for _, h in grp) + (len(grp) - 1) * gap_v + th + pad_g
          for grp, (_, th) in groups]
    ph = max(h for _, h in plain)
    H = max(max(gh), ph) + 0.16

    fig, ax = canvas(H)
    xs, _ = spread([plain[0][0], gw[0], gw[1], plain[1][0], plain[2][0]],
                   gap_min=0.17)
    gcy = H / 2

    for cx, w, h, tl, (_, th) in zip(xs[1:3], gw, gh, titles, tsz):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, gcy - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=GROUP[0], edgecolor=GROUP[1], linewidth=0.9,
            linestyle=(0, (3.2, 2.0)), zorder=0))
        step = tsize * 1.42 / 72.0
        for i, s in enumerate(tl):
            ax.text(cx, gcy + h / 2 - pad_g / 2 - (i + 0.5) * step, s,
                    ha="center", va="center", fontsize=tsize,
                    fontweight="bold", zorder=3)

    def stack(cx, width, items, sizes, th, gtop):
        y, cys = gtop - th - pad_g / 2, []
        for i, (l, (_, h)) in enumerate(zip(items, sizes)):
            pal = BLUE_D if i == len(items) - 1 else BLUE_M
            draw_box(ax, cx, y - h / 2, width, h, l, pal, inner)
            cys.append((y - h / 2, h))
            y -= h + gap_v
        for (cy, hh), (ny, nh) in zip(cys[:-1], cys[1:]):
            arrow(ax, (cx, cy - hh / 2), (cx, ny + nh / 2))

    stack(xs[1], gw[0] - 0.18, front, fsz, tsz[0][1], gcy + gh[0] / 2)
    stack(xs[2], gw[1] - 0.18, s4d, ssz, tsz[1][1], gcy + gh[1] / 2)

    for cx, l, pal, (w, _) in ((xs[0], inp, BLUE_L, plain[0]),
                               (xs[3], att, BLUE_M, plain[1]),
                               (xs[4], out, GREEN, plain[2])):
        draw_box(ax, cx, gcy, w, ph, l, pal, size)

    # No edge labels here: the gaps between the tall group frames are too
    # narrow to hold one without it landing on a neighbouring box.
    for x0, x1 in [(xs[0] + plain[0][0] / 2, xs[1] - gw[0] / 2),
                   (xs[1] + gw[0] / 2, xs[2] - gw[1] / 2),
                   (xs[2] + gw[1] / 2, xs[3] - plain[1][0] / 2),
                   (xs[3] + plain[1][0] / 2, xs[4] - plain[2][0] / 2)]:
        arrow(ax, (x0, gcy), (x1, gcy))

    fig.savefig(OUT / "phenossm_arch.png", dpi=400)
    plt.close(fig)
    print("saved phenossm_arch.png  %.2f x %.2f in" % (TEXTW, H))


if __name__ == "__main__":
    method_flow()
    phenossm_arch()
