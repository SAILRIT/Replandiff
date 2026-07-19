"""Overlap QA: re-executes plots.py with save() instrumented to draw the canvas
and geometrically test (a) every legend bbox and (b) every free Text/annotation
bbox against every data artist (line vertices, scatter offsets, bar rectangles,
errorbar segments) in display coordinates. Reports violations per figure."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")

REPORT = []

def _bbox_pad(bb, pad=1.0):
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)

def _artist_points(ax):
    """All data-artist points/corners in display coords for one axes."""
    pts = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            pts.append(ln.get_transform().transform(xy))
    for col in ax.collections:                    # scatter + errorbar caps/segs
        try:
            off = col.get_offsets()
            if len(off):
                pts.append(col.get_offset_transform().transform(off))
        except Exception:
            pass
        try:
            for path in col.get_paths():
                v = path.vertices
                if len(v):
                    pts.append(col.get_transform().transform(v))
        except Exception:
            pass
    for p in ax.patches:                          # bars
        bb = p.get_window_extent()
        pts.append(np.array([[bb.x0, bb.y0], [bb.x1, bb.y0],
                             [bb.x0, bb.y1], [bb.x1, bb.y1],
                             [(bb.x0+bb.x1)/2, bb.y1]]))
    return np.vstack(pts) if pts else np.zeros((0, 2))

def _check(fig, name):
    fig.canvas.draw()
    issues = []
    legends = list(fig.legends)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for ax in fig.axes:
        P = _artist_points(ax)
        axbb = ax.get_window_extent()
        for lg in legends:
            lb = _bbox_pad(lg.get_window_extent(), 1.5)
            if lg in fig.legends:                 # outside legend: must not touch axes
                if lb.overlaps(axbb):
                    ov = max(0, min(lb.x1, axbb.x1) - max(lb.x0, axbb.x0)) * \
                         max(0, min(lb.y1, axbb.y1) - max(lb.y0, axbb.y0))
                    if ov > 4:
                        issues.append(f"fig-legend intrudes into axes ({ov:.0f}px2)")
            if len(P):
                inside = ((P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) &
                          (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)).sum()
                if inside > 0:
                    issues.append(f"legend overlaps {inside} data points")
        for txt in ax.texts:                      # annotations inside axes
            if not txt.get_text().strip():
                continue
            tb = _bbox_pad(txt.get_window_extent(), 0.5)
            if len(P):
                inside = ((P[:, 0] > tb.x0) & (P[:, 0] < tb.x1) &
                          (P[:, 1] > tb.y0) & (P[:, 1] < tb.y1)).sum()
                if inside > 0:
                    issues.append(f"text '{txt.get_text()[:14]}' overlaps "
                                  f"{inside} data points")
        # tick labels / axis labels clipped? (rely on tight bbox; check texts fit fig)
    REPORT.append((name, issues))

for fname in ["plots.py", "plots_extra.py"]:
    src = open(fname).read()
    src = src.replace('''def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")''',
'''def save(fig, name):
    _check(fig, name)
    fig.savefig(f"figures/{name}.pdf")''')
    g = {"__name__": "__qa__", "_check": _check}
    exec(src, g)
print("\n===== OVERLAP AUDIT =====")
bad = 0
for name, issues in REPORT:
    if issues:
        bad += 1
        print(f"[FAIL] {name}")
        for i in issues:
            print("   -", i)
    else:
        print(f"[ok]   {name}")
print(f"{len(REPORT)-bad}/{len(REPORT)} figures clean")
