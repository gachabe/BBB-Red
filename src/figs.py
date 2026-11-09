"""Utilidades comunes de los análisis 24-27: guardar una figura con su descripción y rellenar bloques del README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import describe_figure

COL = {"GBM": "#c0392b", "LGG": "#2471a3", "otro": "#7f8c8d"}
MARK_COL = {"endotelial": "#27ae60", "mural": "#8e44ad", "inmune": "#e67e22", "glía": "#16a085", "otros": "#bdc3c7"}


def save_fig(fig, outdir, name: str, lines, dpi: int = 130) -> Path:
    """Guarda `<name>.png` en outdir y `descriptions/<name>.txt` con lo que expresa la figura (convención del proyecto)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / (name if name.endswith(".png") else name + ".png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    describe_figure(outdir, p.name, lines)
    return p


def inject_block(readme, tag: str, text: str) -> None:
    """Sustituye lo que hay entre <!-- TAG_START --> y <!-- TAG_END --> del README por `text`."""
    s, e = f"<!-- {tag}_START -->", f"<!-- {tag}_END -->"
    rp = Path(readme)
    txt = rp.read_text(encoding="utf-8")
    assert s in txt and e in txt, f"el README no tiene los marcadores {tag}_START / {tag}_END"
    a, b = txt.index(s) + len(s), txt.index(e)
    rp.write_text(txt[:a] + "\n" + text.strip("\n") + "\n" + txt[b:], encoding="utf-8")
