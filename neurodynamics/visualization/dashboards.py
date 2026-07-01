"""Dashboard building utilities."""
from typing import Any, Dict, List


class Dashboard:
    """Lightweight dashboard that collects panels for later rendering."""

    def __init__(self, title: str = "Neurodynamics Dashboard"):
        self.title = title
        self._panels: List[Dict[str, Any]] = []

    def add_panel(self, name: str, figure: Any,
                  row: int = 0, col: int = 0) -> "Dashboard":
        self._panels.append({"name": name, "figure": figure,
                              "row": row, "col": col})
        return self

    def save(self, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(self._panels)
        if n == 0:
            return
        fig_main, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, panel in zip(axes, self._panels):
            ax.set_title(panel["name"])
            ax.axis("off")
        fig_main.suptitle(self.title)
        fig_main.savefig(path, bbox_inches="tight")
        plt.close(fig_main)

    def show(self) -> None:
        for panel in self._panels:
            if hasattr(panel["figure"], "show"):
                panel["figure"].show()
