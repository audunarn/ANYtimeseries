import numpy as np
import pandas as pd
from PySide6.QtWidgets import QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class PlotUtils:
    @staticmethod
    def _resample(t, y, dt, *, start=None, stop=None):
        """Return ``(t_resampled, y_resampled)`` on a uniform grid.

        ``start`` and ``stop`` may be provided to explicitly set the limits of
        the resampled signal.  If omitted, the limits of ``t`` are used.  The
        function falls back to a NumPy-only implementation when ``qats`` is not
        available.
        """
        if start is None:
            start = t[0]
        if stop is None:
            stop = t[-1]
        if stop < start:
            start, stop = stop, start

        try:
            import anyqats as qats, numpy as _np

            try:
                # Preferred when available
                t_r, y_r = qats.signal.resample(y, t, dt)
                sel = (t_r >= start) & (t_r <= stop)
                t_r, y_r = t_r[sel], y_r[sel]
                if t_r.size == 0 or t_r[0] > start or t_r[-1] < stop:
                    raise ValueError
            except Exception:
                # Fallback to TimeSeries.resample or manual interpolation
                try:
                    ts_tmp = qats.TimeSeries("tmp", t, y)
                    y_r = ts_tmp.resample(dt=dt, t_min=start, t_max=stop)
                    t_r = _np.arange(start, stop + 0.5 * dt, dt)
                except Exception:
                    raise
            return t_r, y_r
        except Exception:
            import numpy as _np
            t_r = _np.arange(start, stop + 0.5 * dt, dt)
            y_r = _np.interp(t_r, t, y)
            return t_r, y_r

    @staticmethod
    def _plot_lines(traces, title, y_label, *, mark_extrema=False):
        """
        traces → list of dicts with keys
                 't', 'y', 'label', 'alpha', 'is_mean'
        """
        engine = "default"

        if engine != "default":
            pass

        import matplotlib.pyplot as plt
        from itertools import cycle
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 5))
        palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for tr in traces:
            color = next(palette)
            ax.plot(
                tr["t"],
                tr["y"],
                label=tr["label"],
                linewidth=2 if tr.get("is_mean") else 1,
                alpha=tr.get("alpha", 1.0),
                color=color,
            )
        if mark_extrema and traces:
            all_t = np.concatenate([np.asarray(tr["t"]) for tr in traces])
            all_y = np.concatenate([np.asarray(tr["y"]) for tr in traces])
            max_idx = np.argmax(all_y)
            min_idx = np.argmin(all_y)
            ax.scatter(all_t[max_idx], all_y[max_idx], color="red", label="Max")
            ax.scatter(all_t[min_idx], all_y[min_idx], color="blue", label="Min")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        fig.tight_layout()

        plt.show()

    @staticmethod
    def _tight_draw(fig, canvas):
        """Redraw canvas with a tight layout.

        Matplotlib requires a draw call before ``tight_layout`` can correctly
        calculate text bounding boxes when embedded in Qt.  Without this the
        axes may be misaligned or labels can be clipped.  Drawing once before
        and after ``tight_layout`` ensures a stable layout across all plots.
        """

        canvas.draw()
        fig.tight_layout()
        canvas.draw()