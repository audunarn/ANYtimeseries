"""Dialog for running SWAN/DNORA postprocessing workflows."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
)


class SWANpostDialog(QDialog):
    """Collect folders/POIs and run SWAN postprocessing with optional map picking."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SWANpost")
        self.resize(980, 700)

        self._selected_pois: set[int] = set()
        self._depth_points: np.ndarray | None = None
        self._depth_source: Path | None = None

        root = QVBoxLayout(self)
        controls = QGridLayout()

        controls.addWidget(QLabel("Folders to process:"), 0, 0)
        self.folders_list = QListWidget(self)
        self.folders_list.setSelectionMode(QListWidget.ExtendedSelection)
        controls.addWidget(self.folders_list, 1, 0, 4, 2)

        open_folders_btn = QPushButton("Open folders", self)
        remove_folders_btn = QPushButton("Remove selected", self)
        controls.addWidget(open_folders_btn, 5, 0)
        controls.addWidget(remove_folders_btn, 5, 1)

        controls.addWidget(QLabel("Manual POIs (comma-separated indices):"), 0, 2, 1, 2)
        self.poi_input = QLineEdit(self)
        self.poi_input.setPlaceholderText("Example: 0, 12, 45 (blank = no manual POIs)")
        controls.addWidget(self.poi_input, 1, 2, 1, 2)

        self.map_poi_label = QLabel("Map-picked POIs: none", self)
        controls.addWidget(self.map_poi_label, 2, 2, 1, 2)

        clear_pois_btn = QPushButton("Clear map POIs", self)
        controls.addWidget(clear_pois_btn, 3, 2)

        root.addLayout(controls)

        self.figure = Figure(figsize=(8, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run full postprocessing", self)
        cancel_btn = QPushButton("Cancel", self)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(cancel_btn)
        root.addLayout(run_row)

        open_folders_btn.clicked.connect(self._open_folders)
        remove_folders_btn.clicked.connect(self._remove_selected_folders)
        clear_pois_btn.clicked.connect(self._clear_map_pois)
        self.run_btn.clicked.connect(self._run_processing)
        cancel_btn.clicked.connect(self.reject)
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

        self._render_empty_map("Open one or more folders to generate a depth map")

    def _select_multiple_directories(self) -> list[Path]:
        dlg = QFileDialog(self, "Select SWAN folders")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        list_views = dlg.findChildren((QListView, QTreeView))
        for view in list_views:
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dlg.exec():
            return []
        return [Path(p) for p in dlg.selectedFiles()]

    def _open_folders(self) -> None:
        selected = self._select_multiple_directories()
        if not selected:
            return

        existing = {self.folders_list.item(i).text() for i in range(self.folders_list.count())}
        for folder in selected:
            path = str(folder.expanduser().resolve())
            if path not in existing:
                self.folders_list.addItem(path)
                existing.add(path)

        self._refresh_depth_map()

    def _remove_selected_folders(self) -> None:
        for item in self.folders_list.selectedItems():
            self.folders_list.takeItem(self.folders_list.row(item))
        self._refresh_depth_map()

    def _clear_map_pois(self) -> None:
        self._selected_pois.clear()
        self._update_map_poi_label()
        self._plot_depth_map()

    def _update_map_poi_label(self) -> None:
        if not self._selected_pois:
            self.map_poi_label.setText("Map-picked POIs: none")
            return
        formatted = ", ".join(str(v) for v in sorted(self._selected_pois))
        self.map_poi_label.setText(f"Map-picked POIs: {formatted}")

    @staticmethod
    def _parse_bot_file(bot_path: Path) -> np.ndarray:
        rows: list[tuple[float, float, float]] = []
        with bot_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                except ValueError:
                    continue
                rows.append((x, y, z))
        if not rows:
            raise ValueError(f"No numeric XYZ depth rows found in {bot_path.name}")
        return np.asarray(rows, dtype=float)

    def _refresh_depth_map(self) -> None:
        try:
            from ..swanpost import autodetect_file, load_depth_points_from_nc
        except Exception as exc:
            self._render_empty_map(f"SWANpost import failed: {exc}")
            return

        candidates: list[tuple[Path, np.ndarray]] = []
        for i in range(self.folders_list.count()):
            folder = Path(self.folders_list.item(i).text())
            try:
                nc_path = autodetect_file(folder, ".nc")
                depths = load_depth_points_from_nc(nc_path)
                candidates.append((nc_path, depths))
            except Exception:
                try:
                    bot_path = autodetect_file(folder, ".bot")
                    depths = self._parse_bot_file(bot_path)
                    candidates.append((bot_path, depths))
                except Exception:
                    continue

        if not candidates:
            self._depth_points = None
            self._depth_source = None
            self._selected_pois.clear()
            self._update_map_poi_label()
            self._render_empty_map("No readable depth data found in selected folders (.nc depth or .BOT)")
            return

        # Finest map precedence: prefer the depth file with the highest number of points.
        best_bot, best_depths = max(candidates, key=lambda entry: entry[1].shape[0])
        self._depth_source = best_bot
        self._depth_points = best_depths
        self._selected_pois = {idx for idx in self._selected_pois if idx < best_depths.shape[0]}
        self._update_map_poi_label()
        self._plot_depth_map()

    def _render_empty_map(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Depth map")
        self.canvas.draw_idle()

    def _plot_depth_map(self) -> None:
        if self._depth_points is None or self._depth_points.size == 0:
            self._render_empty_map("No depth points available")
            return

        xyz = self._depth_points
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if np.unique(x).size >= 3 and np.unique(y).size >= 3 and len(x) >= 20:
            contour = ax.tricontourf(x, y, z, levels=24, cmap="viridis")
            self.figure.colorbar(contour, ax=ax, label="Depth")
        else:
            scatter = ax.scatter(x, y, c=z, s=12, cmap="viridis")
            self.figure.colorbar(scatter, ax=ax, label="Depth")

        if self._selected_pois:
            indices = np.asarray(sorted(self._selected_pois), dtype=int)
            indices = indices[(indices >= 0) & (indices < len(x))]
            if indices.size:
                ax.scatter(x[indices], y[indices], s=48, c="red", edgecolors="white", linewidths=0.8, label="POIs")
                for idx in indices:
                    ax.text(x[idx], y[idx], str(int(idx)), fontsize=8, color="white")
                ax.legend(loc="upper right")

        source = self._depth_source.name if self._depth_source else "n/a"
        ax.set_title(f"Depth map ({source}) — click to add POIs")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_map_click(self, event) -> None:
        if self._depth_points is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        points = self._depth_points[:, :2]
        click = np.array([event.xdata, event.ydata], dtype=float)
        distances = np.sum((points - click) ** 2, axis=1)
        idx = int(np.argmin(distances))

        if idx in self._selected_pois:
            self._selected_pois.remove(idx)
        else:
            self._selected_pois.add(idx)

        self._update_map_poi_label()
        self._plot_depth_map()

    def _collect_manual_pois(self) -> list[int]:
        raw = self.poi_input.text().strip()
        if not raw:
            return []
        try:
            return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
        except ValueError as exc:
            raise ValueError("Manual POIs must be comma-separated integers.") from exc

    def _run_processing(self) -> None:
        try:
            from ..swanpost import autodetect_file, load_timeseries, plot_timeseries
        except Exception as exc:
            QMessageBox.critical(self, "SWANpost unavailable", f"Could not import SWANpost dependencies: {exc}")
            return

        folders = [Path(self.folders_list.item(i).text()) for i in range(self.folders_list.count())]
        if not folders:
            QMessageBox.warning(self, "No folders", "Open at least one folder before running.")
            return

        try:
            manual_pois = self._collect_manual_pois()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid POIs", str(exc))
            return

        combined_pois = sorted(set(manual_pois).union(self._selected_pois))
        poi_plan: list[int | None] = combined_pois if combined_pois else [None]

        processed = 0
        failures: list[str] = []

        for folder in folders:
            run_dir = Path(os.path.abspath(os.path.expanduser(str(folder))))
            try:
                nc_path = autodetect_file(run_dir, ".nc")
            except Exception as exc:
                failures.append(f"{run_dir}: {exc}")
                continue

            for poi in poi_plan:
                try:
                    data = load_timeseries(nc_path, point_index=poi)
                    poi_label = "none" if poi is None else str(poi)
                    plot_timeseries(data, title=f"SWANpost — {run_dir.name} — POI {poi_label}")
                    processed += 1
                except Exception as exc:
                    poi_label = "none" if poi is None else str(poi)
                    failures.append(f"{run_dir} (POI {poi_label}): {exc}")

        summary = f"Generated {processed} SWANpost plot(s)."
        if failures:
            preview = "\n".join(failures[:8])
            if len(failures) > 8:
                preview += f"\n...and {len(failures) - 8} more."
            QMessageBox.warning(self, "SWANpost finished with warnings", f"{summary}\n\n{preview}")
        else:
            QMessageBox.information(self, "SWANpost finished", summary)
        self.accept()
