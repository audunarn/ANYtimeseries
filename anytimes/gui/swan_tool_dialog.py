"""SWAN tool dialog for DNORA post-processing workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import subprocess
import sys
import webbrowser
from typing import Iterable

import numpy as np
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anytimes.gui import postprocess_dnora_source as swan_post


@dataclass(frozen=True)
class Poi:
    lat: float
    lon: float

    @property
    def label(self) -> str:
        return f"({self.lat:.6f}, {self.lon:.6f})"


@dataclass(frozen=True)
class PreviewLayer:
    source_nc: Path
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray | None
    land_mask: np.ndarray | None
    resolution_score: float


class SWANToolDialog(QMainWindow):
    """Interactive SWAN post-processing UI wrapper."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # Create as proper top-level window (min/max/titlebar) regardless of parent.
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("SWANtool")
        self.resize(1300, 780)

        self._preview_layers: list[PreviewLayer] = []
        self._preview_nc_path: Path | None = None
        self._is_syncing_point = False

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        split.addWidget(right)
        split.setSizes([530, 770])

        self._build_folder_group(left_layout)
        self._build_poi_group(left_layout)
        self._build_parameter_group(left_layout)
        self._build_action_row(left_layout)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("SWANtool output...")
        left_layout.addWidget(self.log_output, stretch=1)

        self._build_map(right_layout)

    def _build_folder_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("1) Input folders")
        vbox = QVBoxLayout(group)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vbox.addWidget(self.folder_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add folder(s)")
        remove_btn = QPushButton("Remove selected")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        vbox.addLayout(btn_row)

        add_btn.clicked.connect(self._add_folders)
        remove_btn.clicked.connect(self._remove_selected_folders)

        layout.addWidget(group)

    def _build_poi_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("2) Points of interest (POI)")
        vbox = QVBoxLayout(group)

        form = QFormLayout()
        self.poi_lat = QLineEdit()
        self.poi_lon = QLineEdit()
        self.poi_lat.setPlaceholderText("e.g. 65.010180")
        self.poi_lon.setPlaceholderText("e.g. 11.746670")
        form.addRow("Latitude", self.poi_lat)
        form.addRow("Longitude", self.poi_lon)
        vbox.addLayout(form)

        row = QHBoxLayout()
        add_btn = QPushButton("Add POI")
        add_csv_btn = QPushButton("Add POI(s) from file")
        del_btn = QPushButton("Delete selected POI")
        row.addWidget(add_btn)
        row.addWidget(add_csv_btn)
        row.addWidget(del_btn)
        vbox.addLayout(row)

        self.poi_list = QListWidget()
        self.poi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vbox.addWidget(self.poi_list)

        self.poi_lat.textChanged.connect(self._on_manual_point_changed)
        self.poi_lon.textChanged.connect(self._on_manual_point_changed)
        add_btn.clicked.connect(self._add_poi)
        add_csv_btn.clicked.connect(self._add_poi_from_file)
        del_btn.clicked.connect(self._delete_poi)
        self.poi_list.itemSelectionChanged.connect(self._refresh_map)

        layout.addWidget(group)

    def _build_parameter_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("3) Parameters")
        form = QFormLayout(group)

        self.split_report_cb = QCheckBox("Split report files")
        self.split_report_cb.setChecked(True)
        form.addRow(self.split_report_cb)

        self.arrow_resolution = QDoubleSpinBox()
        self.arrow_resolution.setDecimals(0)
        self.arrow_resolution.setRange(1, 10000)
        self.arrow_resolution.setValue(100)
        form.addRow("Arrow resolution", self.arrow_resolution)

        self.theta_step = QDoubleSpinBox()
        self.theta_step.setRange(0.1, 360)
        self.theta_step.setValue(5.0)
        form.addRow("SPEC_DIR_THETA_STEP_DEG", self.theta_step)

        self.spreading_s = QDoubleSpinBox()
        self.spreading_s.setRange(0.1, 1000)
        self.spreading_s.setValue(5.0)
        form.addRow("SPEC_DIR_SPREADING_S", self.spreading_s)

        layout.addWidget(group)

    def _build_action_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run postprocessing (open plots)")
        self.save_btn = QPushButton("Save output")
        row.addWidget(self.run_btn)
        row.addWidget(self.save_btn)
        self.run_btn.clicked.connect(lambda: self._run(save_output=False))
        self.save_btn.clicked.connect(lambda: self._run(save_output=True))
        layout.addLayout(row)

    def _build_map(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Map preview from selected .nc region")
        vbox = QVBoxLayout(group)
        self.map_info = QLabel("Add/select input folders to load a .nc region preview.")
        vbox.addWidget(self.map_info)

        self.map_fig = Figure(figsize=(7, 6), facecolor="white")
        self.map_canvas = FigureCanvasQTAgg(self.map_fig)
        self.map_canvas.setStyleSheet("background: white;")
        self.map_toolbar = NavigationToolbar2QT(self.map_canvas, self)
        self.map_canvas.mpl_connect("button_press_event", self._on_map_clicked)
        vbox.addWidget(self.map_toolbar)
        vbox.addWidget(self.map_canvas)
        layout.addWidget(group)

    def _add_folders(self) -> None:
        while True:
            chosen = QFileDialog.getExistingDirectory(self, "Select SWAN result folder")
            if not chosen:
                break
            if not any(self.folder_list.item(i).text() == chosen for i in range(self.folder_list.count())):
                self.folder_list.addItem(chosen)
                self._log(f"Added folder: {chosen}")
            else:
                self._log(f"Folder already added: {chosen}")
            again = QMessageBox.question(
                self,
                "Add another?",
                "Add another folder?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if again != QMessageBox.Yes:
                break
        self._load_region_preview()

    def _remove_selected_folders(self) -> None:
        for item in self.folder_list.selectedItems():
            self._log(f"Removed folder: {item.text()}")
            self.folder_list.takeItem(self.folder_list.row(item))
        self._load_region_preview()

    def _add_poi(self) -> None:
        poi = self._current_manual_poi()
        if poi is None:
            QMessageBox.warning(self, "Invalid POI", "Please provide valid numeric latitude and longitude.")
            return
        self.poi_list.addItem(QListWidgetItem(poi.label))
        self._log(f"Added POI: {poi.label}")
        self._log_current_pois()
        self._refresh_map()

    def _delete_poi(self) -> None:
        selected = self.poi_list.selectedItems()
        if not selected:
            return
        for item in selected:
            self._log(f"Deleted POI: {item.text()}")
            self.poi_list.takeItem(self.poi_list.row(item))
        self._log_current_pois()
        self._refresh_map()

    def _add_poi_from_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select POI file (CSV/Excel)",
            str(Path.cwd()),
            "POI files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*)",
        )
        if not filepath:
            return

        path = Path(filepath)
        if not path.exists():
            QMessageBox.warning(self, "CSV not found", f"File does not exist:\n{path}")
            return

        try:
            rows, lat_col, lon_col = self._load_poi_table(path)

            added = 0
            skipped = 0
            for idx, row in enumerate(rows, start=2):
                lat_raw = str(row.get(lat_col, "")).strip()
                lon_raw = str(row.get(lon_col, "")).strip()
                if not lat_raw or not lon_raw or lat_raw.lower() == "nan" or lon_raw.lower() == "nan":
                    skipped += 1
                    self._log(f"Skipped row {idx}: missing lat/lon.")
                    continue
                try:
                    poi = Poi(lat=float(lat_raw), lon=float(lon_raw))
                except ValueError:
                    skipped += 1
                    self._log(f"Skipped row {idx}: invalid numeric values lat='{lat_raw}', lon='{lon_raw}'.")
                    continue

                label = poi.label
                if any(self.poi_list.item(i).text() == label for i in range(self.poi_list.count())):
                    skipped += 1
                    self._log(f"Skipped row {idx}: duplicate POI {label}.")
                    continue

                self.poi_list.addItem(QListWidgetItem(label))
                added += 1

            self._log(f"POI import complete: added={added}, skipped={skipped} from {path.name}")
            self._log_current_pois()
            self._refresh_map()
        except Exception as exc:
            QMessageBox.warning(self, "POI import failed", str(exc))

    def _load_poi_table(self, path: Path) -> tuple[list[dict[str, object]], str, str]:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            try:
                import pandas as pd
            except Exception as exc:
                raise RuntimeError("Excel import requires pandas/openpyxl installed.") from exc
            frame = pd.read_excel(path)
            rows = frame.to_dict(orient="records")
            columns = [str(c) for c in frame.columns]
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV file has no header row.")
                rows = list(reader)
                columns = [name for name in reader.fieldnames if name is not None]

        normalized = {name.strip().lower(): name for name in columns if name is not None}
        lat_col = next((normalized[k] for k in ("lat", "latitude") if k in normalized), None)
        lon_col = next((normalized[k] for k in ("lon", "longitude") if k in normalized), None)
        if lat_col is None or lon_col is None:
            raise ValueError(
                "Could not find latitude/longitude columns. "
                "Expected one of: lat/latitude and lon/longitude."
            )
        return rows, lat_col, lon_col

    def _on_manual_point_changed(self) -> None:
        if self._is_syncing_point:
            return
        self._refresh_map()

    def _on_map_clicked(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        self._is_syncing_point = True
        try:
            self.poi_lat.setText(f"{event.ydata:.6f}")
            self.poi_lon.setText(f"{event.xdata:.6f}")
        finally:
            self._is_syncing_point = False
        self._refresh_map()

    def _current_manual_poi(self) -> Poi | None:
        try:
            lat = float(self.poi_lat.text().strip())
            lon = float(self.poi_lon.text().strip())
        except ValueError:
            return None
        return Poi(lat=lat, lon=lon)

    def _poi_values(self) -> list[Poi]:
        values: list[Poi] = []
        for i in range(self.poi_list.count()):
            txt = self.poi_list.item(i).text().strip().strip("()")
            lat_s, lon_s = [x.strip() for x in txt.split(",", maxsplit=1)]
            values.append(Poi(lat=float(lat_s), lon=float(lon_s)))
        return values

    def _folder_paths(self) -> list[Path]:
        return [Path(self.folder_list.item(i).text()) for i in range(self.folder_list.count())]

    def _load_region_preview(self) -> None:
        self._preview_layers = []
        self._preview_nc_path = None

        loaded_layers: list[PreviewLayer] = []
        for folder in self._folder_paths():
            try:
                nc = self._autodetect_preview_nc(folder)
                lat, lon, land_mask, depth_grid = self._read_preview_data_from_nc(nc)
                if lat is None or lon is None:
                    continue
                loaded_layers.append(
                    PreviewLayer(
                        source_nc=nc,
                        lat=lat,
                        lon=lon,
                        depth=depth_grid,
                        land_mask=land_mask,
                        resolution_score=self._grid_resolution_score(lat, lon),
                    )
                )
                self._log(f"Map preview source added: {nc}")
            except Exception as exc:
                self._log(f"Preview skip for {folder}: {exc}")

        if loaded_layers:
            # Draw coarse first and fine last => finest takes precedence visually.
            self._preview_layers = sorted(loaded_layers, key=lambda layer: layer.resolution_score, reverse=True)
            self._preview_nc_path = self._preview_layers[-1].source_nc
            self.map_info.setText(
                f"Preview from {len(self._preview_layers)} folder(s); finest overlay: {self._preview_nc_path.name}"
            )
        else:
            self.map_info.setText("No valid .nc file found for region preview.")

        self._refresh_map()

    def _grid_resolution_score(self, lat: np.ndarray, lon: np.ndarray) -> float:
        def _axis_spacing(arr: np.ndarray) -> float:
            vals = np.asarray(arr, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                return float("inf")
            diffs = np.abs(np.diff(np.unique(vals)))
            diffs = diffs[diffs > 0]
            if diffs.size == 0:
                return float("inf")
            return float(np.nanmedian(diffs))

        if lat.ndim == 2 and lon.ndim == 2:
            dlat = _axis_spacing(lat[:, 0])
            dlon = _axis_spacing(lon[0, :])
        else:
            dlat = _axis_spacing(lat)
            dlon = _axis_spacing(lon)
        return dlat * dlon

    def _autodetect_preview_nc(self, folder: Path) -> Path:
        candidates = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".nc" and "spec" not in p.name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"No non-spec .nc file found in {folder}")
        return sorted(candidates, key=lambda p: (-p.stat().st_size, p.name.lower()))[0]

    def _read_preview_data_from_nc(
        self, nc_path: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        # Local import keeps this function robust in bundled/runtime environments
        # where module globals may not be initialized as expected.
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            lat = self._pick_coord(ds, ("lat", "latitude", "LAT", "nav_lat", "y"))
            lon = self._pick_coord(ds, ("lon", "longitude", "LON", "nav_lon", "x"))
            if lat is None or lon is None:
                lat, lon = self._fallback_lat_lon_from_dims(ds)
            if lat is None or lon is None:
                return None, None, None, None
            lat_vals = np.asarray(lat.values)
            lon_vals = np.asarray(lon.values)

            if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
            elif lat_vals.shape == lon_vals.shape:
                lat_grid, lon_grid = lat_vals, lon_vals
            else:
                return None, None, None, None

            depth_grid = self._extract_depth_grid(ds, lat_grid.shape)
            land_mask = self._extract_land_mask(ds, lat_grid.shape, depth_grid=depth_grid)
            return lat_grid, lon_grid, land_mask, depth_grid

    def _fallback_lat_lon_from_dims(self, ds: xr.Dataset):
        lat_dim = next((d for d in ("lat", "latitude", "y") if d in ds.dims), None)
        lon_dim = next((d for d in ("lon", "longitude", "x") if d in ds.dims), None)
        if lat_dim is None or lon_dim is None:
            return None, None
        if lat_dim not in ds.coords or lon_dim not in ds.coords:
            return None, None
        return ds.coords[lat_dim], ds.coords[lon_dim]

    def _extract_depth_grid(self, ds: xr.Dataset, target_shape: tuple[int, ...]) -> np.ndarray | None:
        depth_candidates = ("depth", "DEPTH", "bathymetry", "h", "topo")
        for name in depth_candidates:
            if name in ds:
                arr = self._to_2d_array(np.asarray(ds[name].values))
                if arr is not None and arr.shape == target_shape:
                    return arr.astype(float)
        return None

    def _extract_land_mask(
        self,
        ds: xr.Dataset,
        target_shape: tuple[int, ...],
        depth_grid: np.ndarray | None = None,
    ) -> np.ndarray | None:
        # Preferred behavior: derive land mask from depth (0 at WL, positive downward).
        if depth_grid is not None and depth_grid.shape == target_shape:
            return ~np.isfinite(depth_grid) | (depth_grid <= 0.0)

        mask_candidates = (
            "land_mask",
            "mask",
            "LANDMASK",
            "wetmask",
            "wetdry_mask",
            "sea_mask",
        )
        for name in mask_candidates:
            if name in ds:
                arr = self._to_2d_array(np.asarray(ds[name].values))
                if arr is not None and arr.shape == target_shape:
                    # Normalize to boolean land mask when possible.
                    return arr > 0.5

        # Fallback: infer land from NaN coverage in Hs.
        hs = self._pick_data_var(ds, getattr(swan_post, "HS_CANDIDATES", ("hs", "swh", "Hsig")))
        if hs is not None:
            arr = np.asarray(hs.values)
            if "time" in hs.dims and arr.ndim >= 3:
                arr2d = self._to_2d_array(arr[0])
            else:
                arr2d = self._to_2d_array(arr)
            if arr2d is not None and arr2d.shape == target_shape:
                return ~np.isfinite(arr2d)

        return None

    def _to_2d_array(self, arr: np.ndarray) -> np.ndarray | None:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[0]
        return None

    def _pick_data_var(self, ds: xr.Dataset, names: Iterable[str]):
        for name in names:
            if name in ds:
                return ds[name]
        return None

    def _pick_coord(self, ds: xr.Dataset, names: Iterable[str]):
        for name in names:
            if name in ds.coords:
                return ds.coords[name]
            if name in ds:
                return ds[name]
        return None

    def _refresh_map(self) -> None:
        self.map_fig.clear()
        ax = self.map_fig.add_subplot(111)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f7fbff")

        if self._preview_layers:
            depth_vals: list[np.ndarray] = []
            all_lon: list[np.ndarray] = []
            all_lat: list[np.ndarray] = []
            for layer in self._preview_layers:
                all_lon.append(layer.lon.ravel())
                all_lat.append(layer.lat.ravel())
                if layer.depth is not None:
                    depth_vals.append(np.asarray(layer.depth, dtype=float).ravel())

            if depth_vals:
                depth_concat = np.concatenate(depth_vals)
                depth_concat = depth_concat[np.isfinite(depth_concat)]
                depth_min = float(np.nanmin(depth_concat))
                depth_max = float(np.nanmax(depth_concat))
            else:
                depth_min, depth_max = 0.0, 1.0

            mesh = None
            for layer in self._preview_layers:
                lat = layer.lat
                lon = layer.lon
                depth = layer.depth

                if depth is not None and depth.shape == lat.shape:
                    depth_plot = np.ma.masked_invalid(depth)
                    mesh = ax.pcolormesh(
                        lon,
                        lat,
                        depth_plot,
                        shading="auto",
                        cmap="viridis",
                        vmin=depth_min,
                        vmax=depth_max,
                        alpha=0.90,
                    )
                    if np.nanmin(depth) <= 0 <= np.nanmax(depth):
                        ax.contour(lon, lat, depth, levels=[0.0], colors="cyan", linewidths=1.0)

                if layer.land_mask is not None and layer.land_mask.shape == lat.shape:
                    land_mask = layer.land_mask.astype(float)
                    mask = np.ma.masked_where(land_mask < 0.5, land_mask)
                    ax.pcolormesh(
                        lon,
                        lat,
                        mask,
                        shading="auto",
                        cmap="OrRd",
                        vmin=0.0,
                        vmax=1.0,
                        alpha=0.35,
                    )
                    ax.contour(
                        lon,
                        lat,
                        land_mask,
                        levels=[0.5],
                        colors="black",
                        linewidths=1.0,
                        alpha=0.95,
                    )

            if mesh is not None:
                self.map_fig.colorbar(mesh, ax=ax, label="Depth [m] (positive downward)")

            lon_concat = np.concatenate(all_lon)
            lat_concat = np.concatenate(all_lat)
            lon_concat = lon_concat[np.isfinite(lon_concat)]
            lat_concat = lat_concat[np.isfinite(lat_concat)]
            if lon_concat.size and lat_concat.size:
                ax.set_xlim(np.nanmin(lon_concat), np.nanmax(lon_concat))
                ax.set_ylim(np.nanmin(lat_concat), np.nanmax(lat_concat))

        manual = self._current_manual_poi()
        if manual is not None:
            ax.scatter([manual.lon], [manual.lat], color="red", s=90, marker="x", label="Manual POI")

        pois = self._poi_values()
        if pois:
            ax.scatter([p.lon for p in pois], [p.lat for p in pois], color="orange", s=45, label="POI list")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

        self.map_canvas.draw_idle()

    def _run(self, save_output: bool) -> None:
        folders = self._folder_paths()
        if not folders:
            QMessageBox.warning(self, "No folders", "Please add at least one input folder.")
            return

        pois = self._poi_values()
        if not pois:
            manual = self._current_manual_poi()
            if manual is not None:
                pois = [manual]
                self._log(f"Using manual POI without adding to list: {manual.label}")
            else:
                QMessageBox.warning(
                    self,
                    "No POI",
                    "Please add at least one valid POI (or enter a valid manual latitude/longitude) before running.",
                )
                return
        self._log("Running SWANtool with parameters:")
        self._log(f"  SPLIT_REPORT_FILES={self.split_report_cb.isChecked()}")
        self._log(f"  DEFAULT_ARROW_RESOLUTION={int(self.arrow_resolution.value())}")
        self._log(f"  SPEC_DIR_THETA_STEP_DEG={self.theta_step.value()}")
        self._log(f"  SPEC_DIR_SPREADING_S={self.spreading_s.value()}")
        self._log(f"  Save output requested: {save_output}")

        started_at = self._now_epoch()
        self._run_source_postprocessor(
            folders=folders,
            pois=pois,
            save_output=save_output,
        )
        if not save_output:
            for folder in folders:
                self._open_new_html_outputs(folder, started_at)

    def _run_source_postprocessor(self, folders: list[Path], pois: list[Poi], save_output: bool) -> None:
        script_path = Path(swan_post.__file__).resolve()
        if not script_path.exists():
            self._log(f"Postprocessor script not found: {script_path}")
            return

        cmd = [sys.executable, str(script_path), *(str(folder) for folder in folders)]
        cmd += ["--wind-arrow-resolution", str(int(self.arrow_resolution.value()))]
        cmd += ["--spec-dir-theta-step-deg", str(self.theta_step.value())]
        cmd += ["--spec-dir-spreading-s", str(self.spreading_s.value())]
        # Ensure report "Map overlay" tab is populated (not "No overlay available").
        cmd += ["--export-hs-format", "geojson", "--export-time-index", "MAX"]
        cmd += ["--split-report-files" if self.split_report_cb.isChecked() else "--single-report-file"]
        cmd += ["--no-auto-open-split-files" if save_output else "--auto-open-split-files"]
        for poi in pois:
            cmd += ["--point-lat", f"{poi.lat:.6f}", "--point-lon", f"{poi.lon:.6f}"]
        self._log(f"Executing: {' '.join(cmd)}")
        try:
            env = dict(**os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            subprocess.run(cmd, check=True, cwd=str(folders[0]), env=env)
            mode = "saved (no auto-open)" if save_output else "generated and auto-opened"
            self._log(f"Postprocessor completed; outputs {mode}.")
        except subprocess.CalledProcessError as exc:
            self._log(f"Postprocessor failed: {exc}")

    def _open_new_html_outputs(self, folder: Path, started_at: float) -> None:
        html_files = sorted(
            [p for p in folder.rglob("*.html") if p.is_file() and p.stat().st_mtime >= started_at],
            key=lambda p: p.stat().st_mtime,
        )
        if not html_files:
            self._log(f"No new HTML outputs detected in: {folder}")
            return
        for html in html_files:
            self._log(f"Opening HTML output in browser: {html}")
            webbrowser.open(html.resolve().as_uri())

    def _now_epoch(self) -> float:
        import time
        return time.time()

    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _log_current_pois(self) -> None:
        pois = self._poi_values()
        if not pois:
            self._log("Current POIs: [none]")
            return
        self._log("Current POIs:")
        for i, poi in enumerate(pois, start=1):
            self._log(f"  {i}. {poi.label}")
