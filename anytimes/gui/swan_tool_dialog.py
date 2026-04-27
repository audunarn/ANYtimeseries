"""SWAN tool dialog for DNORA post-processing workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import postprocess_dnora as swan_post


@dataclass(frozen=True)
class Poi:
    lat: float
    lon: float

    @property
    def label(self) -> str:
        return f"({self.lat:.6f}, {self.lon:.6f})"


class SWANToolDialog(QWidget):
    """Interactive SWAN post-processing UI wrapper."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SWANtool")
        self.resize(1300, 780)

        self._lat_grid: np.ndarray | None = None
        self._lon_grid: np.ndarray | None = None
        self._preview_nc_path: Path | None = None

        root = QVBoxLayout(self)
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
        del_btn = QPushButton("Delete selected POI")
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        vbox.addLayout(row)

        self.poi_list = QListWidget()
        self.poi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vbox.addWidget(self.poi_list)

        self.poi_lat.textChanged.connect(self._refresh_map)
        self.poi_lon.textChanged.connect(self._refresh_map)
        add_btn.clicked.connect(self._add_poi)
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

        self.map_fig = Figure(figsize=(7, 6))
        self.map_canvas = FigureCanvasQTAgg(self.map_fig)
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
        self._lat_grid = None
        self._lon_grid = None
        self._preview_nc_path = None

        for folder in self._folder_paths():
            try:
                nc = swan_post.autodetect_file(folder, ".nc", None)
                lat, lon = self._read_lat_lon_from_nc(nc)
                if lat is None or lon is None:
                    continue
                self._lat_grid = lat
                self._lon_grid = lon
                self._preview_nc_path = nc
                self.map_info.setText(f"Preview from: {nc}")
                self._log(f"Map preview source: {nc}")
                break
            except Exception as exc:
                self._log(f"Preview skip for {folder}: {exc}")

        if self._preview_nc_path is None:
            self.map_info.setText("No valid .nc file found for region preview.")
        self._refresh_map()

    def _read_lat_lon_from_nc(self, nc_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        with xr.open_dataset(nc_path) as ds:
            lat = self._pick_coord(ds, ("lat", "latitude", "LAT", "nav_lat", "y"))
            lon = self._pick_coord(ds, ("lon", "longitude", "LON", "nav_lon", "x"))
            if lat is None or lon is None:
                return None, None
            lat_vals = np.asarray(lat.values)
            lon_vals = np.asarray(lon.values)

            if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
                return lat_grid, lon_grid
            if lat_vals.shape == lon_vals.shape:
                return lat_vals, lon_vals
            return None, None

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

        if self._lat_grid is not None and self._lon_grid is not None:
            lat = self._lat_grid
            lon = self._lon_grid
            ax.scatter(lon.ravel(), lat.ravel(), s=1.0, alpha=0.15, color="tab:blue")
            ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
            ax.set_ylim(np.nanmin(lat), np.nanmax(lat))

        manual = self._current_manual_poi()
        if manual is not None:
            ax.scatter([manual.lon], [manual.lat], color="red", s=80, marker="x", label="Manual POI")

        pois = self._poi_values()
        if pois:
            ax.scatter([p.lon for p in pois], [p.lat for p in pois], color="orange", s=40, label="POI list")

        if manual is not None or pois:
            ax.legend(loc="best")

        self.map_canvas.draw_idle()

    def _run(self, save_output: bool) -> None:
        folders = self._folder_paths()
        if not folders:
            QMessageBox.warning(self, "No folders", "Please add at least one input folder.")
            return

        pois = self._poi_values()
        self._log("Running SWANtool with parameters:")
        self._log(f"  SPLIT_REPORT_FILES={self.split_report_cb.isChecked()}")
        self._log(f"  DEFAULT_ARROW_RESOLUTION={int(self.arrow_resolution.value())}")
        self._log(f"  SPEC_DIR_THETA_STEP_DEG={self.theta_step.value()}")
        self._log(f"  SPEC_DIR_SPREADING_S={self.spreading_s.value()}")
        self._log(f"  Save output requested: {save_output}")

        if save_output:
            QMessageBox.information(
                self,
                "Save output",
                "Save output was requested. The imported postprocessor currently controls actual export behavior.",
            )

        for folder in folders:
            try:
                nc_path = swan_post.autodetect_file(folder, ".nc", None)
            except Exception as exc:
                self._log(f"Skipping folder '{folder}': {exc}")
                continue

            if not pois:
                self._log(f"Plotting default point for {nc_path.name}")
                data = swan_post.load_timeseries(nc_path, point_index=None)
                swan_post.plot_timeseries(data, title=f"SWAN postprocessing — {nc_path.name}")
                continue

            for poi in pois:
                idx = self._nearest_point_index(nc_path, poi)
                self._log(f"Plotting {nc_path.name} at POI {poi.label} (point_index={idx})")
                data = swan_post.load_timeseries(nc_path, point_index=idx)
                swan_post.plot_timeseries(data, title=f"SWAN postprocessing — {nc_path.name} @ {poi.label}")

    def _nearest_point_index(self, nc_path: Path, poi: Poi) -> int | None:
        lat_lon = self._read_lat_lon_from_nc(nc_path)
        lat, lon = lat_lon
        if lat is None or lon is None:
            return None

        dist = np.hypot(lat - poi.lat, lon - poi.lon)
        flat_idx = int(np.nanargmin(dist))
        return flat_idx

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
