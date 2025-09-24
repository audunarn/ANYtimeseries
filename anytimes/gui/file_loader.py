"""Loading utilities for timeseries files and OrcaFlex data."""
from __future__ import annotations

import datetime
import os, re
from array import array
from collections.abc import Sequence
import numpy as np
import pandas as pd
from anyqats import TimeSeries, TsDB
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog
)

from .orcaflex_selector import OrcaflexVariableSelector
from .utils import (
    ORCAFLEX_VARIABLE_MAP,
    _matches_terms,
    _parse_search_terms,
)

class FileLoader:
    def __init__(self, orcaflex_varmap=None, parent_gui=None):
        self.orcaflex_varmap = orcaflex_varmap or {}
        self.parent_gui = parent_gui
        self._last_orcaflex_selection = None
        self._reuse_orcaflex_selection = False
        self.loaded_sim_models = {}
        self.orcaflex_redundant_subs = []
        self.progress_callback = None  # called while pre-loading
        self._last_diffraction_dir = None
        self._diffraction_cache = {}

    @property
    def reuse_orcaflex_selection(self):
        """The radius property."""
        return self._reuse_orcaflex_selection

    @reuse_orcaflex_selection.setter
    def reuse_orcaflex_selection(self, value):
        print("Set radius")
        self._reuse_orcaflex_selection = value

    def preload_sim_models(self, filepaths):
        try:
            import OrcFxAPI
        except ImportError:
            print("OrcFxAPI not available. Cannot preload .sim files.")
            return
        total_files = len(filepaths)
        for idx, path in enumerate(filepaths):
            if path not in self.loaded_sim_models:
                try:
                    model = OrcFxAPI.Model(path)
                    self.loaded_sim_models[path] = model
                    print(f"✅ Loaded OrcaFlex model: {os.path.basename(path)}")
                except Exception as e:
                    print(f"❌ Failed to load OrcaFlex model {os.path.basename(path)}:\n{e}")

            if self.progress_callback:
                self.progress_callback(idx + 1, total_files)
            if self.parent_gui:
                QApplication.processEvents()

    def load_files(self, file_paths):
        tsdbs = []
        errors = []
        sim_files = [fp for fp in file_paths if fp.lower().endswith(".sim")]
        other_files = [fp for fp in file_paths if fp not in sim_files]

        tsdb_by_path = {}
        # --- OrcaFlex handling ---
        if sim_files:
            try:

                picked = self.open_orcaflex_picker(sim_files)
                tsdb_by_path.update(picked)

            except Exception as e:
                for fp in sim_files:
                    errors.append((fp, str(e)))
        # --- Other files ---
        for fp in other_files:
            try:
                tsdb_by_path[fp] = self._load_generic_file(fp)
            except Exception as e:
                errors.append((fp, str(e)))

        for fp in file_paths:
            if fp in tsdb_by_path:
                tsdbs.append(tsdb_by_path[fp])

        return tsdbs, errors

    def _load_orcaflex_file(self, filepath):
        import OrcFxAPI
        # Preload model on first use
        if filepath not in self.loaded_sim_models:
            self.loaded_sim_models[filepath] = OrcFxAPI.Model(filepath)
        model = self.loaded_sim_models[filepath]

        # Reuse previous selection?
        if self._reuse_orcaflex_selection and self._last_orcaflex_selection:
            self.orcaflex_redundant_subs = getattr(
                self, "orcaflex_redundant_subs", []
            )
            return self._load_orcaflex_data_from_specs(
                model, self._last_orcaflex_selection
            )

        # Variable/object selection dialog
        selected, redundant, reuse_all = OrcaflexVariableSelector.get_selection(
            model, self.orcaflex_varmap, self.parent_gui
        )
        if not selected:
            return None

        # Convert extras to OrcFxAPI objects
        specs = []
        for obj_name, var, extra_str in selected:
            obj = model[obj_name]
            for extra, label in self._parse_extras(obj, extra_str or ""):
                specs.append((obj, obj_name, var, extra, label))

        self.orcaflex_redundant_subs = redundant or []
        if reuse_all:
            self._last_orcaflex_selection = specs.copy()
            self._reuse_orcaflex_selection = True

        return self._load_orcaflex_data_from_specs(model, specs)

    def open_orcaflex_picker(self, file_paths):
        """Qt version of the OrcaFlex variable picker."""
        import OrcFxAPI

        missing = [fp for fp in file_paths if fp not in self.loaded_sim_models]
        if missing:

            # Lazily preload missing models so the picker can proceed
            self.preload_sim_models(missing)
            remaining = [fp for fp in missing if fp not in self.loaded_sim_models]
            if remaining:
                raise RuntimeError(
                    "Models not preloaded: "
                    + ", ".join(os.path.basename(m) for m in remaining)
                )


        if self._reuse_orcaflex_selection and self._last_orcaflex_selection:
            result = {}
            for fp in file_paths:
                tsdb = self._load_orcaflex_data_from_specs(
                    self.loaded_sim_models[fp],
                    self._last_orcaflex_selection,
                )
                if tsdb:
                    result[fp] = tsdb
            return result

        dialog = QDialog(self.parent_gui)
        dialog.setWindowTitle("Pick OrcaFlex Variables")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        dialog.resize(1150, 820)
        main_layout = QHBoxLayout(dialog)

        file_side = QVBoxLayout()
        right_side = QVBoxLayout()
        main_layout.addLayout(file_side)
        main_layout.addLayout(right_side)

        red_layout = QHBoxLayout()
        red_layout.addWidget(QLabel("Remove (comma-separated):"))
        redundant_entry = QLineEdit()
        red_layout.addWidget(redundant_entry)
        strip_cb = QCheckBox("Strip '_nodes_(x,y)' suffix")
        red_layout.addWidget(strip_cb)
        right_side.addLayout(red_layout)

        wc_layout = QHBoxLayout()
        wc_layout.addWidget(QLabel("Strip rule:"))
        wc_entry = QLineEdit()
        wc_layout.addWidget(wc_entry)
        regex_cb = QCheckBox("Use as REGEX")
        wc_layout.addWidget(regex_cb)
        right_side.addLayout(wc_layout)

        # Default extras for each object type
        default_group = QGroupBox("Default Extra Input")
        def_layout = QGridLayout(default_group)
        default_inputs = {}
        types = ["Line", "Vessel", "Buoy", "Constraint", "Environment"]
        defaults = {
            "Line": "EndA, mid, EndB",
            "Vessel": "0,0,0",
            "Buoy": "0,0,0",
            "Constraint": "0,0,0",
            "Environment": "0,0,0",
        }
        for row, typ in enumerate(types):
            def_layout.addWidget(QLabel(f"{typ}:"), row, 0)
            e = QLineEdit(defaults[typ])
            default_inputs[typ] = e
            def_layout.addWidget(e, row, 1)
        right_side.addWidget(default_group)

        tabs = QTabWidget()
        right_side.addWidget(tabs)

        per_file_state = {}
        panel_pressure_by_file = {}
        for fp in file_paths:
            model = self.loaded_sim_models[fp]
            obj_map = {
                o.Name: (o, self.orcaflex_varmap[o.typeName])
                for o in model.objects
                if o.typeName in self.orcaflex_varmap
            }

            tab = QWidget()
            tabs.addTab(tab, os.path.basename(fp))
            tab_layout = QHBoxLayout(tab)
            left = QVBoxLayout()
            right = QVBoxLayout()
            tab_layout.addLayout(left)
            tab_layout.addLayout(right)

            left.addWidget(QLabel("Objects"))
            obj_filter = QLineEdit()

            obj_filter.setPlaceholderText("Filter objects")

            left.addWidget(obj_filter)
            obj_scroll = QScrollArea()
            obj_widget = QWidget()
            obj_vbox = QVBoxLayout(obj_widget)
            obj_scroll.setWidgetResizable(True)
            obj_scroll.setWidget(obj_widget)
            left.addWidget(obj_scroll)
            obj_vars = {}
            for name in sorted(obj_map):
                cb = QCheckBox(name)
                obj_vbox.addWidget(cb)
                obj_vars[name] = cb

            obj_btns = QHBoxLayout()
            btn_obj_all = QPushButton("Select All Objects")
            btn_obj_none = QPushButton("Unselect All Objects")
            obj_btns.addWidget(btn_obj_all)
            obj_btns.addWidget(btn_obj_none)
            left.addLayout(obj_btns)


            obj_show_cb = QCheckBox("Show only selected")
            left.addWidget(obj_show_cb)


            var_column = QVBoxLayout()
            var_column.addWidget(QLabel("Variables"))
            var_filter = QLineEdit()

            var_filter.setPlaceholderText("Filter variables")

            var_column.addWidget(var_filter)
            var_scroll = QScrollArea()
            var_widget = QWidget()
            var_vbox = QVBoxLayout(var_widget)
            var_scroll.setWidgetResizable(True)
            var_scroll.setWidget(var_widget)
            var_column.addWidget(var_scroll)
            var_vars = {}

            var_btns = QHBoxLayout()
            btn_var_all = QPushButton("Select All Variables")
            btn_var_none = QPushButton("Unselect All Variables")
            var_btns.addWidget(btn_var_all)
            var_btns.addWidget(btn_var_none)
            var_column.addLayout(var_btns)


            var_show_cb = QCheckBox("Show only selected")
            var_column.addWidget(var_show_cb)


            extra_group = QGroupBox("Position / Arc Length")
            extra_layout = QVBoxLayout(extra_group)
            extra_entry = QLineEdit()

            extra_entry.setPlaceholderText("Arc length / position")
            extra_layout.addWidget(extra_entry)
            extra_layout.addWidget(QLabel("Find Closest:"))
            coord_entry = QLineEdit()
            coord_entry.setPlaceholderText("x,y,z; x,y,z ...")

            extra_layout.addWidget(coord_entry)
            find_btn = QPushButton("Find Closest (Selected)")
            extra_layout.addWidget(find_btn)
            skip_entry = QLineEdit()
            skip_entry.setPlaceholderText("Skip names (comma-separated)")
            extra_layout.addWidget(skip_entry)
            find_all_btn = QPushButton("Find Closest (All)")
            extra_layout.addWidget(find_all_btn)
            pressure_btn = QPushButton("Extract Surface Pressures")
            extra_layout.addWidget(pressure_btn)
            result_table = QTableWidget()
            result_table.setColumnCount(4)
            result_table.setHorizontalHeaderLabels(
                ["Coordinate", "Object", "Node / Panel", "Distance"]
            )
            extra_layout.addWidget(result_table)
            copy_btn = QPushButton("Copy Table")

            def _copy_table(*_, table=result_table):

                lines = [
                    "\t".join(
                        table.horizontalHeaderItem(c).text()
                        for c in range(table.columnCount())
                    )
                ]
                for r in range(table.rowCount()):
                    vals = [
                        table.item(r, c).text() if table.item(r, c) else ""
                        for c in range(table.columnCount())
                    ]
                    lines.append("\t".join(vals))
                QGuiApplication.clipboard().setText("\n".join(lines))

            copy_btn.clicked.connect(_copy_table)
            extra_layout.addWidget(copy_btn)

            right_split = QHBoxLayout()
            right_split.addLayout(var_column)
            right_split.addWidget(extra_group)
            right.addLayout(right_split)


            def rebuild_lists(
                *_,

                obj_filter=obj_filter,
                obj_vars=obj_vars,
                var_filter=var_filter,
                var_vbox=var_vbox,
                var_vars=var_vars,
                obj_map=obj_map,
                extra_entry=extra_entry,
                default_inputs=default_inputs,

                obj_show_cb=obj_show_cb,
                var_show_cb=var_show_cb,

            ):

                def update_var_visibility(*_):
                    terms_var_vis = _parse_search_terms(var_filter.text())
                    for name, cb in var_vars.items():
                        visible = True

                        if terms_var_vis and not _matches_terms(name, terms_var_vis):
                            visible = False

                        if var_show_cb.isChecked() and not cb.isChecked():

                            visible = False
                        cb.setVisible(visible)

                terms_obj = _parse_search_terms(obj_filter.text())
                for name, cb in obj_vars.items():
                    visible = True
                    if terms_obj and not _matches_terms(name, terms_obj):
                        visible = False

                    if obj_show_cb.isChecked() and not cb.isChecked():

                        visible = False
                    cb.setVisible(visible)


                selected = [n for n, cb in obj_vars.items() if cb.isChecked()]
                prev_states = {name: cb.isChecked() for name, cb in var_vars.items()}
                for i in range(var_vbox.count() - 1, -1, -1):
                    w = var_vbox.itemAt(i).widget()
                    if w is not None:
                        w.deleteLater()
                var_vars.clear()

                extra_entry.clear()

                if selected:
                    first_type = obj_map[selected[0]][0].typeName
                    same_type = all(
                        obj_map[n][0].typeName == first_type for n in selected
                    )
                else:
                    same_type = False

                if selected and same_type:
                    otype = first_type
                    terms_var = _parse_search_terms(var_filter.text())
                    for vname in self.orcaflex_varmap.get(otype, []):

                        if not terms_var or _matches_terms(vname, terms_var):

                            cbv = QCheckBox(vname)
                            cbv.setChecked(prev_states.get(vname, False))
                            cbv.toggled.connect(update_var_visibility)
                            var_vbox.addWidget(cbv)
                            var_vars[vname] = cbv

                    if otype == "Line":
                        default_val = default_inputs.get("Line").text().strip()
                        extra_entry.setText(default_val or "EndA, mid, EndB")
                    elif otype in (
                        "Vessel",
                        "Buoy",
                        "Constraint",
                        "Environment",
                    ):
                        default_val = default_inputs.get(otype).text().strip()
                        extra_entry.setText(default_val or "0,0,0")
                else:
                    if not selected:
                        msg = "Select object(s) to see variables"
                    elif not same_type:
                        msg = "Selected objects are not the same type"
                    else:
                        msg = "Select object(s) to see variables"
                    var_vbox.addWidget(QLabel(msg))

                update_var_visibility()

            obj_filter.textChanged.connect(rebuild_lists)
            var_filter.textChanged.connect(rebuild_lists)
            for cb in obj_vars.values():
                cb.toggled.connect(rebuild_lists)
            rebuild_lists()


            def select_all_objects(*_, obj_vars=obj_vars):


                # Block signals while toggling to avoid repeated rebuilds

                for cb in obj_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)


                # Rebuild lists once after bulk toggle
                rebuild_lists()



            def unselect_all_objects(*_, obj_vars=obj_vars):

                for cb in obj_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)


                rebuild_lists()


            btn_obj_all.clicked.connect(select_all_objects)
            btn_obj_none.clicked.connect(unselect_all_objects)

            def select_all_vars(*_, var_vars=var_vars):

                for cb in var_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)


                rebuild_lists()



            def unselect_all_vars(*_, var_vars=var_vars):

                for cb in var_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)


                rebuild_lists()


            btn_var_all.clicked.connect(select_all_vars)
            btn_var_none.clicked.connect(unselect_all_vars)


            obj_show_cb.toggled.connect(rebuild_lists)
            var_show_cb.toggled.connect(rebuild_lists)


            def _update_table(coords, info, table=result_table):
                if table is None:
                    return
                table.setRowCount(len(coords))
                for row, (coord, (name, node, dist)) in enumerate(zip(coords, info)):
                    coord_str = f"{coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}"
                    table.setItem(row, 0, QTableWidgetItem(coord_str))
                    table.setItem(row, 1, QTableWidgetItem(name or ""))
                    node_txt = "" if node is None else str(node)
                    table.setItem(row, 2, QTableWidgetItem(node_txt))
                    dist_txt = "" if dist is None else f"{dist:.3f}"
                    table.setItem(row, 3, QTableWidgetItem(dist_txt))
                table.resizeColumnsToContents()

            def find_closest(
                *_,
                obj_vars=obj_vars,
                obj_map=obj_map,
                coord_entry=coord_entry,
                update_table=_update_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    return
                selected = [
                    obj_map[n][0] for n, cb in obj_vars.items() if cb.isChecked()
                ]
                if not selected:
                    return

                closest_info = self._get_closest_objects(coords, selected)
                chosen = {name for name, _, _ in closest_info}

                for name, cb in obj_vars.items():
                    cb.setChecked(name in chosen)
                rebuild_lists()
                update_table(coords, closest_info)

            find_btn.clicked.connect(find_closest)

            def find_closest_all(
                *_,
                obj_vars=obj_vars,
                obj_map=obj_map,
                coord_entry=coord_entry,
                skip_entry=skip_entry,
                update_table=_update_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    return
                skip_terms = [
                    s.strip().lower()
                    for s in skip_entry.text().split(',')
                    if s.strip()
                ]
                selected = [
                    pair[0]
                    for name, pair in obj_map.items()
                    if not any(term in name.lower() for term in skip_terms)
                ]
                closest_info = self._get_closest_objects(coords, selected)
                chosen = {name for name, _, _ in closest_info}

                for name, cb in obj_vars.items():
                    cb.setChecked(name in chosen)
                rebuild_lists()
                update_table(coords, closest_info)

            find_all_btn.clicked.connect(find_closest_all)

            def extract_surface_pressures(
                *_,
                coord_entry=coord_entry,
                model=model,
                fp=fp,
                update_table=_update_table,
                table=result_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    QMessageBox.warning(
                        dialog,
                        "No Coordinates",
                        "Enter at least one coordinate in the form x,y,z.",
                    )
                    panel_pressure_by_file.pop(fp, None)
                    if table is not None:
                        table.setRowCount(0)
                    return
                coord_tuples = [tuple(float(v) for v in c) for c in coords]
                if not coord_tuples:
                    QMessageBox.warning(
                        dialog,
                        "No Coordinates",
                        "Could not interpret the provided coordinates.",
                    )
                    panel_pressure_by_file.pop(fp, None)
                    if table is not None:
                        table.setRowCount(0)
                    return

                start_dir = self._last_diffraction_dir or os.path.dirname(fp)
                if not start_dir or not os.path.isdir(start_dir):
                    start_dir = os.path.dirname(fp)
                file_path, _ = QFileDialog.getOpenFileName(
                    dialog,
                    "Select Diffraction Model (.owr)",
                    start_dir,
                    "Diffraction data (*.owr *.dat);;All Files (*)",
                )
                if not file_path:
                    return

                directory = os.path.dirname(file_path)
                if directory:
                    self._last_diffraction_dir = directory

                try:
                    import OrcFxAPI
                except ImportError:
                    QMessageBox.critical(
                        dialog,
                        "OrcaFlex Not Available",
                        "OrcFxAPI is required to extract surface pressures.",
                    )
                    return

                diffraction_model = self._diffraction_cache.get(file_path)
                if diffraction_model is None:
                    try:
                        diffraction_model = OrcFxAPI.Diffraction(file_path)
                    except Exception as exc:
                        QMessageBox.critical(
                            dialog,
                            "Diffraction Load Error",
                            f"Failed to load diffraction model:\n{exc}",
                        )
                        return
                    self._diffraction_cache[file_path] = diffraction_model

                target_files = [target_fp for target_fp, cb in file_checks.items() if cb.isChecked()]
                if target_files:
                    if fp in target_files:
                        target_files = [fp] + [f for f in target_files if f != fp]
                    else:
                        target_files.insert(0, fp)
                else:
                    target_files = [fp]

                multi_target = len(target_files) > 1
                success_info = []
                no_data_files = []
                error_files = []

                for target_fp in target_files:
                    state = per_file_state.get(target_fp)
                    if state is None:
                        continue
                    target_model = state.get("model")
                    if target_model is None and target_fp == fp:
                        target_model = model
                    target_table = state.get("table")

                    try:
                        pressure_df, panel_info = self._get_panel_pressure(
                            model=target_model,
                            diffraction_model=diffraction_model,
                            panel_coords=tuple(coord_tuples),
                        )
                    except Exception as exc:
                        panel_pressure_by_file.pop(target_fp, None)
                        if target_table is not None:
                            target_table.setRowCount(0)
                        if multi_target:
                            error_files.append((target_fp, str(exc)))
                            continue
                        QMessageBox.critical(
                            dialog,
                            "Surface Pressure Error",
                            f"Failed to extract surface pressures:\n{exc}",
                        )
                        return

                    if (
                        pressure_df is None
                        or pressure_df.empty
                        or ("Time" in pressure_df.columns and pressure_df.shape[1] <= 1)
                        or ("Time" not in pressure_df.columns and pressure_df.shape[1] == 0)
                    ):
                        panel_pressure_by_file.pop(target_fp, None)
                        if target_table is not None:
                            target_table.setRowCount(0)
                        if multi_target:
                            no_data_files.append(target_fp)
                            continue
                        QMessageBox.warning(
                            dialog,
                            "No Surface Pressures",
                            "No surface pressure data was returned for the provided coordinates.",
                        )
                        return

                    if panel_info is None:
                        panel_info = pd.DataFrame()
                    else:
                        panel_info = panel_info.copy().reset_index(drop=True)

                    panel_pressure_by_file[target_fp] = [
                        {
                            "data": pressure_df.copy(),
                            "info": panel_info.copy(),
                            "diffraction_path": file_path,
                        }
                    ]

                    if not panel_info.empty:
                        surface_var_name = "Surface Pressures"
                        obj_vars_state = state.get("obj_vars", {})
                        obj_map_state = state.get("obj_map", {})
                        var_vars_state = state.get("var_vars", {})
                        rebuild_state = state.get("rebuild")

                        relevant_names = []
                        if "name" in panel_info.columns:
                            for raw_name in panel_info["name"].dropna().unique():
                                if not isinstance(raw_name, str):
                                    continue
                                if raw_name in obj_vars_state:
                                    relevant_names.append(raw_name)

                        selection_changed = False
                        if relevant_names:
                            target_types = {
                                obj_map_state[name][0].typeName
                                for name in relevant_names
                                if name in obj_map_state
                            }
                            for name in relevant_names:
                                cb = obj_vars_state.get(name)
                                if cb is None or cb.isChecked():
                                    continue
                                cb.blockSignals(True)
                                cb.setChecked(True)
                                cb.blockSignals(False)
                                selection_changed = True
                            if target_types:
                                for name, cb in obj_vars_state.items():
                                    if cb is None or not cb.isChecked() or name in relevant_names:
                                        continue
                                    obj_type = obj_map_state.get(name, (None,))[0]
                                    obj_type_name = getattr(obj_type, "typeName", None)
                                    if obj_type_name not in target_types:
                                        cb.blockSignals(True)
                                        cb.setChecked(False)
                                        cb.blockSignals(False)
                                        selection_changed = True
                        if selection_changed and callable(rebuild_state):
                            rebuild_state()
                            var_vars_state = state.get("var_vars", var_vars_state)

                        surface_cb = var_vars_state.get(surface_var_name)
                        if surface_cb is not None and not surface_cb.isChecked():
                            surface_cb.setChecked(True)

                        coords_for_table = []
                        info_for_table = []
                        for _, row in panel_info.iterrows():
                            coord_val = row.get("input_coord")
                            if isinstance(coord_val, (tuple, list, np.ndarray)) and len(coord_val) == 3:
                                coords_for_table.append(np.array(coord_val, dtype=float))
                            else:
                                coords_for_table.append(
                                    np.array(
                                        [
                                            float(row.get("X", 0.0)),
                                            float(row.get("Y", 0.0)),
                                            float(row.get("Z", 0.0)),
                                        ],
                                        dtype=float,
                                    )
                                )
                            node_val = row.get("pidx")
                            node_disp = None if pd.isna(node_val) else int(node_val) + 1
                            dist_val = row.get("distance")
                            if pd.isna(dist_val):
                                dist_val = None
                            info_for_table.append((row.get("name"), node_disp, dist_val))
                        update_table(coords_for_table, info_for_table, table=target_table)
                    else:
                        if target_table is not None:
                            target_table.setRowCount(0)

                    time_col = "Time" if "Time" in pressure_df.columns else None
                    n_series = pressure_df.shape[1] - (1 if time_col else 0)
                    success_info.append((target_fp, n_series))

                if not success_info:
                    if multi_target:
                        if error_files:
                            message_lines = ["Failed to extract surface pressures for:"]
                            for path, err in error_files:
                                message_lines.append(f" - {os.path.basename(path)}: {err}")
                            QMessageBox.critical(
                                dialog,
                                "Surface Pressure Error",
                                "\n".join(message_lines),
                            )
                        elif no_data_files:
                            names = ", ".join(os.path.basename(path) for path in no_data_files)
                            QMessageBox.warning(
                                dialog,
                                "No Surface Pressures",
                                f"No surface pressure data was returned for: {names}.",
                            )
                    return

                if multi_target:
                    total_series = sum(count for _, count in success_info)
                    message_lines = [
                        f"Loaded {total_series} surface pressure series from {len(success_info)} file(s)."
                    ]
                    if no_data_files:
                        names = ", ".join(os.path.basename(path) for path in no_data_files)
                        message_lines.append(f"No surface pressure data was returned for: {names}.")
                    if error_files:
                        message_lines.append("Failed to extract surface pressures for:")
                        for path, err in error_files:
                            message_lines.append(f" - {os.path.basename(path)}: {err}")
                    QMessageBox.information(
                        dialog,
                        "Surface Pressures Extracted",
                        "\n".join(message_lines),
                    )
                else:
                    n_series = success_info[0][1]
                    QMessageBox.information(
                        dialog,
                        "Surface Pressures Extracted",
                        f"Loaded {n_series} surface pressure series.",
                    )

            pressure_btn.clicked.connect(extract_surface_pressures)

            per_file_state[fp] = {
                "obj_vars": obj_vars,
                "var_vars": var_vars,
                "extra_entry": extra_entry,
                "model": model,
                "obj_map": obj_map,
                "rebuild": rebuild_lists,
                "table": result_table,
            }

        apply_specs = {}

        file_group = QGroupBox("Select Sims for this selection")
        file_layout = QVBoxLayout(file_group)
        file_checks = {}
        select_all_btn = QPushButton("Select All Sims")
        file_layout.addWidget(select_all_btn)
        status_label = QLabel()

        def select_all_sims():
            for cb in file_checks.values():
                cb.setChecked(True)

        select_all_btn.clicked.connect(select_all_sims)

        def check_files(*_):
            selected = [fp for fp, cb in file_checks.items() if cb.isChecked()]

            # Disable tabs not part of the selection
            if selected:
                for idx, fp in enumerate(file_paths):
                    tabs.setTabEnabled(idx, fp in selected)
                reuse_cb.setEnabled(False)
            else:
                for idx in range(tabs.count()):
                    tabs.setTabEnabled(idx, True)
                reuse_cb.setEnabled(True)

            if len(selected) < 2:
                status_label.setText("")
                apply_btn.setEnabled(bool(selected))
                return
            base = set(per_file_state[selected[0]]["obj_map"].keys())
            identical = all(
                set(per_file_state[f]["obj_map"].keys()) == base for f in selected[1:]
            )
            if identical:
                status_label.setText("")
                apply_btn.setEnabled(True)
            else:
                status_label.setText("objects not identical")
                apply_btn.setEnabled(False)

        for fp in file_paths:
            cb = QCheckBox(os.path.basename(fp))
            file_layout.addWidget(cb)
            file_checks[fp] = cb
            cb.toggled.connect(check_files)

        file_side.addWidget(file_group)
        file_side.addWidget(status_label)

        apply_btn = QPushButton("Apply Selection to Checked Sims")
        apply_btn.setEnabled(False)
        file_side.addWidget(apply_btn)

        def apply_selection():
            selected = [fp for fp, cb in file_checks.items() if cb.isChecked()]
            if not selected:
                return
            base_fp = file_paths[tabs.currentIndex()] if tabs.count() else selected[0]
            names_base = set(per_file_state[base_fp]["obj_map"].keys())
            if any(set(per_file_state[f]["obj_map"].keys()) != names_base for f in selected):
                status_label.setText("objects not identical")
                return
            st = per_file_state[base_fp]
            sel_objs = [n for n, cb in st["obj_vars"].items() if cb.isChecked()]
            if not sel_objs:
                QMessageBox.warning(dialog, "No Objects", "Select objects first")
                return
            sel_types = {st["obj_map"][n][0].typeName for n in sel_objs}
            if len(sel_types) != 1:
                QMessageBox.warning(dialog, "Type mismatch", "Selected objects are not the same type")
                return
            sel_vars = [v for v, cb in st["var_vars"].items() if cb.isChecked()]
            if not sel_vars:
                QMessageBox.warning(dialog, "No Variables", "Select variables first")
                return
            specs = []
            for obj_name in sel_objs:
                obj = st["obj_map"][obj_name][0]
                for var in sel_vars:
                    for ex, label in self._parse_extras(obj, st["extra_entry"].text()):
                        specs.append((obj_name, var, ex, label))
            for fp in selected:
                apply_specs[fp] = specs.copy()

                st_target = per_file_state[fp]
                for name, cb in st_target["obj_vars"].items():
                    cb.setChecked(name in sel_objs)
                st_target["rebuild"]()
                for var, cb in st_target["var_vars"].items():
                    cb.setChecked(var in sel_vars)
                st_target["extra_entry"].setText(st["extra_entry"].text())

                file_checks[fp].setChecked(False)
            status_label.setText(f"Stored selection for {len(selected)} file(s)")
            apply_btn.setEnabled(False)

        apply_btn.clicked.connect(apply_selection)

        reuse_cb = QCheckBox("Use this selection for all future OrcaFlex files")
        right_side.addWidget(reuse_cb)
        check_files()

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(cancel_btn)
        right_side.addLayout(btn_layout)

        out_specs = {}

        def on_load():
            self.orcaflex_redundant_subs = [
                s.strip() for s in redundant_entry.text().split(",") if s.strip()
            ]
            self._strip_coord_in_labels = strip_cb.isChecked()
            self._try_coord_wildcard = strip_cb.isChecked()
            rule_text = wc_entry.text().strip()
            if rule_text:
                if regex_cb.isChecked():
                    try:
                        pattern = re.compile(rule_text)
                        self._strip_rule = lambda n, p=pattern: p.sub("", n)
                    except re.error:
                        self._strip_rule = None
                else:
                    self._strip_rule = lambda n, t=rule_text: n.replace(t, "")
            else:
                self._strip_rule = None

            out_specs.update(apply_specs)

            for fp, st in per_file_state.items():
                if fp in out_specs:
                    continue
                sel_objs = [n for n, cb in st["obj_vars"].items() if cb.isChecked()]
                if not sel_objs:
                    continue
                sel_types = {st["obj_map"][n][0].typeName for n in sel_objs}
                if len(sel_types) != 1:
                    continue
                sel_vars = [v for v, cb in st["var_vars"].items() if cb.isChecked()]
                if not sel_vars:
                    continue
                specs = []
                for obj_name in sel_objs:
                    obj = st["obj_map"][obj_name][0]
                    for var in sel_vars:
                        for ex, label in self._parse_extras(obj, st["extra_entry"].text()):
                            specs.append((obj_name, var, ex, label))
                out_specs[fp] = specs

            if reuse_cb.isChecked() and file_paths:
                active_fp = file_paths[tabs.currentIndex()] if tabs.count() else None
                if active_fp in out_specs:
                    self._last_orcaflex_selection = out_specs[active_fp].copy()
                    self._reuse_orcaflex_selection = True

                    base_specs = self._last_orcaflex_selection
                    for fp in file_paths:
                        if fp == active_fp or fp in out_specs:
                            continue
                        st = per_file_state[fp]
                        mapped = []
                        obj_names = st["obj_map"].keys()
                        for obj_name, var, ex, label in base_specs:
                            target_name = obj_name
                            if target_name not in obj_names and getattr(self, "_strip_rule", None):
                                stripped = self._strip_rule(obj_name)
                                for cand in obj_names:
                                    if self._strip_rule(cand) == stripped:
                                        target_name = cand
                                        break
                            if target_name in obj_names:
                                mapped.append((target_name, var, ex, label))
                        if mapped:
                            out_specs[fp] = mapped

            missing_files = [fp for fp in file_paths if fp not in out_specs]
            if missing_files:
                resp = QMessageBox.question(
                    dialog,
                    "No Selection",
                    f"{len(missing_files)} file(s) have no selection. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return

            dialog.accept()

        load_btn.clicked.connect(on_load)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return {}

        result = {}
        for fp in file_paths:
            specs = out_specs.get(fp)
            tsdb = None
            if specs:
                tsdb = self._load_orcaflex_data_from_specs(
                    self.loaded_sim_models[fp], specs
                )
            if tsdb is None:
                tsdb = TsDB()

            entries = panel_pressure_by_file.get(fp, [])
            for entry in entries:
                tsdb = self._merge_panel_pressures(
                    tsdb, entry.get("data"), entry.get("info")
                )

            result[fp] = tsdb
        return result


    def _merge_panel_pressures(self, tsdb, pressures_df, panel_info):
        if tsdb is None:
            tsdb = TsDB()
        if pressures_df is None or pressures_df.empty:
            return tsdb
        if "Time" not in pressures_df.columns:
            return tsdb

        time = np.asarray(pressures_df["Time"].to_numpy())
        mapping = {}
        if panel_info is not None and hasattr(panel_info, "empty") and not panel_info.empty:
            for _, row in panel_info.iterrows():
                pidx = row.get("pidx")
                if pd.isna(pidx):
                    continue
                try:
                    key = str(int(pidx) + 1)
                except (TypeError, ValueError):
                    continue
                name = row.get("name")
                coord_val = row.get("input_coord")
                coord_txt = None
                if isinstance(coord_val, (tuple, list, np.ndarray)) and len(coord_val) == 3:
                    coord_txt = f"({coord_val[0]:.2f}, {coord_val[1]:.2f}, {coord_val[2]:.2f})"
                if name:
                    label = f"{name} Panel {int(pidx) + 1}"
                else:
                    label = f"Panel {int(pidx) + 1}"
                if coord_txt:
                    label = f"{label} Surface Pressure @ {coord_txt}"
                else:
                    label = f"{label} Surface Pressure"
                mapping[key] = label

        redundant = getattr(self, "orcaflex_redundant_subs", [])
        existing_names = {ts.name for ts in getattr(tsdb, "register", {}).values()}

        for col in pressures_df.columns:
            if col == "Time":
                continue
            data = pressures_df[col].to_numpy(dtype=float, copy=True)
            label = mapping.get(col, f"Panel {col} Surface Pressure")
            if redundant:
                label = self._strip_redundant(label, redundant)
            base_label = label
            suffix = 1
            while True:
                try:
                    tsdb.add(TimeSeries(label, time, data))
                    break
                except KeyError:
                    suffix += 1
                    label = f"{base_label} ({suffix})"
            existing_names.add(label)

        return tsdb


    def _strip_redundant(self, label, subs):
        for s in subs:
            label = label.replace(s, "")
        label = label.replace("__", "_").replace("::", ":").replace("  ", " ").strip("_:- ")
        return label

    def _resolve_orcaflex_line_end(self, end):
        try:
            import OrcFxAPI
        except ImportError:
            return None
        if end == "EndA":
            return OrcFxAPI.oeEndA
        elif end == "EndB":
            return OrcFxAPI.oeEndB
        elif isinstance(end, (float, int)):
            return OrcFxAPI.oeArcLength(end)
        else:
            return None

    def _parse_xyz_list(self, text: str):
        """Return list of xyz numpy arrays parsed from *text*."""
        import re
        coords = []
        pattern = re.compile(
            r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?"
        )
        for match in pattern.finditer(text):
            try:
                coords.append(
                    np.array(
                        [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                    )
                )
            except ValueError:
                pass
        return coords

    def _get_closest_objects(self, coords, objects):
        """Return info on closest objects for each coordinate."""
        try:
            import OrcFxAPI
        except ImportError:
            return []

        import numpy as np

        if not coords or not objects:
            return []

        specs = []
        info = []
        for obj in objects:
            if obj.typeName == "Constraint":
                for v in ["In-frame X", "In-frame Y", "In-frame Z"]:
                    specs.append(OrcFxAPI.TimeHistorySpecification(obj, v, None))
                info.append((obj.Name, 1, "Constraint"))
            elif obj.typeName == "Line":
                n_nodes = len(obj.NodeArclengths)
                for n in range(n_nodes):
                    for v in ["X", "Y", "Z"]:
                        specs.append(
                            OrcFxAPI.TimeHistorySpecification(
                                obj, v, OrcFxAPI.oeNodeNum(n + 1)
                            )
                        )
                info.append((obj.Name, n_nodes, "Line"))

        if not specs:
            return []

        data = OrcFxAPI.GetMultipleTimeHistories(specs, OrcFxAPI.pnStaticState)[0]

        obj_data = {}
        idx = 0
        for name, count, typ in info:
            if typ == "Constraint":
                arr = np.array([data[idx : idx + 3]])
                idx += 3
            else:
                arr = np.array([
                    data[idx + i * 3 : idx + i * 3 + 3] for i in range(count)
                ])
                idx += 3 * count
            obj_data[name] = arr

        results = []
        for c in coords:
            best_name = None
            best_dist = None
            best_node = None
            for name, arr in obj_data.items():
                if arr.shape[0] == 1:
                    dist = np.linalg.norm(arr[0] - c)
                    node = None
                else:
                    dists = np.linalg.norm(arr - c, axis=1)
                    node_idx = int(np.argmin(dists))
                    dist = dists[node_idx]
                    node = node_idx + 1
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = name
                    best_node = node
                elif dist == best_dist:
                    if node is not None and best_node is not None:
                        if node < best_node:
                            best_name = name
                            best_node = node
            results.append((best_name, best_node, best_dist))

        return results

    def _get_panel_pressure(
        self,
        model,
        diffraction_model,
        time_window=None,
        kpa_to_pa=True,
        panel_coords=None,
        include_hydrostatic=True,
        simidx=0,
        work_folder=None,
        use_existing=True,
        ow_element_nth_resolution: int = 1,
    ):
        try:
            import OrcFxAPI
        except ImportError as exc:
            raise RuntimeError("OrcFxAPI not available") from exc

        if model is None:
            raise ValueError("model is required")
        if diffraction_model is None:
            raise ValueError("diffraction_model is required")

        if time_window is None:
            time_window = OrcFxAPI.SpecifiedPeriod(
                model.simulationStartTime, model.simulationStopTime
            )
        else:
            time_window = OrcFxAPI.SpecifiedPeriod(time_window[0], time_window[1])

        geometry = getattr(diffraction_model, "panelGeometry", None)
        if geometry is None:
            return pd.DataFrame(), pd.DataFrame()

        panel_geometry = [
            (res[0], res[1], res[3][0], res[3][1], res[3][2]) for res in geometry
        ]
        if not panel_geometry:
            return pd.DataFrame(), pd.DataFrame()

        pd_geo = pd.DataFrame(
            panel_geometry, columns=("idx_id", "name", "X", "Y", "Z")
        )
        pd_geo["pidx"] = pd_geo.index.values

        panel_array = pd.DataFrame()
        if panel_coords is None:
            panel_array = pd_geo.loc[pd_geo.idx_id != -1].copy()
            if panel_array.empty:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), panel_array
            panel_array["input_coord"] = list(
                zip(panel_array["X"], panel_array["Y"], panel_array["Z"])
            )
            panel_array["distance"] = np.nan
        else:
            coords_list = []
            for coord in panel_coords:
                if isinstance(coord, np.ndarray):
                    coord = coord.tolist()
                if coord is None:
                    continue
                try:
                    coords_list.append(tuple(float(v) for v in coord))
                except (TypeError, ValueError):
                    continue

            if not coords_list:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), pd.DataFrame()

            available = pd_geo.loc[pd_geo.idx_id != -1].copy()
            if available.empty:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), pd.DataFrame()

            rows = []
            for idx, pt in enumerate(coords_list):
                x, y, z = pt
                this_geo = available.copy()
                this_geo["distance"] = np.sqrt(
                    np.power(this_geo.X.values - x, 2)
                    + np.power(this_geo.Y.values - y, 2)
                    + np.power(this_geo.Z.values - z, 2)
                )
                best = this_geo.sort_values("distance").iloc[0]
                row_dict = best.to_dict()
                row_dict["distance"] = float(row_dict.get("distance", np.nan))
                row_dict["input_coord"] = tuple(pt)
                row_dict["input_index"] = idx
                rows.append(row_dict)

            if rows:
                panel_array = pd.DataFrame(rows)
                if ow_element_nth_resolution and ow_element_nth_resolution > 1:
                    panel_array = panel_array.iloc[::ow_element_nth_resolution]
                panel_array = panel_array.reset_index(drop=True)

        if panel_array is None or panel_array.empty:
            time = model["General"].TimeHistory("Time", time_window)
            return pd.DataFrame({"Time": time}), panel_array

        if "input_index" in panel_array.columns:
            panel_array = panel_array.sort_values("input_index").reset_index(drop=True)

        panel_array = panel_array.dropna(subset=["name"]).reset_index(drop=True)
        if panel_array.empty:
            time = model["General"].TimeHistory("Time", time_window)
            return pd.DataFrame({"Time": time}), panel_array

        if "pidx" in panel_array.columns:
            panel_array["pidx"] = panel_array["pidx"].astype(int)

        all_press = []
        unique_bodies = np.unique(panel_array["name"].values)

        for this_body in unique_bodies:
            press_df = None
            found_this_body_press = False
            if (
                use_existing
                and work_folder
                and isinstance(work_folder, str)
                and os.path.isdir(work_folder)
            ):
                for run_file in os.listdir(work_folder):
                    if (
                        f"of_pressures_sim_idx_{simidx}" in run_file
                        and this_body in run_file
                    ):
                        try:
                            press_df = pd.read_feather(
                                os.path.join(work_folder, run_file)
                            )
                            found_this_body_press = True
                            break
                        except Exception:
                            press_df = None
                            found_this_body_press = False
            if not found_this_body_press:
                panel_ids = panel_array.loc[panel_array.name == this_body, "pidx"].values
                if panel_ids.size == 0:
                    continue
                panel_press = model[this_body].PanelPressureTimeHistory(
                    diffraction=diffraction_model,
                    resultPanels=panel_ids,
                    period=time_window,
                    parameters=OrcFxAPI.PanelPressureParameters(
                        IncludeHydrostaticPressure=include_hydrostatic,
                        IncludeDiffractionPressure=True,
                        IncludeRadiationPressure=True,
                    ),
                )
                press_df = pd.DataFrame(
                    panel_press,
                    columns=np.int32(panel_ids + 1).astype(str),
                )
                if (
                    work_folder
                    and isinstance(work_folder, str)
                    and os.path.isdir(work_folder)
                ):
                    filename = (
                        f"of_pressures_sim_idx_{simidx}_body_{this_body}"
                        f"_resolution_{ow_element_nth_resolution}_"
                        f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.feather"
                    )
                    try:
                        press_df.to_feather(os.path.join(work_folder, filename))
                    except Exception:
                        pass
            if press_df is not None:
                all_press.append(press_df)

        time = model["General"].TimeHistory("Time", time_window)
        if not all_press:
            return pd.DataFrame({"Time": time}), panel_array

        all_press = pd.concat(all_press, axis=1)
        if kpa_to_pa:
            all_press = all_press * 1000
        all_press["Time"] = time
        return all_press, panel_array

    def _parse_extras(self, obj, entry_val: str):
        """Interpret extra strings for an OrcaFlex object."""
        import OrcFxAPI

        entry_val = entry_val.strip()

        if obj.typeName == "Line":
            if not entry_val:
                return [(None, None)]
            tokens = [t.strip() for t in entry_val.split(",") if t.strip()]
            total = obj.NodeArclengths[-1]
            extras = []
            for t in tokens:
                low = t.lower().strip()
                m_node = re.match(r"node\s*(\d+)", low)
                m_arc = re.match(r"arc\s*(\d+(?:\.\d+)?)", low)
                if low == "enda":
                    extras.append((OrcFxAPI.oeArcLength(obj.NodeArclengths[0]), "EndA"))
                elif low == "endb":
                    extras.append((OrcFxAPI.oeArcLength(total), "EndB"))
                elif low in ("mid", "middle"):
                    extras.append((OrcFxAPI.oeArcLength(total / 2), "mid"))
                elif m_node:
                    num = int(m_node.group(1))
                    extras.append((OrcFxAPI.oeNodeNum(num), f"Node {num}"))
                elif m_arc:
                    val = float(m_arc.group(1))
                    extras.append((OrcFxAPI.oeArcLength(val), f"Arc {val}"))
                else:
                    try:
                        val = float(t)
                    except ValueError:
                        continue
                    extras.append((OrcFxAPI.oeArcLength(val), f"Arc {val}"))
            return extras

        if obj.typeName in ("Vessel", "Buoy", "Constraint", "Environment"):
            groups = [g.strip() for g in entry_val.split(";") if g.strip()]
            if not groups:
                groups = ["0,0,0"]
            extras = []
            for grp in groups:
                txt = grp
                if txt.lower().startswith("pos"):
                    txt = txt[3:].strip()
                xyz = [s.strip() for s in txt.strip("() ").split(",") if s.strip()]
                if len(xyz) != 3:
                    continue
                try:
                    coords = [float(v) for v in xyz]
                except ValueError:
                    continue
                label = f"Pos ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                if obj.typeName == "Vessel":
                    extras.append((OrcFxAPI.oeVessel(coords), label))
                elif obj.typeName == "Buoy":
                    extras.append((OrcFxAPI.oeBuoy(coords), label))
                elif obj.typeName == "Constraint":
                    extras.append((OrcFxAPI.oeConstraint(coords), label))
                elif obj.typeName == "Environment":
                    extras.append((OrcFxAPI.oeEnvironment(coords), label))
            return extras if extras else [(None, None)]

        return [(None, None)]

    def _load_orcaflex_data_from_specs(self, model, selection_specs):
        try:
            import OrcFxAPI
        except ImportError:
            print("OrcFxAPI not available. Cannot preload .sim files.")
            return
        tsdb = TsDB()
        time_spec = OrcFxAPI.SpecifiedPeriod(0, model.simulationTimeStatus.CurrentTime)
        time = model["General"].TimeHistory("Time", time_spec)
        object_var_map = {obj.Name: obj for obj in model.objects}
        def _match_obj(name):
            obj = object_var_map.get(name)
            if obj is None and getattr(self, "_strip_rule", None):
                stripped = self._strip_rule(name)
                for cand, o in object_var_map.items():
                    if self._strip_rule(cand) == stripped:
                        obj = o
                        break
            return obj
        resolved_specs = []
        names = []
        redundant_subs = getattr(self, "orcaflex_redundant_subs", [])
        for spec in selection_specs:
            try:
                label_override = None
                if len(spec) == 5:
                    obj, obj_name, var, end, label_override = spec
                elif len(spec) == 4:
                    if isinstance(spec[0], str):
                        obj_name, var, end, label_override = spec
                        obj = _match_obj(obj_name)
                    else:
                        obj, obj_name, var, end = spec
                elif len(spec) == 3:
                    obj_name, var, end = spec
                    obj = _match_obj(obj_name)
                else:
                    continue
                if isinstance(var, str) and var.strip().lower() == "surface pressures":
                    continue
                if obj is None:
                    continue
                short_obj = self._strip_redundant(obj_name, redundant_subs)
                short_var = self._strip_redundant(var, redundant_subs)
                if obj.typeName == "Line":
                    if end == "EndA" or (isinstance(end, str) and end.lower() == "enda"):
                        end_enum = self._resolve_orcaflex_line_end("EndA")
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end_enum)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (EndA)"
                            )
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif end == "EndB" or (isinstance(end, str) and end.lower() == "endb"):
                        end_enum = self._resolve_orcaflex_line_end("EndB")
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end_enum)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (EndB)"
                            )
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif hasattr(OrcFxAPI, "ObjectExtra") and isinstance(end, OrcFxAPI.ObjectExtra):
                        arc_val = getattr(end, 'ArcLength', None)
                        if label_override:
                            label = f"{short_obj}:{short_var} ({label_override})"
                        elif arc_val is not None:
                            label = f"{short_obj}:{short_var} (Arc {arc_val:.2f})"
                        else:
                            label = f"{short_obj}:{short_var} (extra {end})"
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif isinstance(end, (float, int)):
                        try:
                            extra = OrcFxAPI.oeArcLength(end)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (Arc {end:.2f})"
                            )
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, extra)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif isinstance(end, str):
                        try:
                            end_val = float(end)
                            extra = OrcFxAPI.oeArcLength(end_val)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (Arc {end_val:.2f})"
                            )
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, extra)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    else:
                        continue
                elif obj.typeName in ("Vessel", "Buoy", "Constraint", "Environment") and hasattr(OrcFxAPI, "ObjectExtra") and isinstance(end, OrcFxAPI.ObjectExtra):
                    try:
                        spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end)
                        label = (
                            f"{short_obj}:{short_var} ({label_override})"
                            if label_override
                            else f"{short_obj}:{short_var} (pos {end})"
                        )
                        resolved_specs.append(spec_obj)
                        names.append(label)
                    except Exception:
                        continue
                else:
                    try:
                        spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var)
                        label = f"{short_obj}:{short_var}"
                        resolved_specs.append(spec_obj)
                        names.append(label)
                    except Exception:
                        continue
            except Exception:
                continue
        if not resolved_specs:
            return tsdb
        try:
            results = OrcFxAPI.GetMultipleTimeHistories(resolved_specs, time_spec)
            for i, name in enumerate(names):
                tsdb.add(TimeSeries(name, time, results[:, i]))
            return tsdb
        except Exception as e:
            QMessageBox.critical(self.parent_gui, "OrcaFlex Read Error", f"Could not read variables:\n{e}")
            return None

    def _load_generic_file(self, filepath):
        ext = os.path.splitext(filepath)[-1].lower().lstrip(".")
        if ext in ["csv", 'mat', 'dat', 'ts',  'h5', 'pickle', 'tda', 'asc', 'tdms', 'pkl', 'bin']:
            return TsDB.fromfile(filepath)
        elif ext == "xlsx":
            df = pd.read_excel(filepath)
        elif ext == "json":
            df = pd.read_json(filepath)
        elif ext == "feather":
            df = pd.read_feather(filepath)
        elif ext == "parquet":
            df = pd.read_parquet(filepath)
        else:
            raise NotImplementedError(f"No loader for extension: {ext}")
        # Detect time column
        time_col = next((c for c in df.columns if c.lower() in ["time", "t"]), df.columns[0])
        time = df[time_col].values
        tsdb = TsDB()
        skipped = set()

        # Detect potential identifier columns with string values
        id_col = None

        string_cols = [
            c
            for c in df.columns
            if c != time_col
            and pd.api.types.is_string_dtype(df[c])
            and df[c].map(
                lambda x: isinstance(x, str)
                or (not hasattr(x, "__iter__") and pd.isna(x))
            ).all()
        ]

        for sc in string_cols:
            resp = QMessageBox.question(
                self.parent_gui,
                "Identifier Column?",
                f"Column '{sc}' contains strings. Use as identifier?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                id_col = sc
                break
            else:
                skipped.add(sc)


        if id_col:
            split_columns = {}
            for ident, subdf in df.groupby(id_col):
                time_vals = subdf[time_col].values
                for col in df.columns:

                    if col in (time_col, id_col):
                        continue
                    # ensure any pyarrow/extension values are converted to
                    # regular Python objects before further inspection
                    values = []

                    for v in subdf[col].tolist():
                        if hasattr(v, "to_pylist"):
                            v = v.to_pylist()
                        elif isinstance(v, array):
                            v = list(v)
                        elif isinstance(v, np.ndarray):
                            v = v.tolist()
                        values.append(v)

                    if col not in split_columns:
                        # consider only non-null entries when checking for list-like values
                        non_null = []
                        for v in values:
                            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                                non_null.append(v)
                            elif not pd.isna(v):
                                non_null.append(v)
                        if non_null and all(
                            isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                            for v in non_null
                        ):
                            lengths = {len(v) for v in non_null}
                            if len(lengths) == 1:
                                n = lengths.pop()
                                resp = QMessageBox.question(
                                    self.parent_gui,
                                    "Split Column?",
                                    f"Column '{col}' contains list/tuple values of length {n}.\nSplit into {n} columns?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes,
                                )
                                if resp == QMessageBox.Yes:
                                    name_str, ok = QInputDialog.getText(
                                        self.parent_gui,
                                        "Column Names",
                                        f"Enter {n} comma-separated names for '{col}':",
                                    )
                                    if ok:
                                        names = [s.strip() for s in name_str.split(",") if s.strip()]
                                    else:
                                        names = []
                                    if len(names) != n:
                                        names = [f"{col}_{i+1}" for i in range(n)]
                                    split_columns[col] = names
                                else:
                                    split_columns[col] = None
                            else:
                                split_columns[col] = None
                        else:
                            split_columns[col] = None
                    names = split_columns.get(col)
                    if names:
                        for i in range(len(names)):
                            row_vals = []
                            for row in values:
                                if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                                    try:
                                        row_vals.append(row[i])
                                    except Exception:
                                        row_vals.append(np.nan)
                                else:
                                    row_vals.append(np.nan)
                            try:
                                data = np.array(row_vals, dtype=float)
                            except Exception:
                                skipped.add(f"{col}_{i}")
                                continue

                            tsdb.add(TimeSeries(f"{names[i]}_{ident}", time_vals, data))
                        continue
                    try:
                        numeric_values = np.array(values, dtype=float)
                        tsdb.add(TimeSeries(f"{col}_{ident}", time_vals, numeric_values))
                    except Exception:
                        skipped.add(col)

        else:
            for col in df.columns:
                if col == time_col:
                    continue
                # Convert potential extension array values to regular Python
                values = []
                for v in df[col].tolist():
                    if hasattr(v, "to_pylist"):
                        v = v.to_pylist()
                    elif isinstance(v, array):
                        v = list(v)
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    values.append(v)
                # Consider only non-null entries when checking for list-like values
                non_null = []
                for v in values:
                    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                        non_null.append(v)
                    elif not pd.isna(v):
                        non_null.append(v)
                if non_null and all(
                    isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                    for v in non_null
                ):
                    lengths = {len(v) for v in non_null}
                    if len(lengths) == 1:
                        n = lengths.pop()
                        resp = QMessageBox.question(
                            self.parent_gui,
                            "Split Column?",
                            f"Column '{col}' contains list/tuple values of length {n}.\nSplit into {n} columns?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if resp == QMessageBox.Yes:
                            name_str, ok = QInputDialog.getText(
                                self.parent_gui,
                                "Column Names",
                                f"Enter {n} comma-separated names for '{col}':",
                            )
                            if ok:
                                names = [s.strip() for s in name_str.split(",") if s.strip()]
                            else:
                                names = []
                            if len(names) != n:
                                names = [f"{col}_{i+1}" for i in range(n)]
                            for i in range(n):
                                row_vals = []
                                for row in values:
                                    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                                        try:
                                            row_vals.append(row[i])
                                        except Exception:
                                            row_vals.append(np.nan)
                                    else:
                                        row_vals.append(np.nan)
                                try:
                                    data = np.array(row_vals, dtype=float)
                                except Exception:
                                    skipped.add(f"{col}_{i}")
                                    continue

                                tsdb.add(TimeSeries(names[i], time, data))
                            continue
                try:
                    numeric_values = np.array(values, dtype=float)
                    tsdb.add(TimeSeries(col, time, numeric_values))
                except Exception:
                    skipped.add(col)

        if len(tsdb.getm()) == 0:
            if 'time' in df.columns or 't' in df.columns:
                time_col = next((c for c in df.columns if c.lower() in ["time", "t"]), df.columns[0])
                time = df[time_col].values
            else:
                time = np.arange(len(df))
            tsdb.add(TimeSeries("NO_DATA", time, np.full_like(time, np.nan, dtype=float)))
        if skipped:
            print(
                f"Skipped non-numeric columns in {os.path.basename(filepath)}: {', '.join(sorted(skipped))}"
            )
        return tsdb

__all__ = ['FileLoader']

