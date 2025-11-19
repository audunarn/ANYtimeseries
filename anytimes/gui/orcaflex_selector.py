"""Qt dialog for selecting OrcaFlex variables."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .utils import (
    ORCAFLEX_VARIABLE_MAP,
    _matches_terms,
    _parse_search_terms,
    get_object_available_vars,
)
from .layout_utils import apply_initial_size

class OrcaflexVariableSelector(QDialog):
    def __init__(self, model, orcaflex_varmap=None, parent=None, previous_selection=None, allow_reuse=False):
        super().__init__(parent)
        self.setWindowTitle("Select OrcaFlex Variables")
        apply_initial_size(
            self,
            desired_width=1200,
            desired_height=820,
            min_width=860,
            min_height=620,
            width_ratio=0.92,
            height_ratio=0.92,
        )
        self.model = model
        if orcaflex_varmap is not None:
            self.orcaflex_varmap = orcaflex_varmap
        else:
            self.orcaflex_varmap = ORCAFLEX_VARIABLE_MAP or {}
        self.selected = []
        self.reuse_all = False

        main = QVBoxLayout(self)
        obj_var_layout = QHBoxLayout()
        main.addLayout(obj_var_layout)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        obj_var_layout.addLayout(left_layout, stretch=1)
        obj_var_layout.addLayout(right_layout, stretch=1)

        left_layout.addWidget(QLabel("Objects"))
        self.obj_filter = QLineEdit()
        self.obj_filter.setPlaceholderText("Search objects")
        left_layout.addWidget(self.obj_filter)
        self.obj_list = QListWidget()
        self.obj_list.setSelectionMode(QAbstractItemView.MultiSelection)
        left_layout.addWidget(self.obj_list)

        right_layout.addWidget(QLabel("Variables"))
        self.var_filter = QLineEdit()
        self.var_filter.setPlaceholderText("Search variables")
        right_layout.addWidget(self.var_filter)
        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QAbstractItemView.MultiSelection)
        right_layout.addWidget(self.var_list)

        # Only objects for which we have variable types in the varmap
        self.object_map = {
            obj.Name: obj.typeName
            for obj in self.model.objects
            if obj.typeName in self.orcaflex_varmap
        }
        self.all_objects = sorted(self.object_map.keys())
        self._update_object_list("")

        self.extra_entry = QLineEdit()
        self.extra_entry.setPlaceholderText("Arc Lengths / Extra (e.g. EndA, 10.0, EndB)")
        obj_var_layout.addWidget(QLabel("Arc Length/Extra:"))
        obj_var_layout.addWidget(self.extra_entry)

        self.redundant_entry = QLineEdit()
        self.redundant_entry.setPlaceholderText("Redundant substrings (comma-separated)")
        main.addWidget(QLabel("Remove Redundant Substrings from Labels:"))
        main.addWidget(self.redundant_entry)

        self.reuse_cb = QCheckBox("Use this selection for all future OrcaFlex files")
        self.reuse_cb.setChecked(allow_reuse)
        main.addWidget(self.reuse_cb)

        btns = QHBoxLayout()
        self.ok_btn = QPushButton("Load")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        main.addLayout(btns)

        self.obj_list.itemSelectionChanged.connect(self.update_var_list)
        self.obj_filter.textChanged.connect(self._update_object_list)
        self.var_filter.textChanged.connect(self.update_var_list)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.reject)

    def _update_object_list(self, text):
        terms = _parse_search_terms(text)
        selected = {item.text() for item in self.obj_list.selectedItems()}
        self.obj_list.clear()
        for name in self.all_objects:

            if not terms or _matches_terms(name, terms):

                item = QListWidgetItem(name)
                self.obj_list.addItem(item)
                if name in selected:
                    item.setSelected(True)
        self.update_var_list()

    def update_var_list(self, *_):

        self.var_list.clear()
        selected_objs = self.obj_list.selectedItems()
        if not selected_objs:
            return
        obj_type = self.object_map[selected_objs[-1].text()]
        variables = get_object_available_vars(obj_type, self.orcaflex_varmap)
        terms = _parse_search_terms(self.var_filter.text())
        for var in variables:

            if not terms or _matches_terms(var, terms):

                self.var_list.addItem(var)

    def on_ok(self):
        objects = [item.text() for item in self.obj_list.selectedItems()]
        variables = [item.text() for item in self.var_list.selectedItems()]
        if not objects or not variables:
            QMessageBox.warning(self, "Selection required", "Select at least one object and one variable.")
            return
        extras = [x.strip() for x in self.extra_entry.text().split(",") if x.strip()]
        redundant = [x.strip() for x in self.redundant_entry.text().split(",") if x.strip()]
        self.selected = []
        for obj in objects:
            for var in variables:
                if extras:
                    for extra in extras:
                        self.selected.append((obj, var, extra))
                else:
                    self.selected.append((obj, var, None))
        self.redundant = redundant
        self.reuse_all = self.reuse_cb.isChecked()
        self.accept()

    @staticmethod
    def get_selection(model, orcaflex_varmap=None, parent=None, previous_selection=None, allow_reuse=False):
        dlg = OrcaflexVariableSelector(model, orcaflex_varmap, parent, previous_selection, allow_reuse)
        result = dlg.exec()
        if result == QDialog.Accepted:
            return dlg.selected, dlg.redundant, dlg.reuse_all
        else:
            return None, None, None

__all__ = ['OrcaflexVariableSelector']

