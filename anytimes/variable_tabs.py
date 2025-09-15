from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QCheckBox,
    QLineEdit, QLabel, QScrollArea, QPushButton, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
import re

class VariableRowWidget(QWidget):
    rename_requested = Signal(str, str)

    def __init__(self, varname, allow_rename=False, parent=None):
        super().__init__(parent)
        self._name = varname

        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        self.input = QLineEdit()
        self.input.setFixedWidth(70)
        self.label = QLabel(varname)

        layout.addWidget(self.checkbox)
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        if allow_rename:
            self.rename_btn = QPushButton("Rename")
            self.rename_btn.setFixedWidth(70)
            self.rename_btn.clicked.connect(self._prompt_rename)
            layout.addWidget(self.rename_btn)

        layout.addStretch(1)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)

    def _prompt_rename(self):
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Variable",
            "New variable name",
            text=self._name,
        )
        if ok and new_name and new_name != self._name:
            self.rename_requested.emit(self._name, new_name)

class VariableTab(QWidget):
    """VariableTab with search and select-all functionality."""
    checklist_updated = Signal()

    def __init__(self, label, variables, user_var_set=None, allow_rename=False, rename_callback=None):
        super().__init__()
        self.all_vars = sorted(list(variables))
        self.user_var_set = user_var_set or set()
        self.allow_rename = allow_rename
        self.rename_callback = rename_callback
        self.checkboxes = {}
        self.inputs = {}
        self._checked_state = {}
        self._input_state = {}
        layout = QVBoxLayout(self)
        # -- Search and Select All row --
        top_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        top_row.addWidget(self.search_box)
        self.select_all_btn = QPushButton("Select All")
        self.unselect_all_btn = QPushButton("Unselect All")
        top_row.addWidget(self.select_all_btn)
        top_row.addWidget(self.unselect_all_btn)
        layout.addLayout(top_row)
        # -- Scrollable area for variable checkboxes --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self._populate_checkboxes(self.all_vars)
        scroll.setWidget(self.inner)
        layout.addWidget(scroll)
        # Connections
        self.select_all_btn.clicked.connect(lambda: self.set_all(True))
        self.unselect_all_btn.clicked.connect(lambda: self.set_all(False))
        self.search_box.textChanged.connect(self._search_update)

    def _populate_checkboxes(self, vars_to_show):
        """Populate the scroll area with VariableRowWidget entries."""
        # Preserve existing states
        for var, cb in self.checkboxes.items():
            self._checked_state[var] = cb.isChecked()
            self._input_state[var] = self.inputs[var].text()

        # Clear layout
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.takeAt(i).widget()
            if widget:
                widget.deleteLater()
        self.checkboxes.clear()
        self.inputs.clear()

        for var in vars_to_show:
            row = VariableRowWidget(var, allow_rename=self.allow_rename)
            if var in self.user_var_set:
                row.label.setStyleSheet("color: #2277bb;")
            if self.allow_rename and self.rename_callback:
                row.rename_requested.connect(self.rename_callback)
            self.inner_layout.addWidget(row)
            self.checkboxes[var] = row.checkbox
            self.inputs[var] = row.input
            if var in self._checked_state:
                row.checkbox.setChecked(self._checked_state[var])
            if var in self._input_state:
                row.input.setText(self._input_state[var])

        self.inner_layout.addStretch()
        self.checklist_updated.emit()

    def _search_update(self, text):
        terms = _parse_search_terms(text)
        if not terms:
            vars_to_show = self.all_vars
        else:
            vars_to_show = [v for v in self.all_vars if _matches_terms(v, terms)]
        self._populate_checkboxes(vars_to_show)

    def selected_variables(self):
        return [var for var, cb in self.checkboxes.items() if cb.isChecked()]

    def set_all(self, value):
        for cb in self.checkboxes.values():
            cb.setChecked(value)

def _parse_search_terms(text):
    """Return a list of search term groups from ``text``."""
    text = text.lower()
    include_comma = ",," in text
    placeholders = {}

    def _replace(match):
        idx = len(placeholders)
        placeholder = f"\x00{idx}\x00"
        tokens = [t.strip() for t in match.group(1).split(',') if t.strip()]
        placeholders[placeholder] = tokens or [""]
        return placeholder

    text_no_groups = re.sub(r"!!(.*?)!!", _replace, text)

    groups = []
    for tok in [t.strip() for t in text_no_groups.split(',') if t.strip()]:
        if tok in placeholders:
            groups.append(placeholders[tok])
        else:
            groups.append([tok])

    if include_comma:
        groups.append([','])

    return groups

def _matches_terms(name, terms):
    """Return ``True`` if ``name`` matches all search ``terms``."""
    if not terms:
        return True

    name_l = name.lower()
    return all(any(t in name_l for t in group) for group in terms)