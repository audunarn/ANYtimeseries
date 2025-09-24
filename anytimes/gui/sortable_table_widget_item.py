"""Custom table item with numeric-aware sorting."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem

class SortableTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        my = self.data(Qt.ItemDataRole.UserRole)
        other_val = other.data(Qt.ItemDataRole.UserRole)
        if isinstance(my, (int, float)) and isinstance(other_val, (int, float)):
            return my < other_val
        return super().__lt__(other)

__all__ = ['SortableTableWidgetItem']

