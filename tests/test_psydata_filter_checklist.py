import os
import unittest

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QListWidget

from app.lib.filterWindow import FilterWindow, MyModel


class ChecklistModelTests(unittest.TestCase):
    def test_bulk_selection_and_inversion(self):
        model = MyModel([("A", False), ("B", False), ("C", False)])

        model.setAllChecked(True)
        self.assertTrue(model.areAllChecked())
        self.assertEqual(model.checkedCount(), 3)

        model.invertSelections()
        self.assertEqual(model.getCheckedItems(), [])

        model.setData(model.index(1, 0), Qt.Checked, Qt.CheckStateRole)
        model.invertSelections()
        self.assertEqual(model.getCheckedItems(), ["A", "C"])


class ChecklistWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = FilterWindow(
            pd.DataFrame({"subject": ["S01", "S02", "S03"]}),
            QListWidget(),
        )

    def tearDown(self):
        self.window.close()

    def test_select_all_button_uses_checked_state_to_select_and_clear(self):
        self.assertEqual(self.window.checklist_toggle_all_btn.text(), "Select All")
        self.assertFalse(self.window.checklist_toggle_all_btn.isChecked())
        self.window.checklist_toggle_all_btn.click()
        self.assertEqual(self.window.model.checkedCount(), 3)
        self.assertTrue(self.window.checklist_toggle_all_btn.isChecked())
        self.assertEqual(self.window.checklist_toggle_all_btn.text(), "Select All")

        self.window.checklist_toggle_all_btn.click()
        self.assertEqual(self.window.model.checkedCount(), 0)
        self.assertFalse(self.window.checklist_toggle_all_btn.isChecked())
        self.assertEqual(self.window.checklist_toggle_all_btn.text(), "Select All")

    def test_search_filters_without_changing_checks(self):
        self.window.model.setData(self.window.model.index(0, 0), Qt.Checked, Qt.CheckStateRole)
        self.window.checklist_search.setText("s02")

        self.assertEqual(self.window.checklist_proxy_model.rowCount(), 1)
        self.assertEqual(self.window.checklist_proxy_model.index(0, 0).data(), "S02")
        self.assertEqual(self.window.model.getCheckedItems(), ["S01"])

    def test_invert_selects_all_except_prechecked_values(self):
        self.window.model.setData(self.window.model.index(1, 0), Qt.Checked, Qt.CheckStateRole)
        self.window.checklist_invert_btn.click()

        self.assertEqual(self.window.model.getCheckedItems(), ["S01", "S03"])
        self.assertEqual(self.window.checklist_selection_info.text(), "Included: 2 / 3")

    def test_distribution_preview_button_emits_request(self):
        requests = []
        self.window.previewRequested.connect(lambda: requests.append(True))
        self.window.preview_btn.click()
        self.assertEqual(requests, [True])


if __name__ == "__main__":
    unittest.main()
