from PyQt5.QtCore import QRegExp, Qt, QAbstractListModel, QModelIndex, QVariant
from PyQt5.QtGui import QColor, QPalette, QRegExpValidator
from PyQt5.QtWidgets import QWidget, QCheckBox, QListWidgetItem, QMessageBox, QLabel, QComboBox, QStackedWidget, \
    QFrame, QTextEdit, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout, \
    QApplication, QListView, QStyledItemDelegate, QStyleOptionButton, QStyle, QSizePolicy

from app.lib.draggablelistwidget import FilterDragListWidget
from app.psyDataFunc import PsyDataFunc


def get_compare_expression(operator: str, dataValue: str, dataType: str):
    if dataType == 'Raw value':
        outputStr = dataValue
    elif dataType == 'SDs':
        if operator == '<' or operator == '<=':
            outputStr = 'mean + ' + dataValue + ' * SD'
        else:
            outputStr = 'mean - ' + dataValue + ' * SD'
    elif dataType == 'MAD':
        if operator == '<' or operator == '<=':
            outputStr = 'median + ' + dataValue + ' * MAD'
        else:
            outputStr = 'median - ' + dataValue + ' * MAD'
    else:
        if operator == '<' or operator == '<=':
            outputStr = 'mean + Shifting Z * SD'
        else:
            outputStr = 'mean - Shifting Z * SD'

    return " ".join([operator, outputStr])


class CheckboxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Retrieve the text and check state data
        text = index.data(Qt.DisplayRole)
        checked = index.data(Qt.CheckStateRole) == Qt.Checked

        # Draw the checkbox first (on the left side)
        checkbox_style_option = QStyleOptionButton()
        checkbox_style_option.state = QStyle.State_Enabled | (QStyle.State_On if checked else QStyle.State_Off)
        checkbox_style_option.rect = option.rect
        QApplication.style().drawControl(QStyle.CE_CheckBox, checkbox_style_option, painter)

        # Draw the text next (on the right side)
        text_rect = option.rect
        text_rect.setLeft(option.rect.left() + 30)  # Adjust position to the right of the checkbox
        painter.save()
        painter.drawText(text_rect, Qt.AlignVCenter, text)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() == event.MouseButtonRelease:
            # Toggle checkbox state on click
            current_state = index.data(Qt.CheckStateRole)
            new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
            model.setData(index, new_state, Qt.CheckStateRole)
        return True


class MyModel(QAbstractListModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        if not index.isValid():
            return QVariant()

        if role == Qt.DisplayRole:
            return self._data[index.row()][0]
        elif role == Qt.CheckStateRole:
            return Qt.Checked if self._data[index.row()][1] else Qt.Unchecked
        return QVariant()

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if role == Qt.CheckStateRole and index.isValid():
            row = index.row()
            self._data[row] = (self._data[row][0], value == Qt.Checked)
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable

    def getCheckedItems(self):
        # Collect all checked items
        checked_items = [item[0] for item in self._data if item[1]]
        return checked_items

    def resetSelections(self):
        # Reset all checkbox states to unchecked
        for row in range(len(self._data)):
            self._data[row] = (self._data[row][0], False)
        # Notify the view that all data has changed
        self.layoutChanged.emit()


class FilterWindow(QWidget):
    def __init__(self, dataframe, mainFilterList):
        super().__init__()
        self.isVariableNumeric = None
        self.selected_var_name = None
        self.data = dataframe
        self.mainFilterList = mainFilterList
        self.chooseItem = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.resize(300, 700)

        self.floatIntReg = r"-?\d+\.\d+|-?\d+"
        self.isValue2Disabled = False
        self.isValue1Disabled = False

        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        self.setWindowTitle("Define Filter")
        self.var_name_label = QLabel("Variable Names:")
        self.current_filters_label = QLabel("Current Filters:")
        self.befiltered_var_box = QComboBox()

        self.check_list_radio = QCheckBox("Check List")
        self.check_list_radio.setChecked(True)
        self.check_list_radio.setCheckable(True)

        self.range_radio = QCheckBox("Range")
        self.range_radio.setCheckable(True)

        self.cdf_pooling_radio = QCheckBox("Pooling CDF")
        self.cdf_pooling_radio.setCheckable(True)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.setStyleSheet("")

        """
        checklist page widgets
        """
        self.checklist_page = QWidget()

        self.checklist_page_var_info = QLabel("Variable name: target.RT")
        self.checklist_page_info = QLabel()

        # self.check_list = QListWidget()
        # self.check_list.addItems(["1", "2", "4", "5", "6"])
        # self.checklist_page_scrollArea = QScrollArea()
        # self.checklist_page_scrollArea.setFrameShape(QFrame.NoFrame)
        """
        create the model
        """
        # Create the model
        self.model = MyModel([])  # Start with an empty model
        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(CheckboxDelegate())
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)

        """
        range page widgets
        """
        self.range_page = QWidget()
        self.range_page_var_info = QLabel("Variable name: target.RT")
        self.range_page_info = QTextEdit()

        self.logical_operator_comboBox = QComboBox()
        self.logical_operator_comboBox.addItems(["and", "or", "none"])
        self.logical_operator_comboBox.setEditable(False)

        self.compare_operator_comboBox1 = QComboBox()
        self.compare_operator_comboBox1.addItems(["<", "<=", ">", ">="])
        self.compare_operator_comboBox1.setEditable(False)
        self.compare_operator_comboBox2 = QComboBox()
        self.compare_operator_comboBox2.addItems([">", ">=", "<", "<="])
        self.compare_operator_comboBox2.setEditable(False)

        self.value_comboBox1 = QComboBox()
        self.value_comboBox2 = QComboBox()
        self.value_comboBox1.setValidator(QRegExpValidator(QRegExp(self.floatIntReg)))
        self.value_comboBox2.setValidator(QRegExpValidator(QRegExp(self.floatIntReg)))
        self.value_comboBox1.addItems(["1.5", "2.5", "3", "3.29", "4"])
        self.value_comboBox2.addItems(["0.1", "2.5", "3", "3.29", "4"])

        self.data_type_comboBox1 = QComboBox()
        self.data_type_comboBox1.addItems(["Raw value", "SDs", "Shifting Z", "MAD"])
        self.data_type_comboBox1.setEditable(False)
        self.data_type_comboBox2 = QComboBox()
        self.data_type_comboBox2.setEditable(False)
        self.data_type_comboBox2.addItems(["Raw value", "SDs", "Shifting Z", "MAD"])

        self.add_btn = QPushButton("Add Filter")

        """
        cdf pooling widget
        """
        self.cdf_pooling_page = QWidget()

        self.cdf_pooling_var_info = QLabel("Variable name: target.RT")
        self.cdf_pooling_page_info = QTextEdit()

        """
        filter list:
        """
        self.filter_list = FilterDragListWidget()

        """
        buttons
        """
        self.clear_btn = QPushButton("Clear")
        self.clear_all_btn = QPushButton("Clear All")
        self.close_btn = QPushButton("OK")

        self.range_page_info.setReadOnly(True)
        self.range_page_info.setTextInteractionFlags(Qt.NoTextInteraction)
        self.range_page_info.setFrameShape(QFrame.NoFrame)
        self.range_page_info.setObjectName("range_help_info")

        self.cdf_pooling_page_info.setReadOnly(True)
        self.cdf_pooling_page_info.setTextInteractionFlags(Qt.NoTextInteraction)
        self.cdf_pooling_page_info.setFrameShape(QFrame.NoFrame)
        self.cdf_pooling_page_info.setObjectName("cdf_pooling_help_info")

        self.checklist_page_info.setText("""Check the values to include in the filter. Only the selected values can be included
in the view/analysis/summary procedure.""")

        self.range_page_info.setHtml("""
                                    Select a range of values to include in the filter. Only values within the defined range can
                                    be included in the view/analysis/summary procedure.<br>
                                    <br>The <b>Shifting Z</b> method will use
                                    the non-recursive Shifting Z criterion (<i>Van Selst & Jolicoeur, 1994, QJEP</i>) to remove outliers.<br>
                                    <br>The <b>MAD</b> (Median Absolute Deviation, <i>Leys et. al., 2013, JESP</i>) method
                                    will use the <i>n</i>*MAD (MAD = 1.4826*median(|X - median(X)|)), criterion to remove outliers.
                                    """)
        cP = self.range_page_info.palette()
        cP.setColor(QPalette.Base, QColor(255, 255, 255, 0))
        self.range_page_info.setPalette(cP)

        self.cdf_pooling_page_info.setHtml("""
                                            Click the 'Add Filter' button to include the <b>pooling CDF</b> method in the filter. 
                                            This method, proposed by Miller (2024, <i>BRM</i>), uses the pooled cumulative distribution 
                                            function to identify outliers. <br>
                                             <br>However, this method has specific requirements, including a minimum of 50 data 
                                             points per condition/cell to ensure reliable CDF estimation.<br> 
                                             <br>As this is a novel method not yet undergone extensive validation, 
                                             it should be applied cautiously.
                                           """)
        self.cdf_pooling_page_info.setPalette(cP)

        # setup layout
        title_layout = QGridLayout()
        title_layout.addWidget(QLabel("Variable Names:"), 0, 0)
        title_layout.addWidget(self.befiltered_var_box, 0, 1)
        title_layout.setColumnMinimumWidth(2, 15)
        title_layout.addWidget(self.check_list_radio, 0, 3)
        title_layout.addWidget(self.range_radio, 0, 4)
        title_layout.addWidget(self.cdf_pooling_radio, 0, 5)

        select_define_group = QGroupBox("Define Filter")
        filter_list_group = QGroupBox("Filter List")

        """
        checklist page layout
        """
        checklist_layout = QVBoxLayout()
        checklist_layout.addWidget(self.checklist_page_var_info)
        checklist_layout.addWidget(self.checklist_page_info)
        checklist_layout.addWidget(self.list_view)

        """
        cdf pooling page layout
        """
        cdf_pooling_layout = QVBoxLayout()
        cdf_pooling_layout.addWidget(self.cdf_pooling_var_info)
        cdf_pooling_layout.addWidget(self.cdf_pooling_page_info)

        """
        range page layout
        """
        range_layout = QGridLayout()
        #  variable name: and selection info for range:
        range_layout.addWidget(self.range_page_var_info, 0, 0, 1, 4)
        range_layout.addWidget(self.range_page_info, 1, 0, 3, 4)

        # first row of range definition
        range_layout.addWidget(self.compare_operator_comboBox1, 4, 1)
        range_layout.addWidget(self.value_comboBox1, 4, 2)
        range_layout.addWidget(self.data_type_comboBox1, 4, 3)

        # second row of range definition
        range_layout.addWidget(self.logical_operator_comboBox, 5, 0)
        range_layout.addWidget(self.compare_operator_comboBox2, 5, 1)
        range_layout.addWidget(self.value_comboBox2, 5, 2)
        range_layout.addWidget(self.data_type_comboBox2, 5, 3)

        self.checklist_page.setLayout(checklist_layout)
        self.range_page.setLayout(range_layout)
        self.cdf_pooling_page.setLayout(cdf_pooling_layout)

        # add pages to stacked widget
        self.stackedWidget.addWidget(self.checklist_page)
        self.stackedWidget.addWidget(self.range_page)
        self.stackedWidget.addWidget(self.cdf_pooling_page)
        self.stackedWidget.setCurrentIndex(0)

        add_btn_layout = QHBoxLayout()
        add_btn_layout.addStretch()
        add_btn_layout.addWidget(self.add_btn)

        define_layout = QVBoxLayout()
        define_layout.addLayout(title_layout)
        define_layout.addWidget(self.stackedWidget)

        select_define_group.setLayout(define_layout)

        filter_list_layout = QVBoxLayout()
        filter_list_layout.addWidget(self.filter_list)
        filter_list_group.setLayout(filter_list_layout)

        """
        setup buttons
        """
        btns_layout = QHBoxLayout()
        btns_layout.addStretch()
        btns_layout.addStretch()
        btns_layout.addWidget(self.clear_btn)
        btns_layout.addWidget(self.clear_all_btn)
        btns_layout.addWidget(self.close_btn)

        """
        setup all layout
        """
        allLayout = QVBoxLayout()
        allLayout.addWidget(select_define_group)
        allLayout.addLayout(add_btn_layout)
        allLayout.addWidget(filter_list_group)
        allLayout.addLayout(btns_layout)

        self.setLayout(allLayout)

        """
        connect signals and slots
        """
        self.check_list_radio.stateChanged.connect(self.checkListRadioEvent)
        self.range_radio.stateChanged.connect(self.rangeRadioEvent)
        self.cdf_pooling_radio.stateChanged.connect(self.cdfPoolingRadioEvent)

        self.befiltered_var_box.currentIndexChanged.connect(self.changeComboBox)
        self.filter_list.itemClicked.connect(self.onItemClicked)
        self.filter_list.listChanged.connect(self.rowChange)

        self.value_comboBox1.setEditable(True)
        self.value_comboBox1.currentTextChanged.connect(self.valueText1Changed)
        self.value_comboBox2.setEditable(True)
        self.value_comboBox2.currentTextChanged.connect(self.valueText2Changed)

        self.data_type_comboBox1.currentIndexChanged.connect(self.dataType1Changed)
        self.data_type_comboBox2.currentIndexChanged.connect(self.dataType2Changed)

        self.logical_operator_comboBox.currentIndexChanged.connect(self.logicalOperatorChanged)

        self.clear_btn.clicked.connect(self.clearOne)
        self.clear_all_btn.clicked.connect(self.clearAll)
        self.close_btn.clicked.connect(self.closeWindow)

        self.add_btn.clicked.connect(self.addEvent)

        # 初始化 filter
        self.initFilterList()

        self.selected_var_name = self.data.columns[0]
        self.checklist_page_var_info.setText("Variable name: " + self.selected_var_name)
        self.range_page_var_info.setText("Variable name: " + self.selected_var_name)
        self.cdf_pooling_var_info.setText("Variable name: " + self.selected_var_name)
        # 下拉框添加数据
        max_len, t_len = 0, 0
        pt_val = self.befiltered_var_box.font().pointSize()

        for variableName in self.data.columns.tolist():
            t_len = len(variableName)
            if max_len < t_len:
                max_len = t_len

            self.befiltered_var_box.addItem(variableName)

        self.befiltered_var_box.view().setMinimumWidth(max_len * pt_val)

        # initial the checklist with the first variable
        # default to show the first variable in the list
        self.updateModelData(self.data.columns[0])

    def initFilterList(self):
        for iItem in range(self.mainFilterList.count()):
            item = self.mainFilterList.item(iItem)
            self.filter_list.addItem(item.text())

    def logicalOperatorChanged(self):
        if self.logical_operator_comboBox.currentText() == 'none':
            self.value_comboBox2.setEnabled(False)
            self.data_type_comboBox2.setEnabled(False)
        else:
            self.value_comboBox2.setEnabled(True)
            self.data_type_comboBox2.setEnabled(True)

    def valueText2Changed(self, text):
        if text in ['auto', '']:
            return None
        try:
            float(text)
        except ValueError:
            QMessageBox.information(self, "Information", "Please enter a number")
            self.value_comboBox2.clearEditText()

    def valueText1Changed(self, text):
        if text in ['auto', '']:
            return None
        try:
            float(text)
        except ValueError:
            QMessageBox.information(self, "Information", "Please enter a number")
            self.value_comboBox1.clearEditText()

    def dataType2Changed(self, index):
        if index == 2:
            try:
                self.isValue2Disabled = True
                self.value_comboBox2.addItem('auto')
                self.value_comboBox2.setEditable(False)
                self.value_comboBox2.setEnabled(False)
                self.value_comboBox2.setCurrentText('auto')
            except Exception as e:
                print(e)
        else:
            if self.isValue2Disabled:
                self.value_comboBox2.removeItem(self.value_comboBox2.findText('auto'))
                self.value_comboBox2.setEditable(True)
                self.value_comboBox2.setEnabled(True)
                self.isValue2Disabled = False

    def dataType1Changed(self, index):
        if index == 2:
            try:
                self.isValue1Disabled = True
                self.value_comboBox1.addItem('auto')
                self.value_comboBox1.setEditable(False)
                self.value_comboBox1.setEnabled(False)
                self.value_comboBox1.setCurrentText('auto')
            except Exception as e:
                print(e)
        else:
            if self.isValue1Disabled:
                self.value_comboBox1.removeItem(self.value_comboBox1.findText('auto'))
                self.value_comboBox1.setEditable(True)
                self.value_comboBox1.setEnabled(True)
                self.isValue1Disabled = False

    def checkListRadioEvent(self):
        if self.check_list_radio.checkState() == Qt.Checked:
            self.stackedWidget.setCurrentIndex(0)
            self.range_radio.setChecked(False)
            self.cdf_pooling_radio.setChecked(False)

    def rangeRadioEvent(self):
        if self.range_radio.checkState() == Qt.Checked:
            self.stackedWidget.setCurrentIndex(1)
            self.check_list_radio.setChecked(False)
            self.cdf_pooling_radio.setChecked(False)

    def cdfPoolingRadioEvent(self):
        if self.cdf_pooling_radio.checkState() == Qt.Checked:
            self.stackedWidget.setCurrentIndex(2)
            self.check_list_radio.setChecked(False)
            self.range_radio.setChecked(False)

    def changeComboBox(self):

        var_name_label = "Variable name: "
        selected_var_name = self.befiltered_var_box.currentText()
        self.selected_var_name = selected_var_name
        self.checklist_page_var_info.setText(var_name_label + self.selected_var_name)
        self.range_page_var_info.setText(var_name_label + self.selected_var_name)
        self.cdf_pooling_var_info.setText(var_name_label + self.selected_var_name)

        # Get unique values from the selected column
        self.updateModelData(selected_var_name)

    def updateModelData(self, variableName: str):
        self.isVariableNumeric = not isinstance(self.data[variableName].iloc[0], str)
        unique_values = self.data[variableName].dropna().unique()

        # Prepare data for the model: [(value, False), ...]
        data = [(str(value), False) for value in unique_values]
        # Update the model with new data
        self.model.beginResetModel()
        self.model._data = data
        self.model.endResetModel()

    # 添加多选框
    def insertOld(self):
        self.check_list.clear()
        # only check the first value
        self.isVariableNumeric = not isinstance(self.data[self.selected_var_name].iloc[0], str)

        # Filter out null values and get unique non-null values
        non_null_values = self.data[self.selected_var_name].dropna().unique().tolist()
        QApplication.processEvents()

        # in case all values are empty
        if any(non_null_values):
            for value in non_null_values:
                box = QCheckBox(str(value))  # Create a QCheckBox
                listItem = QListWidgetItem()  # 实例化一个Item，QListWidget，不能直接加入QCheckBox
                self.check_list.addItem(listItem)  # Add QListWidgetItem to QListWidget
                self.check_list.setItemWidget(listItem, box)  # Set QCheckBox as the widget for QListWidgetItem

    def getChooseOld(self):
        # if not hasattr(self, 'check_list') or self.check_list is None:
        #     return []

        count = self.check_list.count()  # 得到QListWidget的总个数
        if count == 0:
            return []

        chooses = []
        for i in range(count):
            item = self.check_list.item(i)
            if item is not None:
                widget = self.check_list.itemWidget(item)
                if widget and widget.isChecked():
                    chooses.append(widget.text())

        return chooses

    def getChoose(self):
        # Call the model's method to get checked items
        return self.model.getCheckedItems()

    def addEvent(self):
        if self.stackedWidget.currentIndex() == 0:
            self.addCheckListEvent()
        elif self.stackedWidget.currentIndex() == 1:
            self.addRangeEvent()
        else:
            self.addCdfPoolingEvent()

    # cdfPooling event
    def addCdfPoolingEvent(self):
        text = self.selected_var_name
        text += ":Pooling CDF"
        self.filter_list.addItem(text)
        self.mainFilterList.addItem(text)

    # checklist确认事件
    def addCheckListEvent(self):
        chooses = self.getChoose()
        text = self.selected_var_name
        text += ":"
        # 拼接所有多选框选中的值
        if self.isVariableNumeric:
            all_values_Str = "".join(f" = {v}" for v in chooses)
        else:
            all_values_Str = "".join(f" = '{v}'" for v in chooses)

        self.filter_list.addItem(text + all_values_Str)
        self.mainFilterList.addItem(text + all_values_Str)
        self.clearCheckedBox()

    def clearCheckedBox(self):
        # Call the model's method to reset all selections
        self.model.resetSelections()

    # def clearCheckedBoxOld(self):
    #
    #     total_items = self.check_list.count()
    #
    #     for index in range(total_items):
    #         item = self.check_list.item(index)
    #         if item is not None:
    #
    #             checkbox_widget = self.check_list.itemWidget(item)
    #             if checkbox_widget is not None:
    #                 checkbox_widget.setChecked(False)

    # range 的 add 按钮按下事件
    def addRangeEvent(self):
        text = self.selected_var_name
        logical_operator_text = self.logical_operator_comboBox.currentText()
        value_text1 = self.value_comboBox1.currentText()
        value_text2 = self.value_comboBox2.currentText()
        data_type_text1 = self.data_type_comboBox1.currentText()
        data_type_text2 = self.data_type_comboBox2.currentText()
        compare_operator_text1 = self.compare_operator_comboBox1.currentText()
        compare_operator_text2 = self.compare_operator_comboBox2.currentText()

        if data_type_text1 != 'Shifting Z' and value_text1 == '':
            msg = QMessageBox(QMessageBox.Warning, "Warning", "Input value cannot be empty.")
            msg.exec_()
            return None
        if logical_operator_text != "none" and data_type_text2 != 'Shifting Z' and value_text2 == '':
            msg = QMessageBox(QMessageBox.Warning, "Warning", "Input value cannot be empty.")
            msg.exec_()
            return None

        logical_operator_text = logical_operator_text.strip()
        compare_operator_text1 = compare_operator_text1.strip()
        compare_operator_text2 = compare_operator_text2.strip()

        expression1 = get_compare_expression(compare_operator_text1, value_text1, data_type_text1)

        if logical_operator_text != "none":
            expression2 = get_compare_expression(compare_operator_text2, value_text2, data_type_text2)

            tmpText = " ".join([expression1, logical_operator_text, expression2])
        else:
            tmpText = expression1

        self.filter_list.addItem(text + ":" + tmpText)
        self.mainFilterList.addItem(text + ":" + tmpText)
        self.resetRangeBox()

    def resetRangeBox(self):
        self.logical_operator_comboBox.setCurrentIndex(0)

        self.compare_operator_comboBox1.setCurrentIndex(0)
        self.compare_operator_comboBox2.setCurrentIndex(0)

        self.value_comboBox1.setEditable(True)
        self.value_comboBox1.setCurrentText('')
        self.value_comboBox1.clear()
        self.value_comboBox1.addItems(["1.5", "2.5", "3", "3.29", "4"])

        self.value_comboBox2.setEditable(True)
        self.value_comboBox2.clear()
        self.value_comboBox2.setCurrentText('')
        self.value_comboBox2.addItems(["0.1", "2.5", "3", "3.29", "4"])

        self.data_type_comboBox1.setCurrentIndex(0)
        self.data_type_comboBox2.setCurrentIndex(0)

    # 清空
    def clearAll(self):
        self.filter_list.clear()
        self.mainFilterList.clear()

    def onItemClicked(self, item):
        self.chooseItem = item

    # 清除选中项
    def clearOne(self):
        if self.chooseItem is None:
            msg = QMessageBox(QMessageBox.Warning, "Warning", "Please select a filter item to clear.")
            msg.exec_()
        else:
            row = self.filter_list.row(self.chooseItem)
            self.filter_list.takeItem(row)
            self.mainFilterList.takeItem(row)
            self.chooseItem = None

    # 关闭窗口
    def closeWindow(self):
        self.close()

    def closeEvent(self, event):
        event.accept()

    def rowChange(self):
        items = [self.filter_list.item(index).text() for index in range(self.filter_list.count())]

        self.mainFilterList.clear()
        self.mainFilterList.addItems(items)
