from PySide2.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, \
    QTreeWidgetItem, QGraphicsItem, QStyle, QWidget, QVBoxLayout, QPushButton, QLabel, QMainWindow, QMessageBox
from PySide2.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QKeySequence
from PySide2.QtCore import Qt, QCoreApplication, QRectF, QObject
from PySide2.QtUiTools import QUiLoader
from ConfigWindow import ConfigWindow
from Sample import Sample
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import fpc.pyplot_template
import numpy as np
import pandas as pd
import cv2
import traceback
import ctypes
import json
import os

matplotlib.use("Qt5Agg")


class MainWindow:
    def __init__(self):
        self.ui = QUiLoader().load("resources/main.ui")
        self.setupUI()

        self.scene_sample = QGraphicsScene()
        self.ui.graphicsView_sample.setScene(self.scene_sample)

        self.scene_figure = QGraphicsScene()
        self.ui.graphicsView_figure.setScene(self.scene_figure)

        self.config = None
        self.row_analysis = True
        self.history_path = None

        self.template = None
        self.samples = []
        self.data = None
        self.seg_index = None

        self.setup_callbacks()
        self.show_sample()
        self.plot()

        self.load_temp()
        self.config_window = ConfigWindow(8, self)

    def setupUI(self):
        self.ui.setWindowTitle("Strain Meter")
        self.ui.setWindowIcon(QPixmap("resources/icon.ico"))

        self.ui.tree.setColumnWidth(0, 192)
        self.ui.tree.setColumnWidth(1, 128)
        self.ui.tree.setColumnWidth(2, 88)
        self.ui.tree.setColumnWidth(3, 88)

        self.ui.spinBox_position.setMaximum(999)
        self.ui.spinBox_edge1.setMaximum(999)
        self.ui.spinBox_edge2.setMaximum(999)
        self.ui.spinBox_width.setMaximum(999)

        self.ui.spinBox_position_2.setMaximum(999)
        self.ui.spinBox_edge1_2.setMaximum(999)
        self.ui.spinBox_edge2_2.setMaximum(999)
        self.ui.spinBox_width_2.setMaximum(999)

    def setup_callbacks(self):
        self.scene_sample.mousePressEvent = self.mousePressEventLoadSample

        self.ui.groupBox_row.toggled.connect(self.row_switch)
        self.ui.groupBox_col.toggled.connect(self.col_switch)
        self.ui.pushButton_analysis.clicked.connect(self.batch_analysis)
        self.ui.pushButton_template.clicked.connect(self.template_setting)
        self.ui.toolButton_add.clicked.connect(self.openFileLoadSample)
        self.ui.toolButton_del.clicked.connect(self.remove_sample)
        self.ui.pushButton_set_template.clicked.connect(self.change_template)
        self.ui.tree.itemSelectionChanged.connect(self.show_sample_and_figure)

        self.ui.actionNew.triggered.connect(self.new_project)
        self.ui.actionNew.setShortcut(QKeySequence("Ctrl+N"))
        self.ui.actionOpen.triggered.connect(self.open_project)
        self.ui.actionOpen.setShortcut(QKeySequence("Ctrl+O"))
        self.ui.actionSave.triggered.connect(self.save_project)
        self.ui.actionSave.setShortcut(QKeySequence("Ctrl+S"))
        self.ui.actionSaveas.triggered.connect(self.save_as_project)
        self.ui.actionSaveas.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.ui.actionexport_ROI.triggered.connect(self.export_ROI)
        self.ui.actionrename.triggered.connect(self.batch_rename)

        self.ui.actionyingbian.triggered.connect(lambda: self.export_data("yingbian"))
        self.ui.actionbosong.triggered.connect(lambda: self.export_data("bosong"))

        app.aboutToQuit.connect(self.quit)

    def template_setting(self, button):
        segmentation_result = self.template.segmentation
        self.config_window.start(segmentation_result, self.row_analysis)

    def batch_rename(self):
        # 询问
        choice = QMessageBox.question(
            self.ui,
            '重命名',
            '是否按字典序批量重命名以导入 Ncorr?',
        )
        if choice == QMessageBox.Yes:
            for i, sample in enumerate(self.samples):
                if sample.path == self.template.path:
                    self.template.rename(i+1, False)
                sample.rename(i+1, True)

            self.update_sample_tree()
            self.ui.statusbar.showMessage("批量重命名完成")

    def batch_analysis(self, button):
        self.value2config()
        length = self.template.analysis(self.config, self.row_analysis)

        if length == "?":
            self.ui.statusbar.showMessage("模板分析失败：未找到边缘")
            self.update_sample_tree()
            return
        elif length == "error":
            self.ui.statusbar.showMessage("模板分析失败：未知错误")
            self.update_sample_tree()
            return
        else:
            self.ui.statusbar.showMessage(f"模板分析成功")

        left_edge_x = self.template.details[1 - self.row_analysis]["left_edge_x"]
        right_edge_x = self.template.details[1 - self.row_analysis]["right_edge_x"]
        self.config[1 - self.row_analysis]["left_edge"] = left_edge_x
        self.config[1 - self.row_analysis]["right_edge"] = right_edge_x

        for sample in self.samples:
            sample.crop_by_template(self.template.cropped_image)
            sample.analysis(self.config, self.row_analysis)

        self.ui.statusbar.showMessage(f"样本批量分析结束")

        self.config2value()
        self.update_sample_tree()

    def export_data(self, mode=None):
        data = pd.DataFrame(columns=["name", "width", "height"])
        _, template_width, template_height = self.template.get_data()
        for sample in self.samples:
            name, width, height = sample.get_data()
            data.loc[len(data.index)] = [name, width, height]

        # 将后三列设为最小有效数字
        data["εx"] = (data["width"] - template_width) / template_width
        data["εy"] = (data["height"] - template_height) / template_height
        data["v=-εy/εx"] = -data["εy"] / data["εx"]
        for column in ["εx", "εy", "v=-εy/εx"]:
            data[column] = data[column].apply(lambda x: round(x, 6))

        self.data = data

        if mode is None:
            return data
        if mode == "yingbian":
            data = data.loc[:, ["name", "width", "εx"]]
            file_name = "应变数据_" + self.template.name
        else:
            file_name = "泊松比数据_" + self.template.name

        # 使用文件对话框获取用户选择的保存路径
        if os.path.exists(self.history_path):
            path = self.history_path
            path = os.path.join(path, file_name + ".xlsx")
        else:
            path = "C:\\" + file_name + ".xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self.ui,  # 父窗口对象
            "保存数据",  # 标题
            path,  # 起始目录
            "Excel 文件 (*.xlsx)"  # 选择类型过滤项，过滤内容在括号中
        )

        if file_path:
            data.to_excel(file_path, index=False)

            filename = os.path.basename(path)
            parent_dir = os.path.basename(os.path.dirname(path))
            self.ui.statusbar.showMessage(f"数据已保存至 {os.path.join(parent_dir, filename)}")

    def export_ROI(self):
        if self.seg_index is None:
            QMessageBox.critical(
                self.ui,
                '错误',
                '请选择ROI！')
            return
        path = self.template.path
        path = os.path.join(os.path.dirname(path), "strain_ROI")
        if not os.path.exists(path):
            os.mkdir(path)

        ROI_region = self.template.get_ROI_region(self.seg_index)
        for sample in self.samples:
            sample.save_ROI(ROI_region, path)

        self.ui.statusbar.showMessage(f"ROI已保存至源图像目录下 strain_ROI 目录中")

    def row_switch(self, checked):
        self.ui.groupBox_col.setChecked(not checked)
        self.row_col_switch(checked)

    def col_switch(self, checked):
        self.ui.groupBox_row.setChecked(not checked)

    def row_col_switch(self, row_analysis):
        print("row_analysis", row_analysis)
        self.row_analysis = row_analysis

        self.show_sample_and_figure()

    def mousePressEventLoadSample(self, event):
        if self.template is None and event.button() == Qt.LeftButton:
            self.openFileLoadSample()

    def change_template(self, button):
        sample = self.current_sample()
        self.set_template(sample.path)
        self.update_sample_tree()

    def set_template(self, sample_path):
        self.template = Sample(sample_path)
        self.template.crop_by_self()
        self.template.segment()
        for sample in self.samples:
            sample.crop_by_template(self.template.cropped_image)

        self.ui.pushButton_template.setEnabled(True)
        self.ui.pushButton_analysis.setEnabled(True)

    def openFileLoadSample(self, button=None):
        if self.history_path is not None and os.path.exists(self.history_path):
            path = self.history_path
        else:
            path = "C:\\"
        image_paths, _ = QFileDialog.getOpenFileNames(
            self.ui,  # 父窗口对象
            "批量添加待分析样本",  # 标题
            path,  # 起始目录
            "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
        )
        if image_paths:
            self.import_sample(image_paths)

    def import_sample(self, image_paths):
        self.ui.graphicsView_sample.setCursor(Qt.ArrowCursor)
        self.history_path = os.path.dirname(image_paths[0])

        if self.template is None:
            self.set_template(image_paths[0])
        for image_path in image_paths:
            sample = Sample(image_path)
            self.samples.append(sample)
            sample.crop_by_template(self.template.cropped_image)

        self.give_short_name()
        self.update_sample_tree()

    def give_short_name(self):
        if len(self.samples) < 2:
            return
        names = [sample.name for sample in self.samples]
        prefix = os.path.commonprefix(names)

        for sample in self.samples:
            sample.short_name = sample.name[len(prefix):]

        template_name = self.template.name
        self.template.short_name = template_name[len(prefix):]

    def show_sample_and_figure(self):
        sample = self.current_sample()
        if sample is None:
            self.ui.pushButton_set_template.setEnabled(False)
            self.ui.toolButton_del.setEnabled(False)
            return
        elif sample == "应变":
            self.ui.pushButton_set_template.setEnabled(False)
            self.ui.toolButton_del.setEnabled(False)
            self.plot_yingbian()
            return
        elif sample == "泊松比":
            self.ui.pushButton_set_template.setEnabled(False)
            self.ui.toolButton_del.setEnabled(False)
            self.plot_bosong()
            return

        elif sample.name == self.template.name:
            self.ui.pushButton_set_template.setEnabled(False)
            self.ui.toolButton_del.setEnabled(False)
            title = sample.short_name + " (模板)"

        else:
            self.ui.pushButton_set_template.setEnabled(True)
            self.ui.toolButton_del.setEnabled(True)
            title = sample.short_name

        self.show_sample(sample)
        self.plot(sample.fig[1 - self.row_analysis])

        print("当前样品：", title)
        self.ui.title.setText(title)

    def remove_sample(self, button):
        sample = self.current_sample()
        self.samples.remove(sample)
        self.update_sample_tree()

    def update_sample_tree(self):
        first_item = self.ui.tree.topLevelItem(0)
        for i in range(first_item.childCount()):
            child = first_item.takeChild(0)  # 从第一个节点中取出第一个子节点，直到没有子节点为止

        if self.template is not None:
            template_item = QTreeWidgetItem()
            template_item.setText(0, self.template.short_name)
            first_item.addChild(template_item)
            self.ui.tree.setCurrentItem(template_item)

        # 清空第二个节点下的所有子节点
        second_item = self.ui.tree.topLevelItem(1)  # 假设第二个节点的索引为 1
        if second_item is not None:
            for i in range(second_item.childCount()):
                child = second_item.takeChild(0)  # 从第二个节点中取出第一个子节点，直到没有子节点为止

        # 添加样本
        for sample in self.samples:
            sample_item = QTreeWidgetItem()
            sample_item.setText(0, sample.short_name)
            sample_item.setText(1, sample.matching_accuracy)

            if not isinstance(sample.width, str):
                width = f"{sample.width:.1f}"
            else:
                width = sample.width
            if not isinstance(sample.height, str):
                height = f"{sample.height:.1f}"
            else:
                height = sample.height
            sample_item.setText(2, width)
            sample_item.setText(3, height)
            second_item.addChild(sample_item)

        self.ui.tree.expandAll()

    def current_sample(self):
        current_item = self.ui.tree.currentItem()
        if current_item is None:
            return None
        parent = current_item.parent()
        if parent is not None:
            if parent.text(0) == "模板":
                return self.template
            elif parent.text(0) == "数据":
                if current_item.text(0) == "应变":
                    return "应变"
                elif current_item.text(0) == "泊松比":
                    return "泊松比"
            else:
                index = parent.indexOfChild(current_item)
                if self.samples:
                    return self.samples[index]
        return None

    def show_sample(self, sample=None):
        # 清除场景中的所有项目
        self.scene_sample.clear()

        if sample is None:
            pixmap = QPixmap("resources/add_template.png")
            image_width = pixmap.width()
            image_height = pixmap.height()
            scale_factor = 1
            pixmap_item = self.scene_sample.addPixmap(pixmap)
            self.ui.graphicsView_sample.setScene(self.scene_sample)
        else:
            self.scene_sample.clear()
            image = sample.mark_image(self.row_analysis)

            # 获取视图和场景的大小
            view_width = self.ui.graphicsView_sample.width() - 8
            view_height = self.ui.graphicsView_sample.height() - 8

            # 获取图像的原始大小
            image_width = image.shape[1]
            image_height = image.shape[0]

            # 计算缩放比例，保持长宽比不变
            scale_factor = min(view_width / image_width, view_height / image_height)

            # 将图像从 BGR 格式转换为 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 缩放图像并使用平滑插值方法
            scaled_image = cv2.resize(image_rgb, (int(image_width * scale_factor), int(image_height * scale_factor)),
                                      interpolation=cv2.INTER_LINEAR)

            # 将 OpenCV 格式的图像转换为 QImage 格式
            q_image = QImage(scaled_image, scaled_image.shape[1], scaled_image.shape[0], scaled_image.shape[1] * 3,
                             QImage.Format_RGB888)

            # 创建 QPixmap 对象
            pixmap = QPixmap.fromImage(q_image)

            # 创建 QGraphicsPixmapItem 对象
            pixmap_item = QGraphicsPixmapItem(pixmap)

            # 添加图像项到场景中
            self.scene_sample.addItem(pixmap_item)

        # 将场景的中心点移动到视图中央
        self.scene_sample.setSceneRect(0, 0, image_width * scale_factor, image_height * scale_factor)
        self.ui.graphicsView_sample.setAlignment(Qt.AlignCenter)

    def plot(self, fig=None):
        # 创建 Matplotlib 图形
        if fig is None:
            fig = plt.Figure(figsize=(10, 6), tight_layout=True)  # tight_layout: 用于去除画图时两边的空白
            ax = fig.add_subplot(111)  # 添加子图
            ax.set_ylim(0, 255)
            ax.set_xlim(0, 100)
            ax.set_title('Red Channel Analysis')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Red Value')
            ax.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
            ax.spines['right'].set_visible(False)  # 去掉绘图时右面的横线

        # 将 Matplotlib 图形转换为 Qt 控件
        canvas = FigureCanvas(fig)
        canvas.setGeometry(0, 0, 500, 300)  # 设置图形大小为 figsize=(10, 6) 的宽度和高度的十倍

        # 将图形添加到 QGraphicsScene 中
        self.scene_figure.addWidget(canvas)
        # 调整场景大小与图像大小一致
        self.ui.graphicsView_figure.setSceneRect(0, 0, 500, 300)  # 设置场景大小与 canvas 大小相同
        self.ui.graphicsView_figure.show()

    def plot_yingbian(self):
        fig = plt.Figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.set_title('Strain Data')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Strain (%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data = self.export_data()

        # 散点图
        ax.plot(data["εx"] * 100, 'o', label='εx')

        self.plot(fig)

    def plot_bosong(self):
        fig = plt.Figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel('width')
        ax.set_ylabel('height')
        ax.set_title('Poisson Ratio')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data = self.export_data()
        width = np.array(data["width"]).reshape(-1, 1)
        height = np.array(data["height"]).reshape(-1, 1)

        # 去掉nan值
        nan_arr = np.isnan(width) | np.isnan(height)
        width = width[~nan_arr]
        height = height[~nan_arr]

        if len(width) == 0:
            self.plot(fig)
            return

        # 散点图
        ax.plot(width, height, 'o', label='data')

        # 使用线性回归模型拟合数据
        model = LinearRegression()
        model.fit(width.reshape(-1, 1), height)

        # 画出拟合直线
        x_fit = np.linspace(width.min()-1, width.max()+1, 100)
        y_fit = model.predict(x_fit.reshape(-1, 1))
        slope = -model.coef_[0]

        ax.plot(x_fit, y_fit, color='red', label='fit')

        ax.set_title(f'Poisson Ratio: {slope:.6f}')
        ax.legend()

        self.plot(fig)

    def save_temp(self):
        if self.template is not None:
            template_path = self.template.path
        else:
            template_path = None
        sample_paths = [sample.path for sample in self.samples]

        data = {
            "config": self.config,
            "history_path": self.history_path,
            "template_path": template_path,
            "sample_paths": sample_paths
        }
        with open("temp/temp_project.json", "w") as f:
            json.dump(data, f)

    def load_temp(self):
        with open("temp/temp_project.json", "r") as f:
            data = json.load(f)
            self.config = data["config"]
            self.history_path = data["history_path"]
            template_path = data["template_path"]
            sample_paths = data["sample_paths"]

        filter_sample_paths = [path for path in sample_paths if os.path.exists(path)]

        if filter_sample_paths:
            self.import_sample(sample_paths)
            if template_path is not None and os.path.exists(template_path) and self.template.path != template_path:
                self.set_template(template_path)

        if self.config is not None:
            self.config2value()

    def import_config(self, config, row_analysis):
        self.seg_index = config["index"]
        self.config[1-row_analysis]["position"] = config["position"]
        self.config[1-row_analysis]["left_edge"] = config["left_edge"]
        self.config[1-row_analysis]["right_edge"] = config["right_edge"]

        self.config2value()
        self.ui.statusbar.showMessage("模板边缘选择成功！")

    def config2value(self):
        self.ui.spinBox_position.setValue(self.config[0]["position"])
        self.ui.spinBox_edge1.setValue(self.config[0]["left_edge"])
        self.ui.spinBox_edge2.setValue(self.config[0]["right_edge"])
        self.ui.spinBox_width.setValue(self.config[0]["width"])

        self.ui.spinBox_position_2.setValue(self.config[1]["position"])
        self.ui.spinBox_edge1_2.setValue(self.config[1]["left_edge"])
        self.ui.spinBox_edge2_2.setValue(self.config[1]["right_edge"])
        self.ui.spinBox_width_2.setValue(self.config[1]["width"])

    def value2config(self):
        config = {
            "position": self.ui.spinBox_position.value(),
            "left_edge": self.ui.spinBox_edge1.value(),
            "right_edge": self.ui.spinBox_edge2.value(),
            "width": self.ui.spinBox_width.value()
        }

        config_2 = {
            "position": self.ui.spinBox_position_2.value(),
            "left_edge": self.ui.spinBox_edge1_2.value(),
            "right_edge": self.ui.spinBox_edge2_2.value(),
            "width": self.ui.spinBox_width_2.value()
        }

        self.config = [config, config_2]

    def quit(self):
        self.save_temp()

    def new_project(self):
        self.template = None
        self.samples = []
        self.seg_index = None

        self.ui.graphicsView_sample.setCursor(Qt.PointingHandCursor)  # 将光标设置为点击手势

        self.update_sample_tree()
        self.show_sample()
        self.plot()

    def open_project(self):
        pass

    def save_project(self):
        pass

    def save_as_project(self):
        pass


# QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Strain Meter")
app = QApplication([])
main_window = MainWindow()
main_window.ui.show()
app.exec_()
