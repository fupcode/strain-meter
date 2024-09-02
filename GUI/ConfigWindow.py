import cv2
from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QLabel, QPushButton, \
    QVBoxLayout, QWidget, QGraphicsItem, QListWidgetItem, QListWidget, QHBoxLayout, QSizePolicy, QListView, \
    QAbstractItemView
from PySide2.QtGui import QPixmap, QImage, QColor, QPen, QRadialGradient, QBrush
from PySide2.QtCore import Qt, QRectF


class PixelItem(QGraphicsItem):
    def __init__(self, x, y, color, pixel_scale, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.color = color
        self.selected = False
        self.pixel_scale = pixel_scale

    def boundingRect(self):
        return QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)

    def paint(self, painter, option, widget=None):
        # 绘制像素块
        scaled_rect = QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)
        painter.fillRect(scaled_rect, self.color)

        # 如果选中状态为 True，则绘制外边框
        if self.selected:
            # 定义渐变
            gradient = QRadialGradient(scaled_rect.center(), self.pixel_scale / 2)
            gradient.setColorAt(0, QColor(Qt.transparent))
            gradient.setColorAt(0.5, QColor(Qt.green))
            gradient.setColorAt(1, QColor(Qt.transparent))

            # 设置画笔和渐变
            pen = QPen(QBrush(gradient), 3)  # 根据放大倍率调整笔刷宽度
            painter.setPen(pen)
            painter.setBrush(Qt.transparent)

            # 绘制渐变圆
            outer_rect = scaled_rect.adjusted(self.pixel_scale / 8, self.pixel_scale / 8, -self.pixel_scale / 8,
                                              -self.pixel_scale / 8)
            painter.drawEllipse(outer_rect)

    def hoverEnterEvent(self, event):
        self.update()

    def hoverLeaveEvent(self, event):
        self.update()

    def setSelected(self, selected):
        # 设置选中状态并更新绘制
        self.selected = selected
        self.update()


class ImageViewer(QGraphicsView):
    def __init__(self, pixel_scale=16):
        super().__init__()
        self.segmentations = None
        self.index = 0
        self.image_shape = None
        self.setMouseTracking(True)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.scroll_factor = pixel_scale / 10
        self.selected_row_lines = [None, None]  # 用于存储被选中的行
        self.selected_pixel = None  # 用于存储被选中的像素
        self.previous_selected_pixel = None  # 用于存储之前被选中的像素
        self.pixel_scale = pixel_scale  # 像素放大倍率
        self.row_analysis = True  # 是否进行行分析
        self.process = 0

    def drawRowBoundaries(self):
        if self.selected_row_lines is not None and not self.process:
            pen = QPen(QColor(200, 0, 0))
            x = self.selected_pixel.x
            y = self.selected_pixel.y
            if self.selected_row_lines:
                self.scene.removeItem(self.selected_row_lines[0])
                self.scene.removeItem(self.selected_row_lines[1])
                self.selected_row_lines.clear()

            if self.row_analysis:
                # 绘制行边界
                self.selected_row_lines.append(
                    self.scene.addLine(0, y, self.image_shape[1] * self.pixel_scale,
                                       y, pen))  # 绘制上边界
                self.selected_row_lines.append(
                    self.scene.addLine(0, y + self.pixel_scale, self.image_shape[1] * self.pixel_scale,
                                       y + self.pixel_scale, pen))  # 绘制下边界

            else:
                # 绘制列边界
                self.selected_row_lines.append(
                    self.scene.addLine(x, 0, x,
                                       self.image_shape[0] * self.pixel_scale, pen))  # 绘制左边界
                self.selected_row_lines.append(
                    self.scene.addLine(x + self.pixel_scale, 0, x + self.pixel_scale,
                                       self.image_shape[0] * self.pixel_scale, pen))  # 绘制右边界

    def draw_image(self):
        image = self.segmentations[self.index]["image"]
        self.image_shape = image.shape
        height, width, channels = image.shape

        self.scene.clear()
        for y in range(height):
            for x in range(width):
                color = QColor(*image[y, x][::-1])  # OpenCV使用BGR格式
                item = PixelItem(x * self.pixel_scale, y * self.pixel_scale, color, self.pixel_scale)
                self.scene.addItem(item)

        self.setSceneRect(self.scene.itemsBoundingRect())  # 设置场景矩形为所有项的边界矩形
        self.align_center()

    def align_center(self):
        view_rect = self.viewport().rect()  # 获取视图的矩形区域
        scene_rect = self.sceneRect()  # 获取场景的矩形区域

        # 计算场景在视图中央的偏移量
        dx = (view_rect.width() - scene_rect.width()) / 2 - scene_rect.x()
        dy = (view_rect.height() - scene_rect.height()) / 2 - scene_rect.y()

        # 调整场景的偏移量，使其在视图中央
        self.translate(dx, dy)

    def switch_image(self, index):
        # 重置对之前图像项的引用
        self.selected_row_lines = [None, None]
        self.selected_pixel = None
        self.previous_selected_pixel = None

        # 设置新的图像索引
        self.index = index

        # 绘制新的图像
        self.draw_image()

        # 重设缩放大小
        self.resetTransform()

    def start(self, segmentations, row_analysis):
        self.segmentations = segmentations
        self.row_analysis = row_analysis
        self.process = 0

        self.switch_image(self.index)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            # 进行缩放
            scale_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale(scale_factor, scale_factor)
        else:
            super().wheelEvent(event)

    def scrollContentsBy(self, dx: int, dy: int):
        super().scrollContentsBy(dx, dy)
        if self.selected_row_lines and not self.process:
            self.scene.removeItem(self.selected_row_lines[0])
            self.scene.removeItem(self.selected_row_lines[1])
            self.selected_row_lines.clear()

    def mousePressEvent(self, event):
        item = self.find_item_with_event_pos(event.pos())
        if item is None:
            return

        # 之前选中的像素的标记
        if self.selected_pixel:
            if self.previous_selected_pixel is None and self.process:
                self.previous_selected_pixel = self.selected_pixel
            else:
                self.selected_pixel.setSelected(False)

        if self.process:
            # 另选一个像素
            if self.row_analysis:
                x = item.x
                y = self.previous_selected_pixel.y
            else:
                y = item.y
                x = self.previous_selected_pixel.x
            self.selected_pixel = self.find_item_with_pixel_pos((x, y))
        else:
            self.selected_pixel = item
        self.selected_pixel.setSelected(True)
        self.drawRowBoundaries()

    def find_item_with_event_pos(self, pos):
        items = self.items(pos)
        for item in items:
            if isinstance(item, PixelItem):
                return item
        return None

    def find_item_with_pixel_pos(self, pos):
        x, y = pos
        for item in self.scene.items():
            if isinstance(item, PixelItem):
                if item.x == x and item.y == y:
                    return item
        return None

    def get_relative_position(self, pos):
        relative_x = pos[0] - self.horizontalScrollBar().value()
        relative_y = pos[1] - self.verticalScrollBar().value()
        return relative_x, relative_y

    def return_config(self):
        image = self.segmentations[self.index]["image"]
        start = self.segmentations[self.index]["start"]
        height, width, _ = image.shape
        region = (start[0], start[1], width, height)

        x1 = self.previous_selected_pixel.x // self.pixel_scale
        y1 = self.previous_selected_pixel.y // self.pixel_scale
        x2 = self.selected_pixel.x // self.pixel_scale
        y2 = self.selected_pixel.y // self.pixel_scale

        if self.row_analysis:
            config = {
                "index": self.index,
                "position": y1,
                "left_edge": x1,
                "right_edge": x2,
                "ROI_image": image,
                "ROI_region": region
            }

        else:
            config = {
                "index": self.index,
                "position": x1,
                "left_edge": y1,
                "right_edge": y2,
                "ROI_image": image,
                "ROI_region": region
            }

        return config


class ConfigWindow(QMainWindow):
    def __init__(self, pixel_scale=8, mainWindow=None):
        super().__init__()
        self.row_analysis = True
        self.parent = mainWindow
        self.pixel_scale = pixel_scale
        self.setWindowTitle("边缘位置选取")
        self.setWindowIcon(QPixmap("resources/icon.ico"))

        # 设置窗口大小
        self.desktop_width_08 = int(QApplication.desktop().width() * 0.8)
        self.desktop_height_08 = int(QApplication.desktop().height() * 0.8)
        self.setFixedSize(self.desktop_width_08, self.desktop_height_08)

        # 添加图像显示器
        self.image_viewer = ImageViewer(pixel_scale)
        self.setCentralWidget(self.image_viewer)

        # 添加显示文字的标签
        self.label = QLabel("选择边缘", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(400, 30)

        # 设置标签的尺寸策略
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label.setFixedHeight(30)  # 设置标签的固定高度
        self.label.setMinimumWidth(400)  # 设置标签的最小宽度

        # 添加确认按钮
        self.confirm_button = QPushButton("下一步", self)
        self.confirm_button.setFixedSize(150, 50)
        self.confirm_button.clicked.connect(self.confirm_click)

        # 创建缩略图列表
        self.thumbnail_list_width = 200  # 创建缩略图列表
        self.thumbnail_list = QListWidget(self)
        self.thumbnail_list.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置选择模式为单选
        self.thumbnail_list.itemClicked.connect(self.thumbnail_clicked)

        # 设置列表项的大小策略
        self.thumbnail_list.setResizeMode(QListView.Adjust)
        self.thumbnail_list.setSpacing(8)

        # 设置缩略图列表的尺寸策略
        self.thumbnail_list.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.thumbnail_list.setFixedWidth(self.thumbnail_list_width)  # 设置缩略图列表的固定宽度

        # 创建水平布局，包含缩略图列表和垂直布局中的其他组件
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.thumbnail_list)

        # 创建垂直布局，包含标签、图像视图和确认按钮
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        v_layout.addWidget(self.image_viewer)
        v_layout.addWidget(self.confirm_button, alignment=Qt.AlignHCenter)

        # 将垂直布局添加到水平布局中
        h_layout.addLayout(v_layout)

        # 创建一个 QWidget 作为中心窗口
        central_widget = QWidget()
        central_widget.setLayout(h_layout)
        self.setCentralWidget(central_widget)

    def set_window_size(self, shape):
        image_height, image_width, _ = shape

        need_height = image_height * self.pixel_scale
        need_width = image_width * self.pixel_scale

        change = False
        if need_height < self.desktop_height_08:
            change = True
            window_height = need_height + 80
        else:
            window_height = self.desktop_height_08

        if need_width < self.desktop_width_08:
            change = True
            window_width = need_width + 80 + self.thumbnail_list_width
        else:
            window_width = self.desktop_width_08

        if change:
            self.setFixedSize(window_width, window_height)

    def set_prompt(self):
        if self.image_viewer.process == 0:
            if self.row_analysis:
                self.label.setText("请点击选择左端点")
            else:
                self.label.setText("请点击选择上端点")
            self.confirm_button.setText("下一步")
        else:
            if self.row_analysis:
                self.label.setText("请点击选择右端点")
            else:
                self.label.setText("请点击选择下端点")
            self.confirm_button.setText("完成")

    def start(self, segmentations, row_analysis):
        self.row_analysis = row_analysis
        self.image_viewer.process = 0
        shape = segmentations[0]["image"].shape
        self.set_window_size(shape)
        self.set_prompt()

        self.show_thumbnails(segmentations)

        self.image_viewer.start(segmentations, row_analysis)

        self.show()

    def confirm_click(self):
        if self.image_viewer.process == 0:
            self.image_viewer.process = 1
            self.set_prompt()

        else:
            config = self.image_viewer.return_config()
            self.parent.import_config(config, self.row_analysis)

            self.hide()

    def show_thumbnails(self, segmentations):
        # 清空缩略图列表
        self.thumbnail_list.clear()

        # 添加缩略图项
        for index, seg in enumerate(segmentations):
            thumbnail = self.generate_thumbnail(seg["image"])
            item = QListWidgetItem()
            item.setData(Qt.UserRole, index)  # 存储图像索引
            self.thumbnail_list.addItem(item)
            self.thumbnail_list.setItemWidget(item, thumbnail)
            item.setSizeHint(thumbnail.sizeHint())

    def generate_thumbnail(self, image):
        # 将 OpenCV 格式的图像转换为 QImage 格式
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # 计算缩略图的高度，以保持宽高比并且与组件宽度一致
        thumbnail_width = self.thumbnail_list.width() - 30
        thumbnail_height = int(height * thumbnail_width / width)
        q_image = q_image.scaled(thumbnail_width, thumbnail_height, Qt.KeepAspectRatio)

        # 创建 QLabel 来显示缩略图
        thumbnail = QLabel()
        thumbnail.setPixmap(QPixmap.fromImage(q_image))

        # 设置边框样式
        thumbnail.setStyleSheet("border: 4px solid transparent;")  # 默认为透明边框

        return thumbnail

    def thumbnail_clicked(self, item):
        # 获取点击的缩略图项的数据
        index = item.data(Qt.UserRole)

        if self.image_viewer.index != index:
            self.thumbnail_set_selected_style(item)
            # 更新标签
            self.image_viewer.process = 0
            self.set_prompt()
            # 切换到对应图像
            self.image_viewer.switch_image(index)

    def thumbnail_set_selected_style(self, item):
        # 清除之前选中项的边框样式
        for i in range(self.thumbnail_list.count()):
            widget_item = self.thumbnail_list.item(i)
            if widget_item is not None:
                widget = self.thumbnail_list.itemWidget(widget_item)
                thumbnail_label = widget.findChild(QLabel)
                if thumbnail_label is not None:
                    thumbnail_label.setStyleSheet("border: 4px solid transparent;")

        # 设置选中项的边框样式
        widget = self.thumbnail_list.itemWidget(item)
        thumbnail_label = widget.findChild(QLabel)
        if thumbnail_label is not None:
            thumbnail_label.setStyleSheet("border: 4px solid lightblue;")


if __name__ == "__main__":
    class Main:
        def import_config(self, config, row_analysis):
            print(config)


    app = QApplication([])
    main = Main()
    image = cv2.imread("resources/add_template.png")
    window = ConfigWindow(8, mainWindow=main)
    window.start(image, True)
    app.exec_()
