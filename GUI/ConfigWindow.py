import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QLabel, QPushButton, \
    QVBoxLayout, QWidget, QGraphicsItem, QListWidgetItem, QListWidget, QHBoxLayout, QSizePolicy, QListView, \
    QAbstractItemView
from PySide2.QtGui import QPixmap, QImage, QColor, QPen, QRadialGradient, QBrush
from PySide2.QtCore import Qt, QRectF
import cv2


class PixelItem(QGraphicsItem):
    def __init__(self, x, y, color, pixel_scale, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.color = color
        self.selected = False
        self.pixel_scale = pixel_scale

        self.keypoint = False

    def boundingRect(self):
        return QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)

    def paint(self, painter, option, widget=None):
        # 绘制像素块
        scaled_rect = QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)
        painter.fillRect(scaled_rect, self.color)

        # 如果选中状态为 True，则绘制外边框
        if self.keypoint:
            if self.selected:
                color = Qt.red
            else:
                color = Qt.gray
            # 定义渐变
            gradient = QRadialGradient(scaled_rect.center(), self.pixel_scale / 2)
            gradient.setColorAt(0, QColor(Qt.transparent))
            gradient.setColorAt(0.5, QColor(color))
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

    def setKeypoint(self, keypoint):
        self.keypoint = keypoint
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
        self.pixel_scale = pixel_scale  # 像素放大倍率

        self.sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.005)
        self.keypoints = None
        self.keypoints_info = {}
        self.descriptors = None

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

        # 设置新的图像索引
        self.index = index

        # 绘制新的图像
        self.draw_image()

        # 更新关键点
        self.reset_keypoints()

        # 重设缩放大小
        self.resetTransform()

    def start(self, segmentations):
        self.segmentations = segmentations

        self.switch_image(0)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            # 进行缩放
            scale_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale(scale_factor, scale_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        item = self.find_item_with_event_pos(event.pos())
        if item is None:
            return

        x = item.x
        y = item.y

        keypoint, key_item, pos, selected = self.find_key_with_pixel_pos((x, y))
        if keypoint is None:
            return

        key_item.setSelected(not selected)
        self.keypoints_info[pos]["selected"] = not selected

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

    def reset_keypoints(self):
        image = self.segmentations[self.index]["image"]
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        self.keypoints = keypoints
        self.descriptors = descriptors

        self.keypoints_info = {}
        for keypoint in keypoints:
            x, y = keypoint.pt
            x = int(x) * self.pixel_scale
            y = int(y) * self.pixel_scale
            item = self.find_item_with_pixel_pos((x, y))
            self.keypoints_info[(x, y)] = {
                "keypoint": keypoint,
                "selected": True
            }

            item.setKeypoint(True)
            item.setSelected(True)

    def find_key_with_pixel_pos(self, pos):
        x, y = pos
        x = x // self.pixel_scale
        y = y // self.pixel_scale
        for keypoint in self.keypoints:
            if abs(keypoint.pt[0] - x) < 1.5 and abs(keypoint.pt[1] - y) < 1.5:
                pixel_pos = (int(keypoint.pt[0]) * self.pixel_scale, int(keypoint.pt[1]) * self.pixel_scale)
                item = self.find_item_with_pixel_pos(pixel_pos)
                return keypoint, item, pixel_pos, item.selected
        return None, None, None, None

    def return_config(self):
        selected_keypoints = []
        descriptors = []  # 使用列表存储描述符向量
        for i, keypoint in enumerate(self.keypoints):
            x, y = keypoint.pt
            x = int(x) * self.pixel_scale
            y = int(y) * self.pixel_scale
            if self.keypoints_info[(x, y)]["selected"]:
                selected_keypoints.append(keypoint)
                descriptors.append(self.descriptors[i])  # 将描述符添加到列表中

        # 将描述符列表转换为 numpy 数组
        descriptors = np.array(descriptors)

        segmentation = self.segmentations[self.index]
        height, width, _ = segmentation["image"].shape
        start_x, start_y = segmentation["start"]
        region = [start_x, start_y, width, height]
        config = {
            "ROI": region,
            "seg_index": self.index,
            "keypoints": selected_keypoints,
            "descriptors": descriptors
        }
        return config


class ConfigWindow(QMainWindow):
    def __init__(self, pixel_scale=8, mainWindow=None):
        super().__init__()
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
        self.label = QLabel("选择特征点（红色：已选中；灰色：未选中）", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(400, 30)

        # 设置标签的尺寸策略
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label.setFixedHeight(30)  # 设置标签的固定高度
        self.label.setMinimumWidth(400)  # 设置标签的最小宽度

        # 添加确认按钮
        self.confirm_button = QPushButton("完成", self)
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

    def start(self, segmentations):
        shape = segmentations[0]["image"].shape
        self.set_window_size(shape)

        self.show_thumbnails(segmentations)

        self.image_viewer.start(segmentations)

        self.show()

    def confirm_click(self):
        config = self.image_viewer.return_config()
        self.parent.import_config(config)

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
    pass
