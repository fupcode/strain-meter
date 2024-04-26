
import threading
from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QPushButton, QVBoxLayout, QWidget
from PySide2.QtGui import QColor, QPainter, QPen
from PySide2.QtCore import QRectF, Qt

class PixelItem(QGraphicsItem):
    def __init__(self, x, y, color, pixel_scale, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.color = color
        self.pixel_scale = pixel_scale

    def boundingRect(self):
        return QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)

    def paint(self, painter, option, widget=None):
        scaled_rect = QRectF(self.x, self.y, self.pixel_scale, self.pixel_scale)
        painter.fillRect(scaled_rect, self.color)

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

    def add_pixel_item(self, x, y, color, pixel_scale):
        item = PixelItem(x * pixel_scale, y * pixel_scale, color, pixel_scale)
        self.scene.addItem(item)

def long_running_task(queue):
    # 模拟创建大量 GUI 对象
    for y in range(50):
        for x in range(50):
            queue.put((x, y, QColor(Qt.red), 16))

def main():
    # 创建 Qt 应用程序
    app = QApplication([])

    # 创建主窗口
    window = QMainWindow()
    window.setWindowTitle("ImageViewer")
    window.setGeometry(100, 100, 800, 600)

    # 创建 ImageViewer 实例
    image_viewer = ImageViewer()
    window.setCentralWidget(image_viewer)

    # 创建队列用于通信
    queue = queue.Queue()

    # 创建后台线程来执行耗时任务
    thread = threading.Thread(target=long_running_task, args=(queue,))
    thread.start()

    # 主循环
    while True:
        try:
            # 从队列中获取 GUI 对象并显示
            x, y, color, pixel_scale = queue.get(timeout=1)
            image_viewer.add_pixel_item(x, y, color, pixel_scale)
        except queue.Empty:
            # 队列为空时跳出循环
            break

    # 显示主窗口
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
