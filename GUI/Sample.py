import matplotlib.pyplot as plt
import fpc.pyplot_template
import cv2
import numpy as np
import os
import traceback


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def cv2_imwrite(file_path, img):
    cv2.imencode('.png', img)[1].tofile(file_path)


def one_dimensional_analysis(image, config, row_analysis):
    def find_extremes(arr, mode='left', start=None, end=None):

        def last_argmin(arr):
            min_value = np.min(arr)
            min_indices = np.where(arr == min_value)[0]
            last_min_index = max(min_indices)
            return last_min_index

        def last_argmax(arr):
            max_value = np.max(arr)
            max_indices = np.where(arr == max_value)[0]
            last_max_index = max(max_indices)
            return last_max_index

        if start is None:
            start = 0
        if end is None:
            end = len(arr)

        if mode == 'left':
            # 从最低点的右边寻找最高点
            min_index = last_argmin(arr[start:end])
            max_index = min_index + np.argmax(arr[start + min_index:end])
        else:
            # 从最低点的左边寻找最高点
            min_index = np.argmin(arr[start:end])
            max_index = last_argmax(arr[start:end])

        min_index += start
        max_index += start

        min_value = arr[min_index]
        max_value = arr[max_index]

        return min_index, min_value, max_index, max_value

    def find_intersection(data, target_value, x_range):
        # Convert all inputs to float to avoid overflow when subtracting
        data = [int(x) for x in data]
        target_value = target_value
        x_min, x_max = map(int, x_range)

        min_dist = float('inf')
        x_value = None
        prev_x, prev_y = None, None

        for cur_x in range(x_min, x_max + 1):
            if cur_x < len(data):
                cur_y = data[cur_x]
                if prev_y is not None and ((prev_y - target_value) * (cur_y - target_value) <= 0):
                    if abs(cur_y - prev_y) < 1e-10:
                        approx_x = cur_x if abs(cur_y - target_value) < abs(prev_y - target_value) else prev_x
                    else:
                        approx_x = (target_value - prev_y) * (cur_x - prev_x) / (cur_y - prev_y) + prev_x
                    if abs(approx_x - (x_max + x_min) / 2) < min_dist:
                        min_dist = abs(approx_x - (x_max + x_min) / 2)
                        x_value = approx_x
                prev_x, prev_y = cur_x, cur_y

        return x_value

    # 读取图像
    image_array = np.array(image)
    image_width, image_height, _ = image_array.shape
    distance = "?"
    details = None

    # 创建 Matplotlib 图形
    fig = plt.Figure(figsize=(10, 6), tight_layout=True)  # tight_layout: 用于去除画图时两边的空白
    ax = fig.add_subplot(111)  # 添加子图

    # 获取配置信息
    position = config["position"]
    left_edge = config["left_edge"]
    right_edge = config["right_edge"]
    width = config["width"]
    left_width = width
    right_width = width

    if left_edge - left_width < 0:
        left_width = left_edge
    if right_edge + right_width >= image_width:
        right_width = image_width - right_edge - 1

    x_min = left_edge - 3 * left_width
    x_max = right_edge + 3 * right_width
    if x_min < 0:
        x_min = 0
    if x_max >= image_width:
        x_max = image_width - 1
    ax.set_xlim(x_min, x_max)

    # 获取图像的红色通道
    if row_analysis:
        red_channel = image_array[position, :, 2]  # 仅取第一行红色通道
    else:
        red_channel = image_array[:, position, 2]
    ax.plot(red_channel, 'r-')

    try:
        # 分析左边缘
        left_min_x, left_min_value, left_max_x, left_max_value = find_extremes(red_channel, mode='left',
                                                                               start=left_edge - left_width,
                                                                               end=left_edge + left_width)

        left_edge_y = left_min_value / 2 + left_max_value / 2  # y坐标为最高点和最低点的中间值
        left_edge_x = find_intersection(red_channel, left_edge_y, (left_min_x, left_max_x))  # 计算x坐标

        # 分析右边缘
        right_min_x, right_min_value, right_max_x, right_max_value = find_extremes(red_channel, mode='right',
                                                                                   start=right_edge - right_width,
                                                                                   end=right_edge + right_width)

        right_edge_y = right_min_value / 2 + right_max_value / 2  # y坐标为最高点和最低点的中间值
        right_edge_x = find_intersection(red_channel, right_edge_y, (right_max_x, right_min_x))  # 计算x坐标

        # 绘制图像
        if left_edge_x is None or right_edge_x is None:
            raise RuntimeError("未找到边缘")

        else:
            ax.plot([left_max_x - left_width, left_max_x + left_width], [left_max_value, left_max_value], 'b-')
            ax.plot([left_min_x - left_width, left_min_x + left_width], [left_min_value, left_min_value], 'b-')
            # 标记最高点
            ax.plot(left_max_x, left_max_value, 'bo')
            ax.annotate("$R_{max}$", xy=(left_max_x, left_max_value), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=12, color='b')
            # 标记最低点
            ax.plot(left_min_x, left_min_value, 'bo')
            ax.annotate("$R_{min}$", xy=(left_min_x, left_min_value), xytext=(0, -16), textcoords='offset points',
                        ha='center', fontsize=12, color='b')
            # 标记交点
            ax.plot(left_edge_x, left_edge_y, 'bo')
            ax.annotate("$R_{edge}$", xy=(left_edge_x, left_edge_y), xytext=(-24, -4), textcoords='offset points',
                        ha='center', fontsize=12, color='b')

            ax.plot([right_max_x - right_width, right_max_x + right_width], [right_max_value, right_max_value], 'g-')
            ax.plot([right_min_x - right_width, right_min_x + right_width], [right_min_value, right_min_value], 'g-')
            # 标记最高点
            ax.plot(right_max_x, right_max_value, 'go')
            ax.annotate("$R_{max}^{'}$", xy=(right_max_x, right_max_value), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=12, color='g')
            # 标记最低点
            ax.plot(right_min_x, right_min_value, 'go')
            ax.annotate("$R_{min}^{'}$", xy=(right_min_x, right_min_value), xytext=(0, -16), textcoords='offset points',
                        ha='center', fontsize=12, color='g')
            # 标记交点
            ax.plot(right_edge_x, right_edge_y, 'go')
            ax.annotate("$R_{edge}^{'}$", xy=(right_edge_x, right_edge_y), xytext=(24, -4), textcoords='offset points',
                        ha='center', fontsize=12, color='g')

            # 设定x轴范围
            x_min = left_edge_x - 3 * left_width
            x_max = right_edge_x + 3 * right_width
            if x_min < 0:
                x_min = 0
            if x_max >= image_width:
                x_max = image_width - 1
            ax.set_xlim(x_min, x_max)

            # 设定标记竖线最高高度
            y_axis_min, y_axis_max = ax.get_ylim()
            y_axis_min -= 4
            y_axis_max += 4
            ax.set_ylim(y_axis_min, y_axis_max)

            left_edge_y_normalized = (left_edge_y - y_axis_min) / (y_axis_max - y_axis_min)
            right_edge_y_normalized = (right_edge_y - y_axis_min) / (y_axis_max - y_axis_min)

            ax.axvline(x=left_edge_x, ymin=0, ymax=left_edge_y_normalized, color='k', linestyle='--')
            ax.axvline(x=right_edge_x, ymin=0, ymax=right_edge_y_normalized, color='k', linestyle='--')

            # 计算测距双向箭头的起点和终点位置
            arrow_start_x = left_edge_x + 1.5
            arrow_end_x = right_edge_x - 1.5
            arrow_y = y_axis_min + 0.5 * (left_edge_y - y_axis_min)

            # 绘制双向箭头
            ax.arrow(arrow_start_x, arrow_y, arrow_end_x - arrow_start_x, 0, color='orange', width=0.03, head_width=0.5,
                     head_length=1)
            ax.arrow(arrow_end_x, arrow_y, -(arrow_end_x - arrow_start_x), 0, color='orange', width=0.03,
                     head_width=0.5,
                     head_length=1)

            # 标注长度
            distance = right_edge_x - left_edge_x
            # 保留位数
            distance = round(distance, 2)
            ax.annotate(f'L={distance}', xy=(left_edge_x + distance / 2, arrow_y),
                        xytext=(0, 16), textcoords='offset points', ha='center', fontsize=12, color='orange')

            print(f"L={distance}")
            details = {
                "position": position,
                "left_edge_x": round(left_edge_x),
                "right_edge_x": round(right_edge_x),
                "left_width": left_width,
                "right_width": right_width,
            }

    except Exception as e:
        print("RuntimeError:")
        print(traceback.format_exc())

        ax.axvline(x=left_edge, color='b', linestyle='--')
        ax.axvline(x=left_edge - left_width, color='b', linestyle='-')
        ax.axvline(x=left_edge + left_width, color='b', linestyle='-')

        ax.axvline(x=right_edge, color='g', linestyle='--')
        ax.axvline(x=right_edge - right_width, color='g', linestyle='-')
        ax.axvline(x=right_edge + right_width, color='g', linestyle='-')

    # 设置图像标题和轴标签
    ax.set_title('Red Channel Analysis')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Red Value')

    ax.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
    ax.spines['right'].set_visible(False)  # 去掉绘图时右面的横线

    return distance, fig, details


class Sample:
    def __init__(self, image_path):
        self.raw_image = cv2_imread(image_path)
        self.path = image_path
        self.cropped_image = None
        self.intelligent_cropped_image = None
        self.intelligent_crop_start = None
        self.scene = None
        self.matching_accuracy = None
        self.segmentation = []

        self.width = ''
        self.height = ''
        self.fig = [None, None]
        self.details = [None, None]

        name = os.path.basename(image_path)
        self.name = os.path.splitext(name)[0]
        self.short_name = self.name

    def rename(self, index, save=False):
        self.name = "Image_" + f"{index:02d}"
        self.short_name = f"{index:02d}"
        if save:
            os.rename(self.path, os.path.join(os.path.dirname(self.path), f"{self.name}.png"))
        self.path = os.path.join(os.path.dirname(self.path), f"{self.name}.png")

    def crop_by_template(self, template=None):
        if template is None:
            self.cropped_image = self.raw_image
            return
        # 模板匹配
        result = cv2.matchTemplate(self.raw_image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # 获取模板的宽度和高度
        h, w = template.shape[:2]

        # 计算匹配准确度并赋值给属性
        matching_accuracy = result[max_loc[1], max_loc[0]]
        self.matching_accuracy = f"{matching_accuracy * 100:.2f}%"

        # 裁剪图像
        cropped_image = self.raw_image[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
        self.cropped_image = np.ascontiguousarray(cropped_image)

    def crop_outer_border(self, crop_ratio=0.2):
        image_height, image_width, _ = self.raw_image.shape
        left = int(image_width * crop_ratio)
        right = int(image_width * (1 - crop_ratio))
        top = int(image_height * crop_ratio)
        bottom = int(image_height * (1 - crop_ratio))
        cropped_image = self.raw_image[top:bottom, left:right]

        return cropped_image

    def crop_by_self(self, crop_ratio=0.2):
        cropped_image = self.crop_outer_border(crop_ratio)
        self.cropped_image = np.ascontiguousarray(cropped_image)

    def crop_by_region(self, region):
        start_x, start_y, width, height = region
        cropped_image = self.cropped_image[start_y:start_y + height, start_x:start_x + width]
        return cropped_image

    def segment(self, crop_ratio=0.2, min_area_threshold=250, border=8):
        cropped_image = self.crop_outer_border(crop_ratio)
        height, width, _ = cropped_image.shape

        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # (5, 5) 是高斯核的大小，0 是标准差

        # 应用 bitwise_not
        bitwise_not_image = cv2.bitwise_not(gray_image)

        # 自适应均值阈值处理
        adaptive_threshold = cv2.adaptiveThreshold(bitwise_not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 25,
                                                   2)

        # 使用闭运算填充图像内部的小孔和断裂
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("adaptive_threshold", adaptive_threshold)
        # cv2.imshow("closed", closed)

        # 查找图像的轮廓
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建分割后的图像列表
        segmented_result = []

        # 遍历所有轮廓
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)

            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            # 检查轮廓是否与图像边界相连
            if x > border and y > border and x + w < width - border and y + h < height - border and area > min_area_threshold:
                # 加入分割后的图像列表
                start_x = max(0, x - border)
                start_y = max(0, y - border)
                end_x = min(x + w + border, width - 1)
                end_y = min(y + h + border, height - 1)
                segmented_image = cropped_image[start_y:end_y, start_x:end_x]
                segmented_image = np.ascontiguousarray(segmented_image)

                segmentation = {
                    "image": segmented_image,
                    "start": (start_x, start_y),
                    "area": area
                }
                segmented_result.append(segmentation)

        # 按面积降序排序
        segmented_result.sort(key=lambda x: x["area"], reverse=True)

        self.segmentation = segmented_result
        return segmented_result

    def mark_image(self, row_analysis, scale=4):
        details = self.details[1 - row_analysis]
        if details is None:
            return self.cropped_image

        image = self.cropped_image.copy()

        # 将原始图像放大为四倍
        half_scale = scale // 2
        new_width = image.shape[1] * scale
        new_height = image.shape[0] * scale
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # 标记
        position = int(details["position"] * scale)
        left_edge_x = int(details["left_edge_x"] * scale)
        right_edge_x = int(details["right_edge_x"] * scale)
        left_width = int(details["left_width"] * scale)
        right_width = int(details["right_width"] * scale)

        if row_analysis:
            cv2.line(resized_image, (0, position), (new_width, position), (0, 0, 255), 1)
            cv2.line(resized_image, (0, position + scale), (new_width, position + scale), (0, 0, 255), 1)

            cv2.circle(resized_image, (left_edge_x + half_scale, position + half_scale), half_scale, (0, 255, 0), 1)
            cv2.circle(resized_image, (right_edge_x + half_scale, position + half_scale), half_scale, (0, 255, 0), 1)

            x_min = left_edge_x - 3 * left_width
            x_max = right_edge_x + 3 * right_width
            if x_min < 0:
                x_min = 0
            if x_max >= new_width:
                x_max = new_width - 1

            crop_width = x_max - x_min
            crop_half_width = crop_width // 2

            y_min = max(0, position - crop_half_width)
            y_max = min(new_height, position + crop_half_width)

            result = resized_image[y_min:y_max, x_min:x_max]

        else:
            cv2.line(resized_image, (position, 0), (position, new_height), (0, 0, 255), 1)
            cv2.line(resized_image, (position + scale, 0), (position + scale, new_height), (0, 0, 255), 1)

            cv2.circle(resized_image, (position + half_scale, left_edge_x + half_scale), half_scale, (0, 255, 0), 1)
            cv2.circle(resized_image, (position + half_scale, right_edge_x + half_scale), half_scale, (0, 255, 0), 1)

            y_min = left_edge_x - 3 * left_width
            y_max = right_edge_x + 3 * right_width

            if y_min < 0:
                y_min = 0
            if y_max >= new_height:
                y_max = new_height - 1

            crop_height = y_max - y_min
            crop_half_height = crop_height // 2

            x_min = max(0, position - crop_half_height)
            x_max = min(new_width, position + crop_half_height)

            result = resized_image[y_min:y_max, x_min:x_max]

        return result

    def analysis(self, config, row_analysis):
        config = config[1 - row_analysis]
        fig = None
        details = None
        length = "error"

        try:
            length, fig, details = one_dimensional_analysis(self.cropped_image, config, row_analysis)
        except Exception as e:
            print("RuntimeError:")
            print(traceback.format_exc())

        if row_analysis:
            self.width = length
            self.fig[0] = fig
            self.details[0] = details
        else:
            self.height = length
            self.fig[1] = fig
            self.details[1] = details
        return length

    def get_data(self):
        if isinstance(self.width, str):
            width = np.NAN
        else:
            width = self.width
        if isinstance(self.height, str):
            height = np.NAN
        else:
            height = self.height
        return [self.short_name, width, height]

    def get_ROI_region(self, index):
        if index < len(self.segmentation):
            seg = self.segmentation[index]
            image = seg["image"]
            start = seg["start"]
            start_x, start_y = start
            height, width, _ = image.shape
            return start_x, start_y, width, height

        else:
            return None

    def save_ROI(self, region, dirname):
        cropped_image = self.crop_by_region(region)
        path = os.path.join(dirname, f"ROI_{self.name}.png")
        cv2_imwrite(path, cropped_image)


if __name__ == "__main__":
    path = "../figure_0311/Video Image--003 01.png"
    sample = Sample(path)
    sample.crop_by_self()
    segmentations = sample.segment()
    segmented_images = [segmentation["image"] for segmentation in segmentations]

    cv2.imshow("crop_image", sample.cropped_image)
    for i, segmented_image in enumerate(segmented_images):
        cv2.imshow(f"segmented_image_{i}", segmented_image)
        cv2.waitKey(0)  # 等待按键输入
        cv2.destroyWindow(f"segmented_image_{i}")  # 关闭窗口

    cv2.destroyAllWindows()
