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
    image_height, image_width, _ = image_array.shape
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

    if row_analysis:
        limit = image_width
    else:
        limit = image_height

    if left_edge - left_width < 0:
        left_width = left_edge
    if right_edge + right_width >= limit:
        right_width = image_width - right_edge - 1

    x_min = left_edge - 3 * left_width
    x_max = right_edge + 3 * right_width
    if x_min < 0:
        x_min = 0
    if x_max >= limit:
        x_max = limit - 1
    ax.set_xlim(x_min, x_max)

    # 获取图像的灰度图
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    if row_analysis:
        gray_scale = gray_image[position, :]
    else:
        gray_scale = gray_image[:, position]
    ax.plot(gray_scale, 'r-')

    try:
        # 分析左边缘
        left_min_x, left_min_value, left_max_x, left_max_value = find_extremes(gray_scale, mode='left',
                                                                               start=left_edge - left_width,
                                                                               end=left_edge + left_width)

        left_edge_y = left_min_value / 2 + left_max_value / 2  # y坐标为最高点和最低点的中间值
        left_edge_x = find_intersection(gray_scale, left_edge_y, (left_min_x, left_max_x))  # 计算x坐标

        # 分析右边缘
        right_min_x, right_min_value, right_max_x, right_max_value = find_extremes(gray_scale, mode='right',
                                                                                   start=right_edge - right_width,
                                                                                   end=right_edge + right_width)

        right_edge_y = right_min_value / 2 + right_max_value / 2  # y坐标为最高点和最低点的中间值
        right_edge_x = find_intersection(gray_scale, right_edge_y, (right_max_x, right_min_x))  # 计算x坐标

        # 绘制图像
        if left_edge_x is None or right_edge_x is None:
            raise RuntimeError("未找到边缘")

        else:
            ax.plot([left_max_x - left_width, left_max_x + left_width], [left_max_value, left_max_value], 'b-')
            ax.plot([left_min_x - left_width, left_min_x + left_width], [left_min_value, left_min_value], 'b-')
            # 标记最高点
            ax.plot(left_max_x, left_max_value, 'bo')
            ax.annotate("$G_{max}$", xy=(left_max_x, left_max_value), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=12, color='b')
            # 标记最低点
            ax.plot(left_min_x, left_min_value, 'bo')
            ax.annotate("$G_{min}$", xy=(left_min_x, left_min_value), xytext=(0, -16), textcoords='offset points',
                        ha='center', fontsize=12, color='b')
            # 标记交点
            ax.plot(left_edge_x, left_edge_y, 'bo')
            ax.annotate("$G_{edge}$", xy=(left_edge_x, left_edge_y), xytext=(-24, -4), textcoords='offset points',
                        ha='center', fontsize=12, color='b')

            ax.plot([right_max_x - right_width, right_max_x + right_width], [right_max_value, right_max_value], 'g-')
            ax.plot([right_min_x - right_width, right_min_x + right_width], [right_min_value, right_min_value], 'g-')
            # 标记最高点
            ax.plot(right_max_x, right_max_value, 'go')
            ax.annotate("$G_{max}^{'}$", xy=(right_max_x, right_max_value), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=12, color='g')
            # 标记最低点
            ax.plot(right_min_x, right_min_value, 'go')
            ax.annotate("$G_{min}^{'}$", xy=(right_min_x, right_min_value), xytext=(0, -16), textcoords='offset points',
                        ha='center', fontsize=12, color='g')
            # 标记交点
            ax.plot(right_edge_x, right_edge_y, 'go')
            ax.annotate("$G_{edge}^{'}$", xy=(right_edge_x, right_edge_y), xytext=(24, -4), textcoords='offset points',
                        ha='center', fontsize=12, color='g')

            # 设定x轴范围
            x_min = left_edge_x - 3 * left_width
            x_max = right_edge_x + 3 * right_width
            if x_min < 0:
                x_min = 0
            if x_max >= limit:
                x_max = limit - 1
            ax.set_xlim(x_min, x_max)

            # 设定标记竖线最高高度
            y_axis_min, y_axis_max = ax.get_ylim()
            y_axis_min -= 4
            y_axis_max += 8
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

            details = {
                "position": position,
                "left_edge_x": left_edge_x,
                "right_edge_x": right_edge_x,
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
    ax.set_title('Gray Scale Analysis')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Gray Scale Value')

    return distance, fig, details


class Sample:
    def __init__(self, image_path):
        self.raw_image = cv2_imread(image_path)
        self.path = image_path
        self.ROI = None
        self.scene = None
        self.first_matching_accuracy = ""
        self.matching_accuracy = ""
        self.angle = ""
        self.segmentation = []

        self.width = ''
        self.height = ''
        self.left_edge_array = []
        self.right_edge_array = []
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
            return
        # 模板匹配
        result = cv2.matchTemplate(self.raw_image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # 获取模板的宽度和高度
        h, w = template.shape[:2]

        # 计算匹配准确度并赋值给属性
        matching_accuracy = result[max_loc[1], max_loc[0]]
        self.first_matching_accuracy = f"{matching_accuracy * 100:.2f}%"

        # 裁剪图像
        cropped_image = self.raw_image[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
        self.raw_image = np.ascontiguousarray(cropped_image)

    def rotate_by_template(self, template=None):
        if template is None:
            return

        self.raw_image = cv2_imread(self.path)

        # 使用SIFT算法检测关键点和描述子
        sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
        kp1, des1 = sift.detectAndCompute(self.raw_image, None)
        kp2, des2 = sift.detectAndCompute(template, None)
        # 使用FLANN匹配器匹配关键点描述子
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # 获取匹配的良好关键点对
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

        # 提取匹配的关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算旋转变换矩阵
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            # 无法估计旋转变换矩阵
            M = np.eye(2, 3, dtype=np.float32)

        R = M[0:2, 0:2]
        angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        self.angle = "{:2.2f}°".format(angle)
        det_R = np.linalg.det(R)
        R = R / (det_R ** 0.5)
        M[0:2, 0:2] = R

        # 将原始图像进行旋转变换
        rotated_image = cv2.warpAffine(self.raw_image, M, (self.raw_image.shape[1], self.raw_image.shape[0]),
                                       flags=cv2.INTER_CUBIC)
        rotated_image = rotated_image[0:self.raw_image.shape[0], 0:self.raw_image.shape[1]]
        self.raw_image = np.ascontiguousarray(rotated_image)

    def rotate_by_angle(self, angle):
        self.angle = "{:2.2f}°".format(angle)
        image = cv2_imread(self.path)
        # 以原点为中心旋转
        R = cv2.getRotationMatrix2D((0, 0), angle, 1)
        rotated_image = cv2.warpAffine(image, R, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        self.raw_image = np.ascontiguousarray(rotated_image)

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
        self.raw_image = np.ascontiguousarray(cropped_image)

    def crop_by_region(self, image, region):
        start_x, start_y, width, height = region
        cropped_image = image[start_y:start_y + height, start_x:start_x + width]
        return cropped_image

    def segment(self, min_area_threshold=250, border=40):
        cropped_image = self.raw_image
        height, width, _ = cropped_image.shape

        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # (5, 5) 是高斯核的大小，0 是标准差

        # 应用 bitwise_not
        bitwise_not_image = cv2.bitwise_not(gray_image)

        # 自适应均值阈值处理
        adaptive_threshold = cv2.adaptiveThreshold(bitwise_not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 17,
                                                   2)

        # 使用闭运算填充图像内部的小孔和断裂
        kernel = np.ones((25, 25), np.uint8)
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
            if area > min_area_threshold:
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
        if self.ROI is None:
            return self.raw_image

        details = self.details[1 - row_analysis]
        if details is None:
            return self.ROI

        image = self.ROI.copy()

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
        expanded_width = int(details["expand_width"] * scale)

        if row_analysis:
            cv2.line(resized_image, (0, position), (new_width, position), (0, 0, 255), 1)
            cv2.line(resized_image, (0, position + expanded_width), (new_width, position + expanded_width), (0, 0, 255), 1)

            x_min = left_edge_x - 3 * left_width
            x_max = right_edge_x + 3 * right_width + expanded_width
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
            cv2.line(resized_image, (position + expanded_width, 0), (position + expanded_width, new_height), (0, 0, 255), 1)

            y_min = left_edge_x - 3 * left_width
            y_max = right_edge_x + 3 * right_width + expanded_width

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

    def row_analysis(self, config, row_analysis):
        fig = None
        details = None
        length = "error"

        try:
            length, fig, details = one_dimensional_analysis(self.ROI, config, row_analysis)
        except Exception as e:
            print("RuntimeError:")
            print(traceback.format_exc())

        return length, fig, details

    def analysis(self, config, row_analysis):
        row_analysis_config = config[1-row_analysis].copy()
        expand_width = config[1-row_analysis]["expand_width"]
        length, fig, details = self.row_analysis(row_analysis_config, row_analysis)
        self.fig[1-row_analysis] = fig
        details["expand_width"] = expand_width
        self.details[1-row_analysis] = details

        if isinstance(length, str):
            return length

        left_edge_array = []
        right_edge_array = []
        for relative_position in range(expand_width):
            length, fig, details = self.row_analysis(row_analysis_config, row_analysis)
            if isinstance(length, str):
                left_edge = np.NAN
                right_edge = np.NAN
            else:
                left_edge = details["left_edge_x"]
                right_edge = details["right_edge_x"]
            left_edge_array.append(left_edge)
            right_edge_array.append(right_edge)

            row_analysis_config["position"] += 1
            row_analysis_config["left_edge"] = round(left_edge)
            row_analysis_config["right_edge"] = round(right_edge)

        self.left_edge_array = np.array(left_edge_array)
        self.right_edge_array = np.array(right_edge_array)
        length_array = self.right_edge_array - self.left_edge_array
        length_mean = np.mean(length_array)
        if row_analysis:
            self.width = length_mean
        else:
            self.height = length_mean
        return length_mean

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

    def crop_ROI(self, template, region):
        start_x, start_y, width, height = region
        start_x = max(0, start_x - 20)
        start_y = max(0, start_y - 20)
        width = min(width + 40, self.raw_image.shape[1])
        height = min(height + 40, self.raw_image.shape[0])

        cropped_image = self.raw_image[start_y:start_y + height, start_x:start_x + width]

        # 模板匹配
        result = cv2.matchTemplate(cropped_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 计算匹配精度
        self.matching_accuracy = f"{max_val*100:.2f} %"

        # 裁剪图像
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        self.ROI = cropped_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

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
        raw_image = cv2_imread(self.path)
        cropped_image = self.crop_by_region(raw_image, region)
        path = os.path.join(dirname, f"ROI_{self.name}.png")
        cv2_imwrite(path, cropped_image)


if __name__ == "__main__":
    path = r"C:\Users\fpc\Desktop\python\strain-meter\sample\figure0512fpc01\0512fpc01--Video Image--056 01.png"
    sample = Sample(path)
    sample.crop_by_self()
    segmentations = sample.segment()
    segmented_images = [segmentation["image"] for segmentation in segmentations]

    for i, segmented_image in enumerate(segmented_images):
        cv2.imshow(f"segmented_image_{i}", segmented_image)
        cv2.waitKey(0)  # 等待按键输入
        cv2.destroyWindow(f"segmented_image_{i}")  # 关闭窗口

    cv2.destroyAllWindows()
