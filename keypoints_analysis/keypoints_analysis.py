import matplotlib.pyplot as plt
import fpc.pyplot_template
import cv2
import numpy as np


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def cv2_imwrite(file_path, img):
    cv2.imencode('.png', img)[1].tofile(file_path)


def keypoints_analysis(image1, image2):
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 提取良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配点的坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 绘制特征点
    image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 计算仿射变换矩阵
    transformation_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(transformation_matrix)
    dx1 = transformation_matrix[0, 0]
    dx2 = transformation_matrix[1, 0]
    length = np.sqrt(dx1 ** 2 + dx2 ** 2)
    print(length)

    # 对图像进行配准
    registered_image = cv2.warpPerspective(image1, transformation_matrix, (image2.shape[1], image2.shape[0]))

    # 显示图像和特征点
    plt.figure()
    plt.imshow(image1_with_keypoints)
    plt.figure()
    plt.imshow(image2_with_keypoints)
    plt.figure()