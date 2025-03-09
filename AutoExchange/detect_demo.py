import cv2
import numpy as np


def imag_process(imag_path):

    # 1. 读取图像（BGR格式）
    img = cv2.imread(imag_path)
    if img is None:
        raise FileNotFoundError("图像路径错误！")

    # 2. 分离通道（B, G, R）
    b, g, r = cv2.split(img)

    # 3. 叠加红蓝通道（使用cv2.add避免溢出）
    combined = cv2.add(r, b)  # 或者：combined = r + b（但可能溢出）

    #高斯模糊

    blurred = cv2.GaussianBlur(combined, (5, 5), 0)

    # 4. 二值化处理（调整阈值以适应实际需求）
    threshold_value = 200  # 可根据需要调整阈值（如100、150等）
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # 形态学处理（膨胀腐蚀）
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

def draw_contours(imag_path, contours):

    # 1. 读取图像（BGR格式）
    img = cv2.imread(imag_path)
    if img is None:
        raise FileNotFoundError("图像路径错误！")

    # 在原图上绘制轮廓
    output_image = img.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)  # 绿色轮廓，线宽为1

    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Contours on Image', output_image)

    # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_arrow(contours):
    # 参数设置
    min_area = 500        # 最小面积阈值
    max_area = 1000       # 最大面积阈值
    min_aspect_ratio = 1.1  # 最小长宽比（细长轮廓）
    max_aspect_ratio = 2.5  # 最大长宽比（短胖轮廓）

    selected_contours = []
    for cnt in contours:
        # 1. 过滤面积
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # 2. 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)

        # 3. 检查长宽比
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # 4. 检查近似轮廓
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        print(len(approx))
        if len(approx) <3 or len(approx) > 6:
            continue

        selected_contours.append(cnt)
    return selected_contours

img_path = "AutoExchange\station_red.png"
contours = imag_process(img_path)
selected_contours = select_arrow(contours)
draw_contours(img_path, selected_contours)