import time
from ultralytics import YOLO
import mss
import cv2
import numpy as np

def capture_half_screen():
    # 创建一个MSS对象
    sct = mss.mss()

    # 获取所有屏幕的信息
    monitors = sct.monitors

    # 选择左边屏幕（通常是第一个）
    left_screen = monitors[1]  # 注意：monitors[0] 是全屏快照，monitors[1] 是第一个实际屏幕

    # 设置捕获区域为左边屏幕的一半
    monitor_half = {
        'left': left_screen['left'],
        'top': left_screen['top'],
        'width': left_screen['width'] // 2,
        'height': left_screen['height']
    }

    return monitor_half

def test_model():
    # 加载yolov8的预训练模型，这个模型是yolov8使用了coco数据集训练的通用目标检测模型
    # 我们将它作为基础模型：
    model = YOLO('runs/detect/train15/weights/best.pt')  # 用于加载模型

    # 将模型移动到GPU（如果需要）
    model.to('cuda')  # 取消注释以使用GPU

    # 检查设备
    print(f"Model is using device: {model.device}")

    # 获取左边屏幕一半的捕获区域
    monitor = capture_half_screen()

    while True:
        # 捕获左边屏幕一半的画面
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        # 将图像从BGRA转换为BGR
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 使用模型进行预测
        results = model.predict(img, show=True, conf=0.5, classes=[0, 2])

        # 显示捕获的画面（可选）
        # cv2.imshow('Left Half Screen Capture', img)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放所有窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sct = mss.mss()  # 创建一个全局的MSS对象
    test_model()
    # time.sleep(1000)
