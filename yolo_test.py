import time

from ultralytics import YOLO


def test_model():
    # 加载yolov8的预训练模型，这个模型是yolov8使用了coco数据集训练的通用目标检测模型
    # 我们将它作为基础模型：

    model = YOLO('runs/detect/train15/weights/best.pt')  # 用于加载模型

    # 将模型移动到GPU
    # model.to('cuda')

    # 检查设备
    print(f"Model is using device: {model.device}")

    # 读取一些测试数据
    # model.predict(source="datasets/ai_data/source_files/source_files/JapanPPE.mp4", show=True, conf=0.5, save=True, classes=[0, 2,])

    # 读取摄像头输入
    model.predict(source=0, show=True, conf=0.5, save=True, classes=[0, 2])

if __name__ == '__main__':
    test_model()
    # time.sleep(1000)
