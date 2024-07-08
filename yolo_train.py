from ultralytics import YOLO #导入Yolo模块

# 加载yolov8的预训练模型，这个模型是yolov8使用了coco数据集训练的通用目标检测模型
# 我们将它作为基础模型，在该模型的基础上，训练安全帽模型

model=YOLO('yolov8n.pt') #用于加载模型

#训练用户自定义的数据集，数据的配置保存在safehat.yaml中，epochs等于100表示100轮迭代

model.train(data='safehat.yaml',epochs=100)

#使用验证集验证效果
model.val()