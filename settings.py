"""
Yolo V2模型全局变量
"""
# 数据集目录
DATASET_DIR = 'YOLO_V2_TrainTest'
# 定义每个网格的Anchor Boxes Num
ANCHORS_BOXES_NUM = 5
# 定义feature map的w
FEATURE_MAP_W = 13
# 定义Feature map的h
FEATURE_MAP_H = 13
# 划分训练集和测试集的比例
SPLIT_RATIO = 0.7
# 模型名
MODEL_NAME = 'YOLO_V2'
# 是否开启调试
DEBUG_OPEN = True
# Anchors存储的路径
ANCHORS_DIR = 'YOLO_V2_Anchors'
# VOC数据集的类别
VOC_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
               'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
               'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
