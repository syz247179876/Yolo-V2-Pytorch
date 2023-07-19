"""
工具
"""
import typing as t


def normalization(pic_w, pic_h, *bbox_coordinate) -> t.Tuple[float, float, float, float]:
    """
    对pic_w, pic_h归一化, 将bbox左上角，右下角点坐标转为bbox中心点坐标，并归一化
    """
    n_w = 1.0 / pic_w
    n_h = 1.0 / pic_h
    mid_x = (bbox_coordinate[0] + bbox_coordinate[2]) / 2.0
    mid_y = (bbox_coordinate[1] + bbox_coordinate[3]) / 2.0
    bbox_w = bbox_coordinate[2] - bbox_coordinate[0]
    bbox_y = bbox_coordinate[3] - bbox_coordinate[1]
    return mid_x * n_w, mid_y * n_h, bbox_w * n_w, bbox_y * n_h
