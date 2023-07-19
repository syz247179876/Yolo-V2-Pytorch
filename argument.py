"""
命令行参数设置
"""
import argparse
import torch.cuda


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opts = None

    def set_process_args(self):
        """
        设置数据预处理时候的参数
        """
        self.parser.add_argument('--random_seed', type=int, default=333, help='use to randomly initialize anchor boxes')
        self.parser.add_argument('--anchors_num', type=int, default=5, help='the number of anchor boxes')
        self.parser.add_argument('--max_iter', type=int, default=300,
                                 help='the max iteration to find best anchor boxes')
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 default=r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012',
                                 help='the base dir of dataset'
                                 )

        self.opts = self.parser.parse_args()

    def set_train_args(self):
        """
        设置训练时的参数
        """
        pass

    def set_test_args(self):
        """
        设置测试时的参数
        """
        pass
