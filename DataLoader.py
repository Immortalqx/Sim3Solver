import yaml
import cv2
import numpy as np

from PointMatcher import read_depth


class DataLoader:
    def __init__(self, param_path):
        """
        从yaml中加载相关配置，但是不处理
        :param param_path: yaml path
        """
        yaml_file = open(param_path, "r")
        yaml_data = yaml.safe_load(yaml_file)

        self.cameraK = [
            [yaml_data["Camera.fx"], 0, yaml_data["Camera.cx"]],
            [0, yaml_data["Camera.fy"], yaml_data["Camera.cy"]],
            [0, 0, 1]
        ]

        self.images1_path = yaml_data["PointCloudA"]["image_path"]
        self.images1_depth_path = yaml_data["PointCloudA"]["image_depth"]
        self.images1_pose = yaml_data["PointCloudA"]["image_pose"]

        self.images2_path = yaml_data["PointCloudB"]["image_path"]
        self.images2_depth_path = yaml_data["PointCloudB"]["image_depth"]
        self.images2_pose = yaml_data["PointCloudB"]["image_pose"]

        self.min_depth_percentile = yaml_data["min_depth_percentile"]
        self.max_depth_percentile = yaml_data["max_depth_percentile"]

    def load_cameraK(self):
        """
        加载相机内参为numpy array格式
        :return: cameraK
        """
        return np.array(self.cameraK)

    def load_images(self):
        """
        加载图像
        :return: images1, images2
        """
        images1 = []
        images2 = []
        for m in self.images1_path:
            images1.append(cv2.imread(m))
        for n in self.images2_path:
            images2.append(cv2.imread(n))
        return images1, images2

    def load_depth(self):
        """
        加载深度图
        :return: depth1,depth2
        """
        depth1 = []
        depth2 = []
        for m in self.images1_depth_path:
            depth1.append(read_depth(m, self.min_depth_percentile, self.max_depth_percentile))
        for n in self.images2_depth_path:
            depth2.append(read_depth(n, self.min_depth_percentile, self.max_depth_percentile))
        return depth1, depth2

    def load_pose(self):
        """
        加载位姿为numpy array
        :return: pose1, pose2
        """
        return np.array(self.images1_pose).reshape(-1, 4, 4), np.array(self.images2_pose).reshape(-1, 4, 4)


if __name__ == "__main__":
    loader = DataLoader("data/param.yaml")
    print(loader.cameraK)
    print(loader.images1_path)
    print(loader.images1_depth_path)
    print(loader.images1_pose)
    print(loader.images2_path)
    print(loader.images2_depth_path)
    print(loader.images2_pose)
    loader.load_cameraK()
    loader.load_images()
    loader.load_depth()
    loader.load_pose()
