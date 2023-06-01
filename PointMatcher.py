import cv2
import numpy as np


def read_array(path):
    """
    将COLMAP产生的bin格式的深度图读入
    :param path: depth image path
    :return: depth image[numpy array]
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def read_depth(path, min_depth_percentile, max_depth_percentile):
    """
    将COLMAP产生的bin格式的深度图读入，并且过滤极值
    :param path: depth iamge path
    :param min_depth_percentile: minimum depth percentile
    :param max_depth_percentile: maximum depth percentile
    :return: depth image[numpy array]
    """
    if min_depth_percentile > max_depth_percentile:
        raise ValueError("min_depth_percentile should be less than or equal "
                         "to the max_depth_perceintile.")

    depth_map = read_array(path)

    min_depth, max_depth = np.percentile(depth_map, [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    return depth_map


def image_to_pointcloud(rgb_image, depth_image, K, pose, scale=1.0):
    """
    根据COLMAP数据库中的image、depth、camera K、pose获取该image对应的点云\n
    NOTICE: 这里过滤掉了无效点云，无法计算图像像素与点云之间的索引！
    :param rgb_image: rgb image
    :param depth_image: depth image
    :param K: camera K
    :param pose: camera pose
    :param scale: scale, default=1.0
    :return: pointcloud[list]
    """
    u = range(0, rgb_image.shape[1])
    v = range(0, rgb_image.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth_image.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    # 点应当在相机前方！
    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb_image[:, :, 0])[valid]
    G = np.ravel(rgb_image[:, :, 1])[valid]
    B = np.ravel(rgb_image[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points


def pixel_to_point(rgb, depth, u, v, K, pose, scale=1.0):
    """
    根据像素点坐标计算该像素点在世界坐标系中的对应地图点\n
    NOTICE: rgb[row][col], depth[row][col], col, row
    :param rgb: rgb image
    :param depth: depth image
    :param u: col
    :param v: row
    :param K: camera K
    :param pose: camera pose
    :param scale: scale, default=1.0
    :return: 3d color point(x,y,z,r,g,b)[list]
    """
    Z = depth * 1.0 / scale
    X = (u * 1.0 - K[0, 2]) * Z / K[0, 0]
    Y = (v * 1.0 - K[1, 2]) * Z / K[1, 1]

    if Z <= 0:
        return None

    position = np.array([X, Y, Z, 1])
    position = np.dot(pose, position)

    R = rgb[0]
    G = rgb[1]
    B = rgb[2]
    color = np.array([R, G, B])

    point = np.hstack((position[0:3], color)).tolist()

    return point


def match_feather_point(img1, img2):
    """
    计算特征点并进行匹配
    :param img1: first image
    :param img2: second image
    :return: keypoint of img1, keypoint of img2, good match from img1 to img2
    """
    # ================ 计算SIFT特征 ================
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # ================ FLANN特征匹配 ================
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # ================ 提取好的匹配点对 ================
    # TODO 此处使用描述子距离得到的匹配对较少，如何尽可能多保留正确匹配对？
    #  使用RANSAC方法？
    #  使用ORBSLAM2的旋转直方图？
    good_match = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:  # 如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
            good_match.append(m)
    return kp1, kp2, good_match


def get_match_pair(kp1, kp2, good_match):
    """
    get match pair from img1 to img2
    :param kp1: keypoint of img1
    :param kp2: keypoint of img2
    :param good_match: good match from im1 to img2
    :return: matched point coordinate[[x1,y1],[x2,y2]][list]
    """
    match_pair = []
    for m in good_match:
        x1 = int(kp1[m.queryIdx].pt[0])
        y1 = int(kp1[m.queryIdx].pt[1])
        x2 = int(kp2[m.trainIdx].pt[0])
        y2 = int(kp2[m.trainIdx].pt[1])
        match_pair.append([[x1, y1], [x2, y2]])
    return match_pair


# TODO 还没有实现的函数
def get_match_points(match_pair, rgb1, rgb2, depth1, depth2, pose1, pose2, K, scale=1.0):
    """
    获得一一匹配的点云对，用于后面的sim3求解
    :param match_pair: matched point coordinate
    :param rgb1: img1 rgb
    :param rgb2: img2 rgb
    :param depth1: img1 depth
    :param depth2: img2 depth
    :param pose1: img1 pose
    :param pose2: img2 pose
    :param K: camera K
    :param scale: scale, default=1.0
    :return: points1, points2
    """
    points1 = []
    points2 = []
    for mp in match_pair:
        point1 = pixel_to_point(img1[mp[0][1]][mp[0][0]],
                                img1_depth[mp[0][1]][mp[0][0]],
                                mp[0][0], mp[0][1],
                                cameraK, img1_pose)
        point2 = pixel_to_point(img2[mp[1][1]][mp[1][0]],
                                img2_depth[mp[1][1]][mp[1][0]],
                                mp[1][0], mp[1][1],
                                cameraK, img2_pose)

        if point1 is None or point2 is None:
            continue

        points1.append(point1)
        points2.append(point2)
    return points1, points2


if __name__ == "__main__":
    from DataLoader import DataLoader

    loader = DataLoader("data/param.yaml")

    images1, images2 = loader.load_images()
    depth1, depth2 = loader.load_depth()
    pose1, pose2 = loader.load_pose()
    cameraK = loader.load_cameraK()

    img1 = images1[0]
    img2 = images2[0]
    img1_depth = depth1[0]
    img2_depth = depth2[0]
    img1_pose = pose1[0]
    img2_pose = pose2[0]

    kp1, kp2, good_match = match_feather_point(img1, img2)
    match_pair = get_match_pair(kp1, kp2, good_match)
    points1, points2 = get_match_points(match_pair, img1, img2, img1_depth, img1_depth, img1_pose, img2_pose, cameraK)
    print(len(points1))
    print(len(points2))
