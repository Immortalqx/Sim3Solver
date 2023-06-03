import numpy as np
import math
import cv2
import random


def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = True):
    """
    Computes the least squares solution parameters of an Sim(m) matrix that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: Tw - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise ValueError("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("Degenerate covariance rank, "
                         "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    # to sim3 matrix
    Tw = np.zeros((4, 4))
    Tw[0:3, 0:3] = r * c
    Tw[0:3, 3] = t
    Tw[3, 3] = 1

    return Tw


def compute_centroid(points):
    """
    输入三个点，求质心和去质心之后的坐标
    :param points: 输入的3D点
    :return: 质心、去质心之后的坐标
    """
    # 每一列(x,y,z)分别求和，计算均值centroid
    centroid = np.sum(points, axis=1) / points.shape[1]
    # 减去质心
    cen_points = points - centroid

    return centroid, cen_points


# 感谢opencv，我不用这个代码了
# def rotate_mat(axis, radian):
#     """
#     轴角转化为旋转矩阵
#     :param axis: 旋转轴
#     :param radian: 旋转角
#     :return: 旋转矩阵
#     """
#     rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
#
#     return rot_matrix


# Sim3计算过程参考论文:
# Horn 1987, Closed-form solution of absolute orientataion using unit quaternions
def compute_sim3(points1: np.ndarray, points2: np.ndarray, FixScale: bool = True):
    """
    根据两组匹配的3D点，计算points2到points1的sim3变换
    :param points1: 匹配的3D点（三个点构成的numpy array）
    :param points2: 匹配的3D点（同上）
    :return: T12i,T21i
    """

    # Step 1: 计算两组点的质心，和去质心坐标
    o1, points1 = compute_centroid(points1)
    o2, points2 = compute_centroid(points2)

    # Step 2: 计算论文中三维点数目n>3的 M 矩阵。这里只使用了3个点
    M = points2.dot(points1.T)

    # Step 3: 计算论文中的 N 矩阵
    N11 = M[0, 0] + M[1, 1] + M[2, 2]  # Sxx + Syy + Szz
    N12 = M[1, 2] - M[2, 1]  # Syz - Szy
    N13 = M[2, 0] - M[0, 2]  # Szx - Sxz
    N14 = M[0, 1] - M[1, 0]  # ...
    N22 = M[0, 0] - M[1, 1] - M[2, 2]
    N23 = M[0, 1] + M[1, 0]
    N24 = M[2, 0] + M[0, 2]
    N33 = -M[0, 0] + M[1, 1] - M[2, 2]
    N34 = M[1, 2] + M[2, 1]
    N44 = -M[0, 0] - M[1, 1] + M[2, 2]

    N = np.array([
        [N11, N12, N13, N14],
        [N12, N22, N23, N24],
        [N13, N23, N33, N34],
        [N14, N24, N34, N44]
    ])

    # Step 4: 特征值分解求最大特征值对应的特征向量，就是我们要求的旋转四元数
    e_val, e_vec = np.linalg.eig(N)
    # 从大到小排序
    idx = e_val.argsort()[::-1]
    e_val = e_val[idx]
    e_vec = e_vec[idx]
    # N 矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数（q0 q1 q2 q3），其中q0 是实部
    # 将(q1 q2 q3)放入vec（四元数的虚部）
    vec = e_vec[0][1:4]
    # 四元数虚部模长 norm(vec)=sin(theta/2), 四元数实部 evec.at<float>(0,0)=q0=cos(theta/2)
    # 这一步的ang实际是theta/2，theta 是旋转向量中旋转角度
    ang = math.atan2(np.linalg.norm(vec), e_vec[0][0])
    # vec/norm(vec)归一化得到归一化后的旋转向量,然后乘上角度得到包含了旋转轴和旋转角信息的旋转向量vec
    vec = 2 * ang * vec / np.linalg.norm(vec)
    # 旋转向量（轴角）转换为旋转矩阵
    vec = 2 * ang * vec / np.linalg.norm(vec)
    R12i = cv2.Rodrigues(vec)[0]  # 疑问：opencv这个函数算出来的另外一半是什么东西？

    # Step 5: Rotate set 2
    # 利用刚计算出来的旋转将三维点旋转到同一个坐标系，P3对应论文里的 r_l,i', Pr1 对应论文里的r_r,i'
    points3 = R12i.dot(points2)

    # Step 6: 计算尺度因子 Scale
    s12i = 1.0
    if FixScale:
        # 论文中有2个求尺度方法。一个是p632右中的位置，考虑了尺度的对称性
        # 代码里实际使用的是另一种方法，这个公式对应着论文中p632左中位置的那个
        # Pr1 对应论文里的r_r,i',P3对应论文里的 r_l,i',(经过坐标系转换的Pr2), n=3, 剩下的就和论文中都一样了
        s12i = np.sum(points1 * points3) / np.sum(cv2.pow(points3, 2))

    # Step 7: 计算平移Translation
    t12i = o1 - s12i * R12i.dot(o2)

    # Step 8: 计算双向变换矩阵，目的是在后面的检查的过程中能够进行双向的投影操作
    # Step 8.1 用尺度，旋转，平移构建变换矩阵 T12
    T12i = np.zeros((4, 4))
    T12i[0:3, 0:3] = s12i * R12i
    T12i[0:3, 3] = t12i
    T12i[3, 3] = 1
    # Step 8.2 T21
    T21i = np.zeros((4, 4))
    T21i[0:3, 0:3] = (1.0 / s12i) * R12i.T
    T21i[0:3, 3] = (1.0 / s12i) * R12i.T.dot(t12i)
    T21i[3, 3] = 1

    return T12i, T21i


# 为了RANSAC，这里需要计算一个误差。
# 考虑到没有尺度的点云，他的距离不是很可控，所以还是选择重投影误差，我认为像素点的范围是可控的
def reproject_error(point3d, point2d, Tcw, K):
    """
    按照给定的Sim3变换进行投影操作,得到3D点的2D投影点,并计算重投影误差\n
    NOTICE: 这里计算的是单个点的像素误差！！！
    :param point3d: 3D点
    :param point2d: 2D点
    :param Tcw: sim3变换矩阵
    :param K: 内参
    :return: 像素差的平方和
    """
    Rcw = Tcw[0:3, 0:3]
    tcw = Tcw[0:3, 3]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 首先将对方关键帧的地图点坐标转换到这个关键帧的相机坐标系下
    point3d = Rcw.dot(point3d) + tcw
    # 投影
    inv_z = 1.0 / point3d[2]
    x = point3d[0] * inv_z
    y = point3d[1] * inv_z
    repoject_point2d = np.array([fx * x + cx, fy * y + cy])

    # print(point2d, repoject_point2d)

    return np.sum((point2d - repoject_point2d) ** 2)


# 重投影误差有问题，可能是COLMAP拿到深度图的时候做了什么处理。这里我尝试用一下点与点之间的记录计算一下
# TODO 这里需要传入变换矩阵，然后对某一个点进行一个处理，变换到同一坐标系下！
def geometric_error(point3d1, point3d2, T21i):
    """
    计算3D点之间的距离
    :param point3d1: 第一个3D点
    :param point3d2: 第二个3D点
    :param T21i: 第二个3D点坐标系到第一个3D点坐标系的sim3矩阵
    :return: 距离的平方
    """
    # R*P+t
    point3d2 = T21i[0:3, 0:3] * point3d2 + T21i[0:3, 3]
    print(np.sum((point3d1 - point3d2) ** 2))
    return np.sum((point3d1 - point3d2) ** 2)


# RANSAC算法模板：https://github.com/Immortalqx/RANSAC/blob/master/RANSAC_3D.py
def RANSAC(point3ds1, point3ds2, point2ds1, point2ds2, K):
    """
    使用RANSAC算法估算模型
    """
    # 数据规模
    SIZE = point3ds1.shape[0]
    # 迭代最大次数，每次得到更好的估计会优化iters的数值，默认10000
    iters = 10000
    # 数据和模型之间可接受的差值，默认50(不超过半径为7的圆)
    sigma = 20
    # 内点数目
    pretotal = 0
    # 希望的得到正确模型的概率，默认0.99
    Per = 0.999
    # 初始化一下
    T12i = None
    T21i = None
    # Tcw = None
    for i in range(iters):
        # 随机在数据中选出三个点去求解模型
        sample_index = random.sample(range(SIZE), 3)
        # sample_index = random.sample(range(SIZE), 10)
        T12i, T21i = compute_sim3(point3ds1[sample_index], point3ds2[sample_index])
        # Tcw = umeyama_alignment(point3ds1[sample_index].T, point3ds2[sample_index].T)

        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            if geometric_error(point3ds1[index], point3ds2[index], T12i) < sigma:
                # if reproject_error(point3ds2[index], point2ds1[index], T12i, K) < sigma and \
                #         reproject_error(point3ds1[index], point2ds2[index], T21i, K) < sigma:
                # if reproject_error(point3ds2[index], point2ds1[index], Tcw, K) < sigma:
                total_inlier = total_inlier + 1
        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            iters = math.log(1 - Per) / math.log(1 - math.pow(total_inlier / SIZE, 2))
            pretotal = total_inlier

        # 判断是否当前模型已经符合超过90%的点
        if total_inlier > SIZE * 0.9:
            break
    print("内点数目:\t", pretotal)
    return T12i, T21i
    # return Tcw


if __name__ == "__main__":
    # test_points1 = np.array([
    #     [1, 2, 3],
    #     [2, 3, 4],
    #     [3, 4, 10]
    # ])
    # test_points2 = np.array([
    #     [9, 8, 7],
    #     [8, 7, 6],
    #     [6, 5, 0]
    # ])

    # compute_sim3(test_points1, test_points2, True)
    # print(geometric_error(test_points1[1], test_points2[1]))

    # points3d = np.array([
    #     [100, 200, 300],
    #     [150, 250, 300]
    # ])
    # points2d = np.array([
    #     [970, 980],
    #     [1140, 1150]
    # ])
    # Tcw = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    # K = np.array([
    #     [1000, 0, 640],
    #     [0, 1000, 320],
    #     [0, 0, 1]
    # ])
    #
    # print(reproject_error(points3d[0], points2d[0], Tcw, K))

    sample_indx = random.sample(range(10), 3)
    test_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(test_array)
    print(sample_indx)
    print(test_array[sample_indx])
