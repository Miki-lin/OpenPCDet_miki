import numpy as np
from pathlib import Path
import os


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    # 对R0_rect进行拓展，然后与Tr_velo_to_cam进行相乘求相反数后再求逆 R0_rect * Tr_velo_to_cam * y=x（y是雷达，x是照相机）
    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.top = np.array([])
        for i in range(0, 11):
            self.top = np.append(self.top, label[i])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.last = np.array([label[14]])


def get_calib(root_split_path, idx):
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return Calibration(calib_file)


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def get_label(root_split_path, idx):
    label_file = root_split_path / 'label_2' / ('%s.txt' % idx)
    assert label_file.exists()
    return get_objects_from_label(label_file)


def write_new_libel(root_split_path, idx, save_num):
    new_libel_file = root_split_path / 'new_label_2' / ('%s.txt' % idx)
    with open(new_libel_file, "a") as f:
        f.write(str(save_num[0]))
        for i in range(1, save_num.shape[0]):
            f.write(' ' + str(save_num[i]))
        f.write('\r\n')


# 去掉文件最后的换行符
def del_n(root_split_path, idx):
    new_libel_file = root_split_path / 'new_label_2' / ('%s.txt' % idx)
    file_object = open(new_libel_file, "rb+")
    file_object.seek(-2, 2)
    file_object.truncate()
    file_object.close()


def get_allfile(path):  # 获取所有文件
    all_file = []
    files = sorted(os.listdir(path))
    for f in files:  # listdir返回文件中所有目录
        # f_name = os.path.join(path, f)
        # f_name=os.path.basename(f_name)#去掉路径
        f = os.path.splitext(f)[0]  # 去掉文件名后缀
        all_file.append(f)
    return all_file


def clean_file(root_split_path, idx):
    new_libel_file = root_split_path / 'new_label_2' / ('%s.txt' % idx)
    file_object = open(new_libel_file, "w")
    file_object.close()


def mkdir_new_label_2(root_split_path):
    new_libel_2 = root_split_path / 'new_label_2'
    if os.path.exists(new_libel_2) is False:
        print("-------mkdir%s-------" % new_libel_2)
        os.mkdir(new_libel_2)


if __name__ == "__main__":
    root_split_path = Path('../../kitti_lidar/training')

    mkdir_new_label_2(root_split_path)
    all_file = get_allfile(root_split_path / 'label_2')  # tickets要获取文件夹名
    print("-------All name loaded-------")
    # print(all_file)

    for file_idx in all_file:
        clean_file(root_split_path, file_idx)
        print("This is the %s.txt" % file_idx)
        calib = get_calib(root_split_path, file_idx)
        obj_list = get_label(root_split_path, file_idx)
        annotations = {}
        for obj in obj_list:
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3)], axis=0)
            # print(annotations['location'])
            loc_lidar = calib.rect_to_lidar(annotations['location'])
            loc_lidar = loc_lidar.reshape(-1)
            # print("top",obj.top[0])
            temp = np.concatenate([obj.top, loc_lidar, obj.last], axis=0)
            # print("concatenate",temp)
            write_new_libel(root_split_path, file_idx, temp)
            # del_n(root_split_path, file_idx)

# 相机坐标系
#
# Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
# Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
# Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
# DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
# DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1000 -1000 -1000 -10
# DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1000 -1000 -1000 -10
# DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10
#
# 点云坐标系
#
# Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 69.7248 -0.4475647 -0.8413476 -1.56
# Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 58.780807 16.559633 -1.676111 1.57
# Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 46.12527 -4.572066 -0.9615387 -1.55
# DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1010.3567 989.2525 999.92944 -10
# DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1010.3567 989.2525 999.92944 -10
# DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1010.3567 989.2525 999.92944 -10
# DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1010.3567 989.2525 999.92944 -10

# 0.47 1.49 69.44
# -16.53 2.39 58.49
# 4.59 1.32 45.84

# 69.7248 -0.4475647 -0.8413476
# 58.780807 16.559633 -1.676111
# 46.12527 -4.572066 -0.9615387