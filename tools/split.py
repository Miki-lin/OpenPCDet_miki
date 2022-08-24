# !/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import os
import shutil
import random


def readFilename(path):  # 读取文件名函数（输入：文件夹路径，一个列表，标签信息）
    allfile = []
    filelist = os.listdir(path)  # 获取path下的所有文件名存入filelist中
    for filename in filelist:  # 遍历filelist
        filepath = os.path.join(path, filename)  # 将路径和文件名组合
        name = filepath.split("/")[-1].split(".")[0]
        file_cls = filepath.split("/")[-1].split(".")[-1]
        if file_cls == 'txt':
            allfile.append(name)  # 将filepath写入allfile中
    return allfile


def WriteDatasToFile(listInfo, txt_path):  # 传入参数为上边函数的list
    file_filename = open(txt_path, mode='w')  # 打开filename.txt
    for idx in range(len(listInfo)):  # 对于list中每一项
        filename = listInfo[idx]  # 使用str获取该项字符串
        print(filename)
        if idx < len(listInfo)-1:
            file_filename.write(filename + '\n')
        else:
            file_filename.write(filename)

    file_filename.close()


if __name__ == '__main__':
    path1 = "../../data/haizhu_354/training/label_2"  # 确定文件夹路径
    allfile1 = readFilename(path1)
    imagesets = '../../data/haizhu_354/ImageSets'
    if not os.path.exists(imagesets):
        os.mkdir(imagesets)
    txt_train = os.path.join(imagesets, 'train.txt')
    txt_val = os.path.join(imagesets, 'val.txt')
    txt_test = os.path.join(imagesets, 'test.txt')

    random.shuffle(allfile1)
    WriteDatasToFile(allfile1, txt_train)
    random.shuffle(allfile1)
    WriteDatasToFile(allfile1, txt_val)
    random.shuffle(allfile1)
    WriteDatasToFile(allfile1, txt_test)
