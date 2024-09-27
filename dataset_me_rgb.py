import os
import scipy.io as scio
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
#from sklearn import preprocessing
def get_contour(bin_img):
    # get contour
    contour_img = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (bin_img[i][j] == 0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum == 0:
                    contour_img[i][j] = 255

    return contour_img

def keypointtrans(data):
    res=np.zeros([24,3])
    res[0,:]=data[0,:]
    res[1, :] = data[18, :]
    res[2, :] = data[22, :]
    res[3, :] = data[1, :]
    res[4, :] = data[19, :]
    res[5, :] = data[23, :]
    res[6, :] = data[2, :]
    res[7, :] = data[20, :]
    res[8, :] = data[24, :]
    res[9, :] = data[2, :]
    res[10, :] = data[21, :]
    res[11, :] = data[25, :]
    res[12, :] = data[3, :]
    res[13, :] = data[4, :]
    res[14, :] = data[11, :]
    res[15, :] = data[26, :]
    res[16, :] = data[5, :]
    res[17, :] = data[12, :]
    res[18, :] = data[6, :]
    res[19, :] = data[13, :]
    res[20, :] = data[7, :]
    res[21, :] = data[14, :]
    res[22, :] = data[9, :]
    res[23, :] = data[16, :]
    return res

def keypointtrans_kp2d(data):
    res=np.zeros([24,2])

    res[1, :] = data[4, :]
    res[2, :] = data[3, :]
    res[4, :] = data[5, :]
    res[5, :] = data[2, :]
    res[7, :] = data[6, :]
    res[8, :] = data[1, :]
    res[10, :] = data[7, :]
    res[11, :] = data[0, :]
    res[12, :] = data[14, :]
    res[15, :] = data[16, :]
    res[16, :] = data[11, :]
    res[17, :] = data[10, :]
    res[18, :] = data[18, :]
    res[19, :] = data[9, :]
    res[20, :] = data[13, :]
    res[21, :] = data[8, :]
    #print("res[1:3, :]",res[1:3, :].shape)
    res_all = np.concatenate((res[1:3, :], res[4:6, :],res[7:9 :],res[10:13, :],res[15:22, :]), axis=0)

    return res_all

#数据增强-rgb
import torchvision.transforms as T


def transform(IsResize, Resize_size, IsTotensor, IsNormalize, Norm_mean, Norm_std, IsRandomGrayscale, IsColorJitter,
              brightness, contrast, hue, saturation, IsCentercrop, Centercrop_size, IsRandomCrop, RandomCrop_size,
              IsRandomResizedCrop, RandomResizedCrop_size, Grayscale_rate, IsRandomHorizontalFlip, HorizontalFlip_rate,
              IsRandomVerticalFlip, VerticalFlip_rate, IsRandomRotation, degrees):
    Resize_transform = []
    Rotation = []
    Color = []
    Tensor = []
    Normalize = []

    # -----------------------------------------------<旋转图像>-----------------------------------------------------------#
    if IsRandomRotation:
        Rotation.append(transforms.RandomRotation(degrees))
    if IsRandomHorizontalFlip:
        Rotation.append(transforms.RandomHorizontalFlip(HorizontalFlip_rate))
    if IsRandomVerticalFlip:
        Rotation.append(transforms.RandomHorizontalFlip(VerticalFlip_rate))

    # -----------------------------------------------<图像颜色>-----------------------------------------------------------#
    if IsColorJitter:
        Color.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
    if IsRandomGrayscale:
        Color.append(transforms.RandomGrayscale(Grayscale_rate))

    # ---------------------------------------------<缩放或者裁剪>----------------------------------------------------------#
    if IsResize:
        Resize_transform.append(transforms.Resize(Resize_size))
    if IsCentercrop:
        Resize_transform.append(transforms.CenterCrop(Centercrop_size))
    if IsRandomCrop:
        Resize_transform.append(transforms.RandomCrop(RandomCrop_size))
    if IsRandomResizedCrop:
        Resize_transform.append(transforms.RandomResizedCrop(RandomResizedCrop_size))
    if (IsResize + IsCentercrop + IsRandomCrop + IsRandomResizedCrop) >= 2:
        print("警告：您同时使用了多种裁剪或缩放方法，请确认这是否是您所预期的")

    # ---------------------------------------------<tensor化和归一化>------------------------------------------------------#
    if IsTotensor:
        Tensor.append(transforms.ToTensor())
    if IsNormalize and len(Color) == 0:
        Normalize.append(transforms.Normalize(Norm_mean, Norm_std))
    else:
        Normalize.append(transforms.Normalize(Norm_mean, Norm_std))
        print("警告：亮度、对比度、转为灰度图等操作会改变图像的均值和方差可能会与归一化操作有冲突，因此不推荐一起使用，除非您清楚您在干什么")

    # 您可以更改数据增强的顺序，但是数据增强的顺序可能会影响最终数据的质量，因此除非您十分明白您在做什么,否则,请保持默认顺序
    transforms_order = [Resize_transform, Rotation, Color, Tensor, Normalize]

    return transforms.Compose(transforms_order)


def get_transform():
    return transform(
        IsResize=True,  # 是否缩放图像
        Resize_size=(256, 256),  # 缩放后的图像大小 如（512,512）->（256,192）
        IsCentercrop=False,  # 是否进行中心裁剪
        Centercrop_size=(256, 256),  # 中心裁剪后的图像大小
        IsRandomCrop=False,  # 是否进行随机裁剪
        RandomCrop_size=(256, 256),  # 随机裁剪后的图像大小
        IsRandomResizedCrop=False,  # 是否随机区域进行裁剪
        RandomResizedCrop_size=(256, 256),  # 随机裁剪后的图像大小
        IsTotensor=True,  # 是否将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
        IsNormalize=True,  # 是否对图像进行归一化操作,即使用图像的均值和方差将图像的数值范围从[0,1]->[-1,1]
        Norm_mean=(0.5, 0.5, 0.5),  # 图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
        Norm_std=(0.5, 0.5, 0.5),  # 图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
        IsRandomGrayscale=False,  # 是否随机将彩色图像转化为灰度图像
        Grayscale_rate=0.5,  # 每张图像变成灰度图像的概率，设置为1的话等同于transforms.Grayscale()
        IsColorJitter=False,  # 是否随机改变图像的亮度、对比度、色调和饱和度
        brightness=0.5,  # 每个图像被随机改变亮度的概率
        contrast=0.5,  # 每个图像被随机改变对比度的概率
        hue=0.5,  # 每个图像被随机改变色调的概率
        saturation=0.5,  # 每个图像被随机改变饱和度的概率
        IsRandomVerticalFlip=False,  # 是否垂直翻转图像
        VerticalFlip_rate=0.5,  # 每个图像被垂直翻转图像的概率
        IsRandomHorizontalFlip=False,  # 是否水平翻转图像
        HorizontalFlip_rate=0.5,  # 每个图像被水平翻转图像的概率
        IsRandomRotation=False,  # 是是随机旋转图像
        degrees=10,  # 每个图像被旋转角度的范围 如degrees=10 则图像将随机旋转一个(-10,10)之间的角度

    )


# 图像分类、识别推荐使用的数据增强配置，
"""def get_transform():
    return transform(
        IsResize=True, #可选
        Resize_size=(256,256), #缩放后的图像大小 如（512,512）->（256,192）
        IsCentercrop=False,#可选
        Centercrop_size=(256,256),#中心裁剪后的图像大小
        IsRandomCrop=False,#可选
        RandomCrop_size=(256,256),#随机裁剪后的图像大小
        IsRandomResizedCrop=False,#可选
        RandomResizedCrop_size=(256,256),#随机裁剪后的图像大小
        IsTotensor=True, #建议保持这样
        IsNormalize=True, #建议保持这样
        Norm_mean=(0.5,0.5,0.5),#图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
        Norm_std=(0.5,0.5,0.5),#图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
        IsRandomGrayscale=False,#建议保持这样
        Grayscale_rate=0.5,#建议保持这样
        IsColorJitter=False,#建议保持这样
        brightness=0.5,#建议保持这样
        contrast=0.5,#建议保持这样
        hue=0.5,#建议保持这样
        saturation=0.5,#建议保持这样
        IsRandomVerticalFlip=True,#建议保持这样
        VerticalFlip_rate=0.5,#建议保持这样
        IsRandomHorizontalFlip=True,#建议保持这样
        HorizontalFlip_rate=0.5,#建议保持这样
        IsRandomRotation=True,#建议保持这样
        degrees=10,#建议10-90之间

    )"""

# 图像生成模型，语义分割类模型，不建议对其进行额外的图像增强，如下即可（只用修改Resize_size和看情况修改Norm_mean和Norm_std）
"""def get_transform():
    return transform(
        IsResize=True, #是否缩放图像
        Resize_size=(256,256), #缩放后的图像大小 如（512,512）->（256,192）
        IsCentercrop=False,#是否进行中心裁剪
        Centercrop_size=(256,256),#中心裁剪后的图像大小
        IsRandomCrop=False,#是否进行随机裁剪
        RandomCrop_size=(256,256),#随机裁剪后的图像大小
        IsRandomResizedCrop=False,#是否随机区域进行裁剪
        RandomResizedCrop_size=(256,256),#随机裁剪后的图像大小
        IsTotensor=True, #是否将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
        IsNormalize=True, #是否对图像进行归一化操作,即使用图像的均值和方差将图像的数值范围从[0,1]->[-1,1]
        Norm_mean=(0.5,0.5,0.5),#图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
        Norm_std=(0.5,0.5,0.5),#图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
        IsRandomGrayscale=False,#是否随机将彩色图像转化为灰度图像
        Grayscale_rate=0.5,#每张图像变成灰度图像的概率，设置为1的话等同于transforms.Grayscale()
        IsColorJitter=False,#是否随机改变图像的亮度、对比度、色调和饱和度
        brightness=0.5,#每个图像被随机改变亮度的概率
        contrast=0.5,#每个图像被随机改变对比度的概率
        hue=0.5,#每个图像被随机改变色调的概率
        saturation=0.5,#每个图像被随机改变饱和度的概率
        IsRandomVerticalFlip=False,#是否垂直翻转图像
        VerticalFlip_rate=0.5,#每个图像被垂直翻转图像的概率
        IsRandomHorizontalFlip=False,#是否水平翻转图像
        HorizontalFlip_rate=0.5,#每个图像被水平翻转图像的概率
        IsRandomRotation=False,#是是随机旋转图像
        degrees=10,#每个图像被旋转角度的范围 如degrees=10 则图像将随机旋转一个(-10,10)之间的角度
    )"""

smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                          [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
    , [2, 5], [5, 8], [1, 4], [4, 7]]

def draw3Dpose(pose_3d,  ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")

    RADIUS = 0.8 # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def draw3Dpose_frames(input):
    # 绘制连贯的骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    i = 0
    j = 0
    while i < input.shape[0]:
        draw3Dpose(input[i], ax)
        plt.pause(10)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1
        if i == input.shape[0]:
            # i=0
            j += 1
        if j == 2:
            break

    plt.ioff()
    plt.show()

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    #print('voxel: ', voxel)
    locations = (points + radius) / voxel
    #print('locations: ', locations)
    locations = locations.astype(int)
    #print('locations: ', locations)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol

def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.grid(False)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def pyplot_draw_point_cloud2(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.grid(False)
    ax.scatter(points[1:10, 0], points[1:10, 1], points[1:10, 2], c=['red'])
    ax.scatter(points[11:, 0], points[11:, 1], points[11:, 2], c=['blue'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def showVoxel(voxel):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxel, edgecolor="k")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

class SiamesePC_20_nonormalization_rgb_crop(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            #print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            # target = np.random.choice(2)
            target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key,rgb = self.data_ti[index], self.data_label[index],self.data_key[index],self.data_rgb[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])

            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = 1
            ti1, labelti,kinect_key,rgb = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index],self.data_test_rgb[index]
            labelkinect = labelti
            siamese_label = labelti
            siamese_index = index
            kinect_6890=0
            # print(labelti)
            labelrgb = labelti
            indexti = labelti
            indexkinect = labelkinect
            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        list_all_rgb_full = []
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            list_all_rgb_full = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            list_label_rgb = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_rgb_full.npy",
                    list_label_rgb)
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb_CROP:", len(list_all_rgb_crop))
        print("length of rgb_FULL:", len(list_all_rgb_full))
        print("length of list_label_all:", len(list_label_all))
        print("length of list_label_rgb:", len(list_label_rgb))
        print(list_label_all)
        print(list_label_rgb)
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_targetrandom(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            #target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_targetrandom_gt2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2=self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2=self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            #target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key2 = self.data_key2[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            kinect_key2 = self.data_test_key2[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target,kinect_key2
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key_calibration.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_targetrandom_rgbgt(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            #target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti = self.data_ti[index], self.data_label[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key= self.data_key[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_target0(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}


    def __getitem__(self, index):
        if self.train:
            # target = np.random.choice(2)
            target = 0
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)

            siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            rgb = self.data_rgb[siamese_index]

            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = 0
            kinect_6890=0
            ti1, labelti,kinect_key,rgb = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index],self.data_test_rgb[index]
            siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            #print("label_rgb",siamese_label)
            #print("index_rgb", siamese_index)
            rgb = self.data_rgb[siamese_index]

            labelrgb = siamese_label
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_target0_rgbgt(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_key = np.asarray(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}


    def __getitem__(self, index):
        if self.train:
            # target = np.random.choice(2)
            target = 0
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)

            siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            rgb = self.data_rgb[siamese_index]

            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = 0
            kinect_6890=0
            ti1, labelti,rgb = self.data_test_ti[index], self.data_test_label[index], self.data_test_rgb[index]
            siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            #print("label_rgb",siamese_label)
            #print("index_rgb", siamese_index)
            rgb = self.data_rgb[siamese_index]
            kinect_key = self.data_test_key[siamese_index]
            labelrgb = siamese_label
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_crop_target0_gt2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2=self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2=self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = 0
            #target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key2 = self.data_key2[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = 0
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            kinect_key2 = self.data_test_key2[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target,kinect_key2
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_full_target0_gt2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2=self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2=self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = 0
            #target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key2 = self.data_key2[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = 0
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            kinect_key2 = self.data_test_key2[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target,kinect_key2
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("./npydata/SiamesePC/list_all_ti.npy")
            list_all_kinect = np.load("./npydata/SiamesePC/list_all_kinect.npy")
            list_index_all = np.load("./npydata/SiamesePC/list_index_all.npy")
            list_all_kinect_key = np.load("./npydata/SiamesePC/list_all_kinect_key.npy")
            list_all_kinect_6890 = np.load("./npydata/SiamesePC/list_all_kinect_6890.npy")
            list_label_all = np.load("./npydata/SiamesePC/list_label_all.npy")
            list_bit = np.load("./npydata/SiamesePC/list_bit.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_ti.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_all_kinect_key.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization\list_label_all.npy")
            list_all_rgb_crop = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_crop.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        list_all_ti = np.asarray(list_all_ti)
        list_all_kinect_key = np.asarray(list_all_kinect_key)
        list_all_kinect_6890 = np.asarray(list_all_kinect_6890)
        list_label_all = np.asarray(list_label_all)
        list_label_rgb = np.asarray(list_label_rgb)
        list_bit = np.asarray(list_bit)
        #数据归一化
        list_all_ti[:,:,:,0:1]=self.normalization(list_all_ti[:,:,:,0:1])
        list_all_ti[:,:,:,1:2]=self.normalization(list_all_ti[:,:,:,1:2])
        list_all_ti[:,:,:,2:3]=self.normalization(list_all_ti[:,:,:,2:3])
        np.save("./npydata/SiamesePC/list_all_ti.npy",list_all_ti)
        np.save("./npydata/SiamesePC/list_all_kinect.npy", list_all_kinect)
        np.save("./npydata/SiamesePC/list_label_all.npy", list_label_all)
        np.save("./npydata/SiamesePC/list_index_all.npy", list_index_all)
        np.save("./npydata/SiamesePC/list_bit.npy", list_bit)
        np.save("./npydata/SiamesePC/list_all_kinect_6890.npy", list_all_kinect_6890)
        np.save("./npydata/SiamesePC/list_all_kinect_key.npy", list_all_kinect_key)
        '''

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_full_targetrandom_gt2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_\
                , self.data_rgb_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2=self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key\
                , self.data_rgb = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2=self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            #target = np.random.choice(2)
            target = 1
            kinect_6890=0
            # 监督学习
            ti1, labelti, kinect_key = self.data_ti[index], self.data_label[index],self.data_key[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key2 = self.data_key2[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti1, labelti,kinect_key = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index]
            kinect_6890=0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            kinect_key2 = self.data_test_key2[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target,kinect_key2
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max=[]
        list_min=[]

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_calibration.npy")
            list_all_kinect = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect.npy")
            list_index_all = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all.npy")
            list_all_kinect_key = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_calibration.npy")
            list_label_all = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all.npy")
            list_all_rgb_full = np.load("/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full.npy")
        else:
            list_all_ti = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_calibration.npy")
            list_all_kinect = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect.npy")
            list_index_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all.npy")
            list_all_kinect_key = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_calibration.npy")
            list_label_all = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all.npy")
            list_all_rgb_full = np.load("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")

            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            def draw3Dpose(pose_3d,pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
                           add_labels=False):  # blue, orange
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.grid(False)
                for i in smpl_connectivity_dict:
                    x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x, y, z, lw=2, c=lcolor)
                    x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x2, y2, z2, lw=2, c=rcolor)


                RADIUS = 1  # space around the subject
                xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
                ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            def draw3Dpose_frames(ti,data_key_in):
                # 绘制连贯的骨架
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
                i = 0
                j = 0
                while i < ti.shape[0]:
                    draw3Dpose(ti[i],data_key_in[i], ax)

                    plt.pause(0.3)
                    # print(ax.lines)
                    plt.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.lines = []
                    i += 1
                    if i == ti.shape[0]:
                        # i=0
                        j += 1
                    if j == 2:
                        break

                plt.ioff()
                plt.show()

            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            #print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            #draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
            for i in range(380):
                '''
                
                list_all_kinect_key[i, :, :, 0] = list_all_kinect_key[i, :, :, 0] + (
                            0 - list_all_kinect_key[i, 0, 0, 0])
                list_all_kinect_key[i, :, :, 1] = list_all_kinect_key[i, :, :, 1] + (
                            4.5 - list_all_kinect_key[i, 0, 0, 1])
                list_all_kinect_key[i, :, :, 2] = list_all_kinect_key[i, :, :, 2] + (
                           0.5 - list_all_kinect_key[i, 0, 0, 2])
                '''
                list_all_ti[i, :, :, 0] = list_all_ti[i, :, :, 0] + (
                        0 - list_all_kinect_key[i, 0, 0, 0])
                list_all_ti[i, :, :, 1] = list_all_ti[i, :, :, 1] + (
                        4.5 - list_all_kinect_key[i, 0, 0, 1])
                list_all_ti[i, :, :, 2] = list_all_ti[i, :, :, 2] + (
                        0.5 - list_all_kinect_key[i, 0, 0, 2])
            print(list_all_ti[0, 0, 0, :])
            #np.save("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_calibration.npy",list_all_kinect_key)
            #np.save("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_calibration.npy",list_all_ti)

            #draw3Dpose_frames(list_all_kinect_key[20, :, :, :],list_all_kinect_key[10, :, :, :])
            #print(list_all_ti.shape)

            #for i in range(0,len(list_all_ti)):
                #for j in range(15,25):
                    #pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
                #pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])


        #max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))


        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key,list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_nonormalization_rgb_full_gt2_kp2d(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_, self.data_kp2d_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_kinect = []
            self.data_label = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_kp2d = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_kp2d.append(self.data_kp2d_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                        self.data_kp2d.append(self.data_kp2d_[idnum[ii]])
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_index = np.asarray(self.data_index)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)
            self.data_kp2d = np.asarray(self.data_kp2d)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb , self.data_kp2d = self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kinect)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_index)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_6890)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_rgb = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_kp2d = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        self.data_test_kp2d.append(self.data_kp2d[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        self.data_test_kp2d.append(self.data_kp2d[idnum[ii]])
            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            print(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_kp2d = np.asarray(self.data_test_kp2d)
            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            # target = np.random.choice(2)
            target = 1
            kinect_6890 = 0
            # 监督学习
            ti1, labelti, kinect_key ,kp2d = self.data_ti[index], self.data_label[index], self.data_key[index], self.data_kp2d[index]
            # print(index)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_rgb[siamese_index]
            kinect_key2 = self.data_key2[siamese_index]
            labelrgb = siamese_label

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti1, labelti, kinect_key ,kp2d = self.data_test_ti[index], self.data_test_label[index], self.data_test_key[index], self.data_test_kp2d[index]
            kinect_6890 = 0
            # print(labelti)
            if target == 0:
                siamese_label = np.random.choice(list(self.data_label_set - set([labelti])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            else:
                siamese_index = index
                siamese_label = labelti
                # while siamese_index==index:
                # siamese_index=np.random.choice(self.label_to_indices[labelti])
            rgb = self.data_test_rgb[siamese_index]
            kinect_key2 = self.data_test_key2[siamese_index]
            labelrgb = siamese_label

            # print(kinect1[0].shape)
        return ti1, labelti, rgb, labelrgb, kinect_6890, kinect_key, target, kinect_key2 , kp2d
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_calibration.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_calibration.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kp2d_crop.npy")
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_calibration.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_calibration.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            list_all_kp2d = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d.npy")
            list_all_kp2d_crop = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d_crop.npy")


            #选取kp2d
            '''
            list_all_kp2d_crop = []
            for i in range(380):
                list_all_kp2d_crop_25=[]
                for j in range(25):
                    #print("before:",list_all_kp2d[i][j].shape)
                    list_all_kp2d_crop_25.append(keypointtrans_kp2d(list_all_kp2d[i][j]))
                    #print("after:", list_all_kp2d[i][j].shape)
                list_all_kp2d_crop.append(list_all_kp2d_crop_25)
            list_all_kp2d_crop = np.asarray(list_all_kp2d_crop)
            print("list_all_kp2d_crop:",list_all_kp2d_crop.shape)
            np.save(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d_crop.npy",
                list_all_kp2d_crop)

            '''
            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            def draw3Dpose(pose_3d, pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
                           add_labels=False):  # blue, orange
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.grid(False)
                for i in smpl_connectivity_dict:
                    x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x, y, z, lw=2, c=lcolor)
                    x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x2, y2, z2, lw=2, c=rcolor)

                RADIUS = 1  # space around the subject
                xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
                ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            def draw3Dpose_frames(ti, data_key_in):
                # 绘制连贯的骨架
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
                i = 0
                j = 0
                while i < ti.shape[0]:
                    draw3Dpose(ti[i], data_key_in[i], ax)

                    plt.pause(0.3)
                    # print(ax.lines)
                    plt.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.lines = []
                    i += 1
                    if i == ti.shape[0]:
                        # i=0
                        j += 1
                    if j == 2:
                        break

                plt.ioff()
                plt.show()

            #draw3Dpose_frames(list_all_kinect_key[0,:,:,:],list_all_kinect_key[1,:,:,:])
            #draw3Dpose_frames(list_all_kinect_key[0, :, :, :], list_all_kinect_key[1, :, :, :])
            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
            '''
            for i in range(380):
                list_all_kinect_key[i, :, :, 0] = list_all_kinect_key[i, :, :, 0] + (
                            0 - list_all_kinect_key[i, 0, 0, 0])
                list_all_kinect_key[i, :, :, 1] = list_all_kinect_key[i, :, :, 1] + (
                            4.5 - list_all_kinect_key[i, 0, 0, 1])
                list_all_kinect_key[i, :, :, 2] = list_all_kinect_key[i, :, :, 2] + (
                           0.5 - list_all_kinect_key[i, 0, 0, 2])
                
                list_all_ti[i, :, :, 0] = list_all_ti[i, :, :, 0] + (
                        0 - list_all_kinect_key[i, 0, 0, 0])
                list_all_ti[i, :, :, 1] = list_all_ti[i, :, :, 1] + (
                        4.5 - list_all_kinect_key[i, 0, 0, 1])
                list_all_ti[i, :, :, 2] = list_all_ti[i, :, :, 2] + (
                        0.5 - list_all_kinect_key[i, 0, 0, 2])
            print(list_all_ti[0, 0, 0, :])
            '''
            # np.save("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_calibration.npy",list_all_kinect_key)
            # np.save("F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_calibration.npy",list_all_ti)

            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :],list_all_kinect_key[10, :, :, :])
            # print(list_all_ti.shape)

            # for i in range(0,len(list_all_ti)):
            # for j in range(15,25):
            # pyplot_draw_point_cloud3(list_all_ti[i][j],list_all_kinect_key[i][j])
            # pyplot_draw_point_cloud3(list_all_ti[i], list_all_kinect_key[i])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full,list_all_kp2d_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_rgb_full_singlemodule_offlinetri(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_, self.data_kp2d_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d_)


            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.6:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            self.label_to_indices1 = {label: np.where(np.asarray(self.data_label1) == label)[0]
                                      for label in self.data_label_set}
            self.label_to_indices2 = {label: np.where(np.asarray(self.data_label2) == label)[0]
                                      for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb , self.kp2d_= self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum) * 0.6 or ii == len(idnum) * 0.6:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label

            rgb_p = self.data_rgb[positive_index]
            rgb_n = self.data_rgb[negative_index]
            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            ti_p = self.data_test_ti[positive_index]
            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label

            rgb_p = self.data_test_rgb[positive_index]
            rgb_n = self.data_test_rgb[negative_index]
            labelrgb_p = labelti
            labelrgb_n = negative_label
        return ti, ti_p, ti_n, rgb, rgb_p,rgb_n,labelti, labelti_p, labelti_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kp2d_crop.npy")
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d_crop.npy")

            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            def draw3Dpose(pose_3d, pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
                           add_labels=False):  # blue, orange
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.grid(False)
                for i in smpl_connectivity_dict:
                    x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x, y, z, lw=2, c=lcolor)
                    x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x2, y2, z2, lw=2, c=rcolor)

                RADIUS = 1  # space around the subject
                xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
                ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            def draw3Dpose_frames(ti, data_key_in):
                # 绘制连贯的骨架
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
                i = 0
                j = 0
                while i < ti.shape[0]:
                    draw3Dpose(ti[i], data_key_in[i], ax)

                    plt.pause(0.3)
                    # print(ax.lines)
                    plt.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.lines = []
                    i += 1
                    if i == ti.shape[0]:
                        # i=0
                        j += 1
                    if j == 2:
                        break

                plt.ioff()
                plt.show()

            list_all_ti[:, :, :, 0:1] = self.normalization(list_all_ti[:, :, :, 0:1])
            list_all_ti[:, :, :, 1:2] = self.normalization(list_all_ti[:, :, :, 1:2])
            list_all_ti[:, :, :, 2:3] = self.normalization(list_all_ti[:, :, :, 2:3])
            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full, list_all_kp2d_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_rgb_full_singlergb_offlinetri(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_, self.data_kp2d_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d_)


            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.6:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])


            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            self.label_to_indices1 = {label: np.where(np.asarray(self.data_label1) == label)[0]
                                      for label in self.data_label_set}
            self.label_to_indices2 = {label: np.where(np.asarray(self.data_label2) == label)[0]
                                      for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb , self.kp2d_= self.dataRead()
            # self.data_label=self.data_label.tolist()
            # self.data_label_set=list(set(self.data_label))
            # self.data_label_set.sort(key=self.data_label.index)
            # self.data_label_set = set(self.data_label_set)

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum) * 0.6 or ii == len(idnum) * 0.6:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            labelti_p = labelti
            labelti_n = negative_label

            rgb_p = self.data_rgb[positive_index]
            rgb_n = self.data_rgb[negative_index]
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]
            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]

        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            labelti_p = labelti
            labelti_n = negative_label

            rgb_p = self.data_test_rgb[index]
            rgb_n = self.data_test_rgb[negative_index]
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]
            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[index]
            key_n = self.data_key[negative_index]
        return rgb, rgb_p,rgb_n, labelti, labelti_p, labelti_n,key_a,key_p,key_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kp2d_crop.npy")
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d_crop.npy")

            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            def draw3Dpose(pose_3d, pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
                           add_labels=False):  # blue, orange
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.grid(False)
                for i in smpl_connectivity_dict:
                    x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x, y, z, lw=2, c=lcolor)
                    x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x2, y2, z2, lw=2, c=rcolor)

                RADIUS = 1  # space around the subject
                xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
                ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            def draw3Dpose_frames(ti, data_key_in):
                # 绘制连贯的骨架
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
                i = 0
                j = 0
                while i < ti.shape[0]:
                    draw3Dpose(ti[i], data_key_in[i], ax)

                    plt.pause(0.3)
                    # print(ax.lines)
                    plt.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.lines = []
                    i += 1
                    if i == ti.shape[0]:
                        # i=0
                        j += 1
                    if j == 2:
                        break

                plt.ioff()
                plt.show()

            list_all_ti[:, :, :, 0:1] = self.normalization(list_all_ti[:, :, :, 0:1])
            list_all_ti[:, :, :, 1:2] = self.normalization(list_all_ti[:, :, :, 1:2])
            list_all_ti[:, :, :, 2:3] = self.normalization(list_all_ti[:, :, :, 2:3])
            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full, list_all_kp2d_crop

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_rgb_full_midmodal_offlinetri(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}


            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kp2d_crop.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")
            list_all_kp2d_crop = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kp2d_crop.npy")

            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            def draw3Dpose(pose_3d, pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
                           add_labels=False):  # blue, orange
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.grid(False)
                for i in smpl_connectivity_dict:
                    x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x, y, z, lw=2, c=lcolor)
                    x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
                    # ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x2, y2, z2, lw=2, c=rcolor)

                RADIUS = 1  # space around the subject
                xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
                ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            def draw3Dpose_frames(ti, data_key_in):
                # 绘制连贯的骨架
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
                i = 0
                j = 0
                while i < ti.shape[0]:
                    draw3Dpose(ti[i], data_key_in[i], ax)

                    plt.pause(0.3)
                    # print(ax.lines)
                    plt.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.lines = []
                    i += 1
                    if i == ti.shape[0]:
                        # i=0
                        j += 1
                    if j == 2:
                        break

                plt.ioff()
                plt.show()

            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", list_all_ti.shape)
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

#剔除kinect异常值后的数据集
class SiamesePC_20_rgb_full_midmodal_offlinetri_final(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                '''
                for ii in range(len(idnum)):
                  
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                '''
                
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[index]
            key_n = self.data_test_key[negative_index]

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_final.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_final.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")
            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices_test:",self.label_to_indices_test)
        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_final.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_final.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")
            print("list_all_kinect_key:",list_all_kinect_key.shape)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        print("list_label_all")
        print("list_label_all:", list_label_all)
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train_diffnum(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0,peoplenum=20):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(peoplenum):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''
            self.data_label_set = set(self.data_label)
            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            print("len(self.data_label_set):",len(self.data_label_set))
            # 原始方式
            for id in range(peoplenum):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            self.data_label_set = set(self.data_test_label)
            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_final.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_final.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")
            print("list_all_kinect_key:",list_all_kinect_key.shape)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        print("list_label_all")
        print("list_label_all:", list_label_all)
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

#按人分离
class SiamesePC_20_rgb_full_midmodal_offlinetri_final_persondivided(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分


            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            self.data_label_set_all = set(self.data_label_)
            self.label_to_indices_all = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                     for label in self.data_label_set_all}

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))


            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set_all)):
                num = list(self.data_label_set_all)[id]
                num = num.astype("int")
                idnum = self.label_to_indices_all[list(self.data_label_set_all)[id]]
                if num < 15:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])


            self.data_label_set = set(self.data_label)
            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_rgb:", self.data_rgb.shape)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}


            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()
            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''
            # 原始数据分配方式
            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            self.data_label_set_all = set(self.data_label)
            self.label_to_indices_all = {label: np.where(np.asarray(self.data_label) == label)[0]
                                         for label in self.data_label_set_all}



            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''
            for id in range(len(self.data_label_set_all)):
                idnum = self.label_to_indices_all[list(self.data_label_set_all)[id]]
                num = list(self.data_label_set_all)[id]
                num = num.astype("int")
                # print(len(idnum))
                if num > 14:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])

            self.data_label_set_test = set(self.data_test_label)
            print("data_label_set:", self.data_label_set_test)
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.data_test_key2 = self.data_test_key
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set_test}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all 

    '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            #print("negative_label:",negative_label)
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n = self.data_rgb[negative_index]
            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
            negative_label = np.random.choice(list(self.data_label_set_test - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            rgb_n = self.data_test_rgb[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n, rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_final.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_final.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")


            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_all_rgb_full))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

#测试集两种模态全排列
class SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train_test(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_test_ti, self.data_kinect, self.data_test_label, self.data_index, self.data_test_rgb, self.data_6890, self.data_test_key \
                , self.data_test_rgb = self.dataRead()

            # 原始数据分配方式
            self.data_label_set = set(self.data_test_label)



            self.label_to_indices = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
            self.data_test_key2 = self.data_test_key
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)

            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            print("label_to_indices:", self.label_to_indices)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_final.npy")
            list_index_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final_test.npy")
            list_all_kinect = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_final.npy")
            list_index_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_index_all_final.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final_test.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final_test.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final_test.npy")
            print("list_all_kinect_key:",list_all_ti.shape)
            print("list_all_rgb_full:", list_all_rgb_full.shape)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_20_feature_final_07train(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_rgb_h_, self.data_rgb_l_, self.data_ti_h_, self.data_ti_l_, self.data_label_, self.data_key_= self.dataRead()
            data_label_train = []

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_rgb_h_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_l_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_ti_h_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_ti_l_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum) * 0.7:
                        self.data_ti_h.append(self.data_ti_h_[idnum[ii]])
                        self.data_ti_l.append(self.data_ti_l_[idnum[ii]])
                        self.data_rgb_h.append(self.data_rgb_h_[idnum[ii]])
                        self.data_rgb_l.append(self.data_rgb_l_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key

            self.data_ti_h = np.asarray(self.data_ti_h)
            self.data_ti_l = np.asarray(self.data_ti_l)
            self.data_rgb_h = np.asarray(self.data_rgb_h)
            self.data_rgb_l = np.asarray(self.data_rgb_l)
            self.data_label = np.asarray(self.data_label)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_rgb_h, self.data_rgb_l, self.data_ti_h, self.data_ti_l, self.data_label, self.data_key = self.dataRead()


            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_rgb_h)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_l)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_ti_h)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_ti_l)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            self.data_test_ti_h = []
            self.data_test_ti_l = []
            self.data_test_rgb_h = []
            self.data_test_rgb_l = []


            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum) * 0.7 or ii == len(idnum) * 0.7:
                        self.data_test_ti_h.append(self.data_ti_h[idnum[ii]])
                        self.data_test_ti_l.append(self.data_ti_l[idnum[ii]])
                        self.data_test_rgb_h.append(self.data_rgb_h[idnum[ii]])
                        self.data_test_rgb_l.append(self.data_rgb_l[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key

            self.data_test_ti_h = np.asarray(self.data_test_ti_h)
            self.data_test_ti_l = np.asarray(self.data_test_ti_l)
            self.data_test_rgb_h = np.asarray(self.data_test_rgb_h)
            self.data_test_rgb_l = np.asarray(self.data_test_rgb_l)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            print("test_ti:", self.data_test_ti_h.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb_h.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}


    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            rgb_h,rgb_l, labelrgb = self.data_rgb_h[index],self.data_rgb_l[index],self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelrgb])

            negative_label = np.random.choice(list(self.data_label_set - set([labelrgb])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            # ti_p = self.data_ti[positive_index]
            # 测试同一对象两模态是否重建一致

            # 更改是否属于同一个id的同一次数据
            ti_h_p = self.data_ti_h[positive_index]
            ti_l_p = self.data_ti_l[positive_index]

            ti_h_n = self.data_ti_h[negative_index]
            ti_l_n = self.data_ti_l[negative_index]

            labelti_p = labelrgb
            labelti_n = negative_label
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            rgb_h, rgb_l, labelrgb = self.data_test_rgb_h[index], self.data_test_rgb_l[index], self.data_test_label[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelrgb])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelrgb])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            # 更改是否属于同一个id的同一次数据
            ti_h_p = self.data_test_ti_h[positive_index]
            ti_l_p = self.data_test_ti_l[positive_index]

            ti_h_n = self.data_test_ti_h[negative_index]
            ti_l_n = self.data_test_ti_l[negative_index]
            labelti_p = labelrgb
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]

        return rgb_h,rgb_l, ti_h_p, ti_l_p, ti_h_n, ti_l_n,labelrgb, labelti_p, labelti_n, key_a, key_p, key_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_rgb_h = []
        list_rgb_l = []
        list_ti_h = []
        list_ti_l = []
        list_label_all = []
        list_all_kinect_key = []

        # 读取rgb数据
        if self.issever:
            list_rgb_h = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_rgb_h.npy")
            list_rgb_l = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_rgb_l.npy")
            list_ti_h = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_ti_h.npy")
            list_ti_l = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_ti_l.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_feature.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_key_feature.npy")
            # ti和kincet_key保持一致：是否calibration
        else:
            list_rgb_h = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_rgb_h.npy")
            list_rgb_l = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_rgb_l.npy")
            list_ti_h = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_ti_h.npy")
            list_ti_l = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_ti_l.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_feature.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_key_feature.npy")

        # max_key
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_ti_l))
        print("length of rgb:", len(list_rgb_l))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_rgb_h, list_rgb_l, list_ti_h, list_ti_l, list_label_all, list_all_kinect_key

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_differentviews_25_07train(Dataset):
    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}


            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n = self.data_rgb[negative_index]
            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n,rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_label_all_int.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_rgb_full.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_label_all_int.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_rgb.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")
            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        plt.imshow(int(list_all_rgb_full[0][0]))
        plt.show()
        # max_key
        print(list_max)
        print(list_min)
        list_all_ti = np.concatenate([list_all_ti,list_all_ti2],axis=0)
        list_all_kinect_key = np.concatenate([list_all_kinect_key, list_all_kinect_key2], axis=0)
        list_label_all = np.concatenate([list_label_all, list_label_all2], axis=0)
        list_all_rgb_full = np.concatenate([list_all_rgb_full, list_all_rgb_full2], axis=0)

        print("list_label_all:",type(list_label_all[0]))
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_differentviews_rgb_full_midmodal_offlinetri_07train(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_kinect_, self.data_label_, self.data_index_, self.data_rgb_, self.data_6890_, self.data_key_ \
                , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}


            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_kinect, self.data_label, self.data_index, self.data_rgb, self.data_6890, self.data_key \
                , self.data_rgb = self.dataRead()


            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)

            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set) - 0):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

        '''
        labelkinect2=labelti
        labelkinect =labelti
        while labelkinect2==labelti:
            labelkinect2=np.random.choice(list(self.data_label_set - set([labelti])))
        #print('labelkinect:',labelkinect)
        #print('labelkinect2:', labelkinect2)
        kinect_index=np.random.choice(self.label_to_indices[labelkinect])
        kinect2_index = np.random.choice(self.label_to_indices[labelkinect2])
        kinect1 = self.data_kinect[kinect_index]
        kinect2 = self.data_kinect[kinect2_index]

        batch all '''

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[index]
            key_n = self.data_test_key[negative_index]

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_label_all_int.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb_differentviews/list_all_rgb_full.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_label_all_int.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb_differentviews\list_all_rgb_full.npy")
            smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

            print(4.5 - list_all_kinect_key[0, 0, 0, 1])
            print(list_all_ti[0, 0, 0, :])
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_all_kinect, list_label_all, list_index_all, list_bit, list_all_kinect_6890, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/content/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/content/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/content/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/content/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "/content\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/content\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "/content\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "/content\straight_road_npy\list_all_image.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("/content\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]

        #plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        #plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        print("key:", list_all_kinect_key.shape)
        key = list_all_kinect_key[120:140,:,0,2].reshape(20,20,1)
        print("key:",key.shape)
        import scipy.io
        scipy.io.savemat("/content\\1.mat",
                         mdict={'key': key})

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_calibration(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\dataset0818_npy_calibration\single_angle\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\dataset0818_npy_calibration\single_angle\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\dataset0818_npy_calibration\single_angle\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\dataset0818_npy_calibration\single_angle\list_all_image.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]
        print("key:", list_all_kinect_key.shape)
        key = list_all_kinect_key[:20, :, 0, 2].reshape(20, 20, 1)
        print("key:", key.shape)
        import scipy.io
        scipy.io.savemat("F:\SouthEast\Reid\\rgb_ti_smpl\\res\mat\\1.mat",
                         mdict={'key': key})
        #plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        #plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))
        '''
        print("key:", list_all_kinect_key.shape)
        key = list_all_kinect_key[:20,:,0,2].reshape(20,20,1)
        print("key:",key.shape)
        import scipy.io
        scipy.io.savemat("F:\SouthEast\Reid\\rgb_ti_smpl\\res\mat\\1.mat",
                         mdict={'key': key})
        '''
        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_persondivided(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_, self.data_key_, self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            print("data_label_set:", self.data_label_set)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
            '''
            # id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 53:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)
            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:", self.label_to_indices)

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 52:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2 = self.data_test_key

            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                         self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            # ti_p = self.data_ti[positive_index]
            # 测试同一对象两模态是否重建一致

            # 更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n = self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                         self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            # 测试返回不同id的rgb

        return rgb, ti_p, ti_n, labelrgb, labelti_p, labelti_n, key_a, key_p, key_n, rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            # ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            # print("list_label_all:", list_label_all)
            # for i in range (331):
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:, :, :, :3]

        # plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        # plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))


        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_persondivided_rgb2dhmr(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_, self.data_key_, self.data_rgb_ ,self.data_kp2d_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            print("data_label_set:", self.data_label_set)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle( self.data_kp2d_)


            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            self.data_kp2d = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
            '''
            # id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 53:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_kp2d.append(self.data_kp2d_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_kp2d = np.asarray(self.data_kp2d)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)
            print("train_kp2d:", self.data_kp2d.shape)
            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:", self.label_to_indices)

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb,self.data_kp2d = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            self.data_test_kp2d = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 52:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        self.data_test_kp2d.append(self.data_kp2d[idnum[ii]])
            self.data_test_key2 = self.data_test_key

            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_kp2d = np.asarray(self.data_test_kp2d)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            print("test_kp2d:", self.data_test_kp2d.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb,kp2d = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                         self.data_label[index], self.data_kp2d[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            # ti_p = self.data_ti[positive_index]
            # 测试同一对象两模态是否重建一致

            # 更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n = self.data_rgb[negative_index]
            kp2d_n = self.data_kp2d[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb,kp2d = self.data_test_ti[index], self.data_test_label[index], \
                                         self.data_test_rgb[index], self.data_test_label[index], self.data_test_kp2d[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]
            kp2d_n = self.data_test_kp2d[negative_index]
            # 测试返回不同id的rgb

        return rgb, ti_p, ti_n, labelrgb, labelti_p, labelti_n, key_a, key_p, key_n, rgb_n,kp2d,kp2d_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []
        list_all_kp2d = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            list_all_kp2d = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/HMR/list_all_kp2d.npy")
            # ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            list_all_kp2d = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\HMR\\list_all_kp2d.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            # print("list_label_all:", list_label_all)
            # for i in range (331):
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:, :, :, :3]

        # plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        # plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))


        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full,list_all_kp2d

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_rgb2dhmr(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_, self.data_key_, self.data_rgb_ ,self.data_kp2d_= self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            print("data_label_set:", self.data_label_set)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle( self.data_kp2d_)


            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            self.data_kp2d = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_kp2d.append(self.data_kp2d_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
            '''
            # id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 53:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_kp2d.append(self.data_kp2d_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
                '''
            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_kp2d = np.asarray(self.data_kp2d)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)
            print("train_kp2d:", self.data_kp2d.shape)
            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:", self.label_to_indices)

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb,self.data_kp2d = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_kp2d)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            self.data_test_kp2d = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        self.data_test_kp2d.append(self.data_kp2d[idnum[ii]])

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 52:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        self.data_test_kp2d.append(self.data_kp2d[idnum[ii]])
            self.data_test_key2 = self.data_test_key
 '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            self.data_test_kp2d = np.asarray(self.data_test_kp2d)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            print("test_kp2d:", self.data_test_kp2d.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb,kp2d = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                         self.data_label[index], self.data_kp2d[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            # ti_p = self.data_ti[positive_index]
            # 测试同一对象两模态是否重建一致

            # 更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n = self.data_rgb[negative_index]
            kp2d_n = self.data_kp2d[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb,kp2d = self.data_test_ti[index], self.data_test_label[index], \
                                         self.data_test_rgb[index], self.data_test_label[index], self.data_test_kp2d[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]
            kp2d_n = self.data_test_kp2d[negative_index]
            # 测试返回不同id的rgb

        return rgb, ti_p, ti_n, labelrgb, labelti_p, labelti_n, key_a, key_p, key_n, rgb_n,kp2d,kp2d_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []
        list_all_kp2d = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            list_all_kp2d = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/HMR/list_all_kp2d.npy")
            # ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            list_all_kp2d = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\HMR\\list_all_kp2d.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            # print("list_label_all:", list_label_all)
            # for i in range (331):
            # draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:, :, :, :3]

        # plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        # plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))


        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full,list_all_kp2d

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res


class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_samesample(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:",self.label_to_indices)



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]

        #plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        #plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        list_all_ti = list_all_ti[:331]
        list_label_all = list_label_all[:331]
        list_all_kinect_key = list_all_kinect_key[:331]
        list_all_rgb_full = list_all_rgb_full[:331]

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_samesample_persondivided(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            print("data_label_set:",self.data_label_set)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
            '''
            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 39:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key


            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)
            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:",self.label_to_indices)



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
        

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 38:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2 = self.data_test_key

            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            '''
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]

        #plt.imshow(list_all_rgb_full[202][19].reshape(224,224,3))
        #plt.show()
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        list_all_ti = list_all_ti[:331]
        list_label_all = list_label_all[:331]
        list_all_kinect_key = list_all_kinect_key[:331]
        list_all_rgb_full = list_all_rgb_full[:331]

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_58_rgb_full_midmodal_offlinetri_final_07train(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_ti_final.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_kinect_key_final.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_label_all_final.npy")
            list_all_rgb_full2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/SiamesePC_20_nonormalization_rgb/list_all_rgb_full_final.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_final.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_kinect_key_final.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_label_all_final.npy")
            list_all_rgb_full2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full_final.npy")

            print("list_all_rgb_full:", list_all_rgb_full.shape)
            print("list_all_rgb_full2:", list_all_rgb_full2.shape)

            print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = np.concatenate([list_all_ti[:, :, :, :3], list_all_ti2[:, :20, :, :3]], axis=0)
        list_all_kinect_key = np.concatenate([list_all_kinect_key, list_all_kinect_key2[:, :20, :, :]], axis=0)
        list_label_all = np.concatenate([list_label_all, list_label_all2], axis=0)
        list_all_rgb_full = np.concatenate([list_all_rgb_full, list_all_rgb_full2[:, :20, :]], axis=0)
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_cedui(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy")
            list_all_rgb_full = np.reshape(list_all_rgb_full,(560,10,3,224,224))
            print(list_all_rgb_full.shape)
            '''
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy", list_all_rgb_full)
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_cedui_persondivided(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_, self.data_key_, self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            print("data_label_set:", self.data_label_set)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}
            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key
            '''
            # id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 53:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            self.data_key2 = self.data_key

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)
            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            print("label_to_indices:", self.label_to_indices)

            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 52:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            self.data_test_key2 = self.data_test_key

            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_ti.npy")

            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy")
            list_all_rgb_full = np.reshape(list_all_rgb_full,(560,10,3,224,224))
            print(list_all_rgb_full.shape)
            '''
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy", list_all_rgb_full)
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''
            self.data_key2 = self.data_key
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_image.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            print(list_label_all2)
            print(list_label_all)
            '''
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy", list_all_rgb_full)
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]
        list_all_ti2 = list_all_ti2[:, :, :, :3]
        list_all_ti = np.concatenate([list_all_ti, list_all_ti2[:, :10, :, :]], axis=0)
        list_all_kinect_key = np.concatenate([list_all_kinect_key, list_all_kinect_key2[:, :10, :, :3]], axis=0)
        list_label_all = np.concatenate([list_label_all, list_label_all2], axis=0)
        list_all_rgb_full = np.concatenate([list_all_rgb_full, list_all_rgb_full2[:, :10, :]], axis=0)
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_persondivided(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)

            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []
            '''
            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
          
            
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if int(num) < 53:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key
            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.data_label_set = set(self.data_label)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)



            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []
            '''
            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if int(num) > 52:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])

            self.data_test_key2 = self.data_test_key
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)
            self.data_label_set = set(self.data_test_label)
            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                          for label in self.data_label_set}
            print("label_to_indices_test:", self.label_to_indices_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_image.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            print(list_label_all2)
            print(list_label_all)
            '''
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy", list_all_rgb_full)
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])
        list_all_ti = list_all_ti[:,:,:,:3]
        list_all_ti2 = list_all_ti2[:, :, :, :3]
        list_all_ti = np.concatenate([list_all_ti, list_all_ti2[:, :10, :, :]], axis=0)
        list_all_kinect_key = np.concatenate([list_all_kinect_key, list_all_kinect_key2[:, :10, :, :3]], axis=0)
        list_label_all = np.concatenate([list_label_all, list_label_all2], axis=0)
        list_all_rgb_full = np.concatenate([list_all_rgb_full, list_all_rgb_full2[:, :10, :]], axis=0)
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_rgbcedui_mmzhengdui(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,self.data_label_2,  self.data_key_ ,self.data_key_2 , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            self.data_label_set2 = set(self.data_label_2)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_2)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_2)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label_2) == label)[0]
                                      for label in self.data_label_set2}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_rgb[list(self.data_label_set)[id]]

                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''

            for id in range(len(self.data_label_set)):

                idnum = self.label_to_indices_ti[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum) * 0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label2.append(self.data_label_2[idnum[ii]])
                        self.data_key2.append(self.data_key_2[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_label2 = np.asarray(self.data_label2)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)


            print("train_ti:", self.data_ti.shape)
            print("train_label_rgb:", self.data_label.shape)
            print("train_label_ti:", self.data_label2.shape)
            print("train_key:", self.data_key.shape)
            print("train_key2:", self.data_key2.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                         for label in self.data_label_set}
            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label_2) == label)[0]
                                        for label in self.data_label_set2}



            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label,self.data_label2,  self.data_key ,self.data_key2 , self.data_rgb  = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)
            self.data_label_set2 = set(self.data_label2)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label2)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key2)

            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label) == label)[0]
                                         for label in self.data_label_set}

            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label2) == label)[0]
                                        for label in self.data_label_set2}


            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_rgb[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])

            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_ti[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label2.append(self.data_label2[idnum[ii]])
                        self.data_test_key2.append(self.data_key2[idnum[ii]])


                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_label2 = np.asarray(self.data_test_label2)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_label2:", self.data_test_label2.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_key2:", self.data_test_key2.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_rgb_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                         for label in self.data_label_set}
            self.label_to_indices_ti_test = {label: np.where(np.asarray(self.data_test_label2) == label)[0]
                                        for label in self.data_label_set2}


    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            # ti_p = self.data_test_ti[positive_index]
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/cedui_npy/list_all_image.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/straight_road_npy/list_all_image.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_label_all.npy")
            list_all_rgb_full2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
            print(list_label_all2)
            print(list_label_all)
            '''
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_all_image.npy", list_all_rgb_full)
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))
            # draw3Dpose_frames(list_all_kinect_key[20, :, :, :], list_all_kinect_key[10, :, :, :])

        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti2, list_label_all,list_label_all2, list_all_kinect_key,list_all_kinect_key2, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_rgbzhengdui_mmdiffview(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train=True, issever=0,angle=15):
        self.train = train
        self.issever = issever
        self.angle = angle
        if self.train:  # train
            # read data
            self.data_ti_, self.data_label_,self.data_label_2,  self.data_key_ ,self.data_key_2 , self.data_rgb_ = self.dataRead()
            data_label_train = []
            '''
            for labels in self.data_label_:
            # 控制用于训练的id
                if labels < 38:
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)

            '''

            # 原始数据划分
            self.data_label_set = set(self.data_label_)
            self.data_label_set2 = set(self.data_label_2)
            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label_2)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key_2)



            # pyplot_draw_point_cloud2(self.data_key_[0][0])

            # self.data_ti=self.data_ti[:len(self.data_ti)*0.8]
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label_2) == label)[0]
                                      for label in self.data_label_set2}

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_rgb[list(self.data_label_set)[id]]

                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
                '''
                if len(idnum) > 12:  # 取12条数据进行训练
                    for ii in range(12):
                        # print(idnum[ii])
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

                else:
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])
            '''

            for id in range(len(self.data_label_set)):

                idnum = self.label_to_indices_ti[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum) * 0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label2.append(self.data_label_2[idnum[ii]])
                        self.data_key2.append(self.data_key_2[idnum[ii]])
            '''


            #id分离
            for id in range(len(self.data_label_set)):
                num = list(self.data_label_set)[id]
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                if num < 38:
                    print(num)
                    for ii in range(len(idnum)):
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_index.append(self.data_index_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
'''

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_label2 = np.asarray(self.data_label2)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)
            self.data_key2 = np.asarray(self.data_key2)


            print("train_ti:", self.data_ti.shape)
            print("train_label_rgb:", self.data_label.shape)
            print("train_label_ti:", self.data_label2.shape)
            print("train_key:", self.data_key.shape)
            print("train_key2:", self.data_key2.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label) == label)[0]
                                         for label in self.data_label_set}
            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label2) == label)[0]
                                        for label in self.data_label_set2}




            # print(self.label_to_indices['20caodongjiang'])
            # self.train_data_pairs=self.generate_data_pair(self.data_ti,self.data_kinect)
        else:  # test
            # read data
            self.data_ti, self.data_label,self.data_label2,  self.data_key ,self.data_key2 , self.data_rgb  = self.dataRead()

            '''
            data_label_train = []
            for labels in self.data_label:
                # 控制用于训练的id
                if labels > 37 :
                    data_label_train.append(labels)
            self.data_label_set = set(data_label_train)
            '''

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)
            self.data_label_set2 = set(self.data_label2)

            # random_state = np.random.RandomState(1)
            # rndom_state.shuffle(self.data_ti)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_kinect)
            # random_state = np.random.RandomState(1)
            # random_state.shuffle(self.data_label)
            # for label in self.data_label_set:
            #     print(label)
            #     print(np.where(np.asarray(self.data_label)==label))
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label2)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key2)

            self.label_to_indices_rgb = {label: np.where(np.asarray(self.data_label) == label)[0]
                                         for label in self.data_label_set}

            self.label_to_indices_ti = {label: np.where(np.asarray(self.data_label2) == label)[0]
                                        for label in self.data_label_set2}


            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_rgb[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])

            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_ti[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label2.append(self.data_label2[idnum[ii]])
                        self.data_test_key2.append(self.data_key2[idnum[ii]])


                '''
                idnum = idnum.tolist()
                idnum.reverse()
                if len(idnum) > 8:  # 15
                    for ii in range(8):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
                        # print(idnum[len(idnum)-1-ii])
                else:
                    for ii in range(len(idnum)):

                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])
            '''

            self.data_test_key2 = self.data_test_key
            '''
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices[list(self.data_label_set)[id]]
                idnum = idnum.tolist()
                idnum.reverse()
                num = list(self.data_label_set)[id]
                # print(len(idnum))
                if num > 37:
                    # print(111)
                    for ii in range(len(idnum)):
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
            '''
            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_label2 = np.asarray(self.data_test_label2)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_key2 = np.asarray(self.data_test_key2)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_label2:", self.data_test_label2.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_key2:", self.data_test_key2.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_rgb_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                         for label in self.data_label_set}
            self.label_to_indices_ti_test = {label: np.where(np.asarray(self.data_test_label2) == label)[0]
                                        for label in self.data_label_set2}


    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            rgb, labelrgb =  self.data_rgb[index], self.data_label[index]
            labelti = labelrgb
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_ti[labelti])

            negative_label_ti = np.random.choice(list(self.data_label_set2 - set([labelti])))
            negative_index_ti = np.random.choice(self.label_to_indices_ti[negative_label_ti])

            negative_label_rgb = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index_rgb = np.random.choice(self.label_to_indices_rgb[negative_label_rgb])

            #ti_p = self.data_ti[positive_index]
            #测试同一对象两模态是否重建一致

            #更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index_ti]
            labelti_p = labelti
            labelti_n = negative_label_ti
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index_ti]
            rgb_n =  self.data_rgb[negative_index_rgb]


        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            rgb, labelrgb = self.data_test_rgb[index], self.data_test_label[index]
            labelti = labelrgb
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_ti_test[labelti])

            negative_label_ti = np.random.choice(list(self.data_label_set2 - set([labelti])))
            negative_index_ti = np.random.choice(self.label_to_indices_ti_test[negative_label_ti])

            negative_label_rgb = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index_rgb = np.random.choice(self.label_to_indices_rgb_test[negative_label_rgb])

            # ti_p = self.data_ti[positive_index]
            # 测试同一对象两模态是否重建一致

            # 更改是否属于同一个id的同一次数据
            ti_p = self.data_ti[positive_index]
            ti_n = self.data_ti[negative_index_ti]
            labelti_p = labelti
            labelti_n = negative_label_ti
            key_a = self.data_key[index]
            # key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index_ti]
            rgb_n = self.data_rgb[negative_index_rgb]

            #测试返回不同id的rgb

        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n
        # return (ti_voxel, kinect1_voxel, labelti, labelkinect), target

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_6890 = []
        list_all_kinect_key = []
        list_all_kinect = []
        list_label_all = []  # 人名字
        list_index_all = []  # id
        list_bit = []  # 二值图像
        list_label_rgb = []  # 人名字
        len_total = []
        len_total_rgb = []
        list_max = []
        list_min = []

        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_1/list_all_ti.npy")
            list_all_kinect_key = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_1/list_all_kinect_key.npy")
            list_label_all = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_1/list_label_all.npy")
            list_all_rgb_full = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_1/list_all_image.npy")

            list_all_ti2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_2/list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_2/list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_2/list_label_all.npy")

            list_all_ti3 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_3/list_all_ti.npy")
            list_all_kinect_key3 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_3/list_all_kinect_key.npy")
            list_label_all3 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_3/list_label_all.npy")

            list_all_ti4 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_4/list_all_ti.npy")
            list_all_kinect_key4 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_4/list_all_kinect_key.npy")
            list_label_all4 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_4/list_label_all.npy")

            list_all_ti5 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_5/list_all_ti.npy")
            list_all_kinect_key5 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_5/list_all_kinect_key.npy")
            list_label_all5 = np.load(
                "/home/sdb/yaotianshun/Reid/npydata/0818/multi_angle_npy/road_5/list_label_all.npy")
            #ti和kincet_key保持一致：是否calibration
        else:
            list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_1\list_all_ti.npy")
            list_all_kinect_key = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_1\list_all_kinect_key.npy")
            list_label_all = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_1\list_label_all.npy")
            list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_1\list_all_image.npy")

            list_all_ti2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_2\list_all_ti.npy")
            list_all_kinect_key2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_2\list_all_kinect_key.npy")
            list_label_all2 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_2\list_label_all.npy")

            list_all_ti3 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_3\list_all_ti.npy")
            list_all_kinect_key3 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_3\list_all_kinect_key.npy")
            list_label_all3 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_3\list_label_all.npy")

            list_all_ti4 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_4\list_all_ti.npy")
            list_all_kinect_key4 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_4\list_all_kinect_key.npy")
            list_label_all4 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_4\list_label_all.npy")

            list_all_ti5 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_5\list_all_ti.npy")
            list_all_kinect_key5 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_5\list_all_kinect_key.npy")
            list_label_all5 = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_5\list_label_all.npy")
            print(list_label_all2)
            print(list_label_all)
            # plt.imshow(list_all_rgb_full[0][0])
            # plt.show()
            '''
            list_all_rgb_full3 = np.reshape(list_all_rgb_full3,(51,20,3,224,224))
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_3\list_all_image.npy", list_all_rgb_full3)
            list_all_rgb_full4 = np.reshape(list_all_rgb_full4, (50, 20, 3, 224, 224))
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_4\list_all_image.npy",
                    list_all_rgb_full4)
            list_all_rgb_full5 = np.reshape(list_all_rgb_full5, (41, 20, 3, 224, 224))
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\multi_angle_npy\\road_5\list_all_image.npy",
                    list_all_rgb_full5)
            
            list_label_all = np.asarray(list_label_all, dtype=int)
            list_label_all = list_label_all+25
            list_label_all = np.asarray(list_label_all, dtype=str)
            np.save("F:\SouthEast\Reid\Reid_data\\npydata\\0818\cedui_npy\list_label_all.npy",list_label_all)
            '''
            #print("list_label_all:", list_label_all)
            #for i in range (331):
                #draw3Dpose_frames(list_all_kinect_key[0,:,:,:])
            # print("list_all_kinect_key:",list_all_kinect_key[0,:,:,1]+(4.5-list_all_kinect_key[0,0,0,1]))

            draw3Dpose_frames(list_all_kinect_key[10, :, :, :])
        list_all_kinect_key_rgb = list_all_kinect_key
        list_label_all_rgb = list_label_all
        if self.angle == 15:
            list_all_ti_out = list_all_ti2
            list_label_all_ti = list_label_all2
            list_all_kinect_key_ti = list_all_kinect_key2
        elif self.angle == 30:
            list_all_ti_out = list_all_ti3
            list_label_all_ti = list_label_all3
            list_all_kinect_key_ti = list_all_kinect_key3
        elif self.angle == 45:
            list_all_ti_out = list_all_ti4
            list_label_all_ti = list_label_all4
            list_all_kinect_key_ti = list_all_kinect_key4
        elif self.angle == 60:
            list_all_ti_out = list_all_ti5
            list_label_all_ti = list_label_all5
            list_all_kinect_key_ti = list_all_kinect_key5
        # max_key
        print(list_max)
        print(list_min)
        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_label_rgb))

        print("list_label_all_ti:", list_label_all_ti.shape)
        print("list_label_all_rgb:", list_label_all_rgb.shape)
        list_all_ti_out = list_all_ti_out[:, :, :, :3]
        # list_all_ti=point_cloud_to_volume(list_all_ti,20,1.0)
        # list_all_kinect = point_cloud_to_volume(list_all_kinect, 20, 1.0)
        return list_all_ti_out, list_label_all_rgb,list_label_all_ti, list_all_kinect_key_rgb,list_all_kinect_key_ti, list_all_rgb_full

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        # res=preprocessing.MaxAbsScaler().fit_transform(data)
        return ((data - np.min(data)) / _range) - 0.5
        # return res

if __name__ == '__main__':
    #list_all_rgb_full = np.load("F:\SouthEast\Reid\Reid_data\\npydata\\0818\straight_road_npy\list_all_image.npy")
    #list_all_rgb_full = list_all_rgb_full.reshape(680,20,224,224,3)
    #print("list_all_rgb_full:",list_all_rgb_full.shape)
    #res = T.RandomHorizontalFlip(p=0.5)(list_all_rgb_full[0])

    #test_data = SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train(train=False,issever=0)
    #test_data = SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train_diffnum(train=False,issever=0,peoplenum=20)
    test_data = SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_rgbzhengdui_mmdiffview(train=False,issever=0)
    #test_data = SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_samesample(train=False,issever=0)

    #test_data = SiamesePC_differentviews_25_07train(train=False)
    #test_data = SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train(train=False)

'''
print("fuck!")
if torch.cuda.is_available():
    device = 'cuda:%d' % (0)
else:
    device = 'cpu'
model = mmWaveModel().to(device)
model2 = Discriminator().to(device)
learning_rate = 0.0002
learning_rate_d = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(model2.parameters(), lr=learning_rate_d)
adversarial_loss = torch.nn.MSELoss()
batchsize=8
length_size=25
train_data=SiamesePC(train=True)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
for epoch in range(700):
    model.train()
    for batch_idx,(data,target) in tqdm(enumerate(train_loader)):
        data=np.asarray(data)
        #batch_size, seq_len, point_num, dim = data[0].shape
        #data_ti_in = np.reshape(data[0], (batch_size * seq_len, point_num, dim))
        data_ti = torch.tensor(data[0], dtype=torch.float32, device=device).squeeze()
        # data_rgb_in = np.reshape(data[4], (batch_size,3, 720, 1280))    #维度调整(720,1280,3)->(3,720,1280)
        # data_rgb = torch.tensor(data[2][:,:,:,24:264,:], dtype=torch.float32,device=device)   #rgb
        data_rgb = torch.tensor(data[2], dtype=torch.float32, device=device)  # rgb
        batch_size, seq_len, point_num, dim = data[4].shape
        data_6890_in = np.reshape(data[4], (batch_size * seq_len, point_num, dim))
        data_6890 = torch.tensor(data_6890_in, dtype=torch.float32, device=device)  #
        batch_size, seq_len, point_num, dim = data[5].shape
        data_key_in = np.reshape(data[5], (batch_size * seq_len, point_num, dim))
        data_key = torch.tensor(data_key_in, dtype=torch.float32, device=device)  #

        optimizer.zero_grad()
        h0 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
        c0 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
        # out_bit=model2(data)
        # print(out_bit.shape)
        q, t, b, v, s, q_b_dis = model(data_ti, data_rgb, h0, c0, batchsize, length_size)

        criterion = nn.L1Loss(reduction='sum')
        loss_v = criterion(v, data_6890)
        loss_s = criterion(s, data_key)
        #print("epoch{}:".format(epoch), loss_s)
        loss = loss_s + 0.001 * loss_v
        loss.backward()
        optimizer.step()
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # Discrimator
        h0_d = torch.zeros((3, batchsize, 91), dtype=torch.float32, device=device)
        c0_d = torch.zeros((3, batchsize, 91), dtype=torch.float32, device=device)
        g_vec, g_loc, hn, cn = model2(q_b_dis, h0_d, c0_d)
        d_loss = adversarial_loss(g_loc, target)
        d_loss.backward()
        optimizer_d.step()
        print(d_loss)
        #pyplot_draw_point_cloud(data_6890.cpu().detach()[0])
        #print(data_6890.cpu().detach()[0].shape)
        print(target)


'''
