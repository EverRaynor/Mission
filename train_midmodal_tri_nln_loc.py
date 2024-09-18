import torch
import torch.nn as nn
#from torchnet import meter
import numpy as np
import random
from torch.utils.data import DataLoader
from losses import TripletLoss
from network_tcmr import mid_modal_hmr_train_2nln_loc
from network_tcmr import mid_modal_hmr_train_nln_loc_S2MA
from network_tcmr import mid_modal_hmr_train_2nln_loc_0429
from network_tcmr import mid_modal_hmr_train_2nln_loc_0503
from network_tcmr import mid_modal_hmr_train_2nln_loc_0505
from network_tcmr import mid_modal_hmr_train_2nln_loc_0510
from network_tcmr import mid_modal_hmr_train_2nln_loc_0511
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14_1regressor
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_noatten
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion
from network_tcmr import mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_woh
from dataset_me_rgb import SiamesePC_20_rgb_full_midmodal_offlinetri_final_07train
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_samesample
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_samesample_persondivided
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_persondivided
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui_calibration
from dataset_me_rgb import SiamesePC_20_rgb_full_midmodal_offlinetri_final_persondivided
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train
from dataset_me_rgb import SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui
from dataset_me_rgb import SiamesePC_58_rgb_full_midmodal_offlinetri_final_07train

from dataset_me_rgb import SiamesePC_differentviews_25_07train
from dataset_me_rgb import SiamesePC_differentviews_rgb_full_midmodal_offlinetri_07train
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T


cudanum = 0
if torch.cuda.is_available():
    device = 'cuda:%d' % (cudanum)
else:
    device = 'cpu'

#s和t分开计算
ifst= 0
iftrain = 0
ifallfusion = 1
import platform
if ('Windows' == platform.system()):
    issever = 0
else:
    issever = 1
ifrunonce = 1
#随机取步态周期
gaitrandom = 0
#mutual attention是否监督self attention输出
self_atten = 1
num_epochs = 5000
learning_rate = 0.0002
#性能最优值
top1_best = 0
top5_best = 0
top10_best = 0
kprgb_best = 100
kpti1_best = 100
kpti2_best = 100
top1_best_test = 0
top5_best_test = 0
top10_best_test = 0
map_best_test = 0
top1_best_test_l = 0
top5_best_test_l = 0
top10_best_test_l = 0
map_best_test_l = 0
top1_best_test_h = 0
top5_best_test_h = 0
top10_best_test_h = 0
map_best_test_h = 0
kprgb_best_test = 100
kpti1_best_test = 100
kpti2_best_test = 100
if issever == 0:
    batchsize=4
    batchsize2 = 4
else:
    batchsize = 4
    batchsize2 = 4

smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]
'''
def draw3Dpose(pose_3d, pose_3d2,pose_3d3,pose_3d4, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2, y2, z2, lw=2, c="red")
        x3, y3, z3 = [np.array([pose_3d3[i[0], j], pose_3d3[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x3, y3, z3, lw=2, c='black')
        x4, y4, z4 = [np.array([pose_3d4[i[0], j], pose_3d4[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x4, y4, z4, lw=2, c='grey')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
'''

def draw3Dpose(pose_3d, pose_3d2, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2, y2, z2, lw=2, c="red")

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def draw3Dpose_only(pose_3d, pose_3d2,pose_3d3,  lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2, y2, z2, lw=2, c="red")
        x3, y3, z3 = [np.array([pose_3d3[i[0], j], pose_3d3[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x3, y3, z3, lw=2, c='black')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()



def draw3Dpose_frames(rgb,data_key_rgb):
    # 绘制连贯的骨架
    fig = plt.figure()
    #fig, axs = plt.subplots(2, 2, figsize=(4, 4))
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.view_init(0, 90)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.view_init(90, 0)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.view_init(0, 0)
    plt.ion()
    i = 0
    j = 0
    while i < rgb.shape[0]:
        draw3Dpose(rgb[i], data_key_rgb[i], ax)
        draw3Dpose(rgb[i], data_key_rgb[i], ax2)
        draw3Dpose(rgb[i], data_key_rgb[i], ax3)
        draw3Dpose(rgb[i], data_key_rgb[i], ax4)
        plt.pause(0.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.view_init(0, 90)
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.view_init(90, 0)
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.view_init(0, 0)
        # ax.lines = []
        i += 1
        if i == rgb.shape[0]:
            # i=0
            j += 1
        if j == 2:
            break

    plt.ioff()
    plt.show()

def pyplot_draw_bar(data):
    fig = plt.figure()
    for i in range(len(data)):
        plt.bar(i,data[i])
    plt.show()

#38人只有20帧，所以+5
length_size=10
length_size_total=20
criterion_keypoints = nn.MSELoss(reduction='none').to(device)
writer1=SummaryWriter('./loss/midmodal_allfusion_38_divded')
matname = './res/midmodal_allfusion_38_divded.mat'
def save_models(epoch):
    path = "./log/midmodal_allfusion_38_divded"
    import os
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + "/model_{}.pth".format(epoch))

#model=mid_modal_hmr_train_2nln_loc(device).to(device)
#model.load('./log/midmodal_nln_0.7train_loc_differentviews/model_{}.pth'.format('best'))
#model=mid_modal_hmr_train_nln_loc_0423(device).to(device)
#model.load('./log/midmodal_nln_split_0.7train_loc/model_{}.pth'.format('best'))
#model.load('./log/midmodal_nln_split_0.7train_loc_softmax/model_{}.pth'.format('best'))
#model=mid_modal_hmr_train_2nln_loc_0503(device).to(device)
#model.load('./log/midmodal_2nln_0503/model_{}.pth'.format('best'))
#model=mid_modal_hmr_train_2nln_loc_0505(device).to(device)
#model=mid_modal_hmr_train_2nln_loc_0510(device).to(device)
#model.load('./log/midmodal_2nln_0510/model_{}.pth'.format('best'))
#model=mid_modal_hmr_train_2nln_loc_0511(device).to(device)
#model.load('./log/midmodal_2nln_0511/model_{}.pth'.format('best'))
#model.load('./log/midmodal_2nln_0505/model_{}.pth'.format('best'))
#model=mid_modal_hmr_train_2nln_loc_0429(device).to(device)
#model.load('./log/midmodal_2nln_0507/model_{}.pth'.format('best'))
#model.load('./log/midmodal_2nln_0502/model_{}.pth'.format('best'))
#model.load('./log/midmodal_2nln_0501/model_{}.pth'.format('best'))
#model.load('./log/midmodal_2nln_0429/model_{}.pth'.format('best'))
#model.load('./log/midmodal_nln_0.7train_loc_S2MA/model_{}.pth'.format('best'))
#model.load('./log/midmodal_nln_0.7train_loc_S2MA_1metric/model_{}.pth'.format('best'))
#model = mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14(device).to(device)
#model = mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14_1regressor(device).to(device)
#model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb(device).to(device)
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len20_featuremap14_1regressor_2rgb_randsample/model_{}.pth'.format('best'))
#model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_noatten(device).to(device)
#model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres(device).to(device)
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres/model_{}.pth'.format('best'))

#model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion(device).to(device)
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion1+2+3/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion1+4/model_{}.pth'.format('best'))

model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_woh(device).to(device)
model.load('./log/final_model_38/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion_58/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion_38_samesample/model_{}.pth'.format('best'))
#model.load('./log/midmodal_pixelatten_len10_featuremap14_1regressor_2rgb_randsample_nlnres_allfusion_38_divded/model_{}.pth'.format('best'))
#model.load('./log/midmodal_allfusion_38_divded/model_{}.pth'.format('best'))

margin=0.3
loss_fn = TripletLoss(margin)
idloss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_data = SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui(train=False,issever=issever)
test_loader = DataLoader(test_data, batch_size = batchsize2, shuffle = True,drop_last=True)
if iftrain==1:
    train_data = SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui(issever=issever)
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
loss_total=0
eval_loss_total=0

def evaluate(qf, ql, gf, gl):
    query = qf
    #print("ql:",ql)
    #print("gl:", gl)
    # score = np.dot(gf, query)
    score=np.sum(np.square(query - gf),1)
    #print(i, query )
    #print(i, np.square(query - gf))
    #print(i, np.sum(np.square(query - gf),1))
    #print(i, gf.shape)
    # score = (query .- gf).pow(2).sum(1)
    # predict index
    index = np.argsort(score)  # from small to large
    #print(i, score)
    # index = index[::-1] ### 取从后向前（相反）的元素
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    # junk_index = 0  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def evaluate_l(qf, ql, gf, gl):
    query = qf
    #print("ql:",ql)
    #print("gl:", gl)
    # score = np.dot(gf, query)
    score=np.sum(np.square(query - gf),1)
    #print(i, query )
    #print(i, np.square(query - gf))
    #print(i, np.sum(np.square(query - gf),1))
    #print(i, gf.shape)
    # score = (query .- gf).pow(2).sum(1)
    # predict index
    index = np.argsort(score)  # from small to large
    #print(i, score)
    # index = index[::-1] ### 取从后向前（相反）的元素
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    # junk_index = 0  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp,score,index

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    #print("rows_good:",rows_good.shape)
    #print("ngood:", ngood)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

for epoch in range(num_epochs):
    gallery_feature = []
    query_feature = []
    gallery_feature_kp = []
    query_feature_kp = []
    gallery_label = []
    query_label = []
    train_rgb_skeleton = []
    train_ti_skeleton = []
    train_rgb_gt = []
    train_ti_gt = []
    flag_single = 0
    flag_single_kp = 0
    flag_multi = 0
    #epoch=epoch+399
    if (epoch+1)%5==0:
        print("epoch: {}".format(epoch+1))

    if iftrain==1:
        training_loss = []
        training_idloss = []
        training_loss_s1 = []
        training_loss_s2 = []
        training_loss_s3 = []
        training_loss_loc = []
        training_loss_s1_mpjpe = []
        training_loss_s2_mpjpe = []
        training_loss_s3_mpjpe = []
        training_loss_s1_mpjpe_self = []
        training_loss_s2_mpjpe_self = []
        training_loss_s1_mpjpe_t = []
        training_loss_s2_mpjpe_t = []
        training_loss_s3_mpjpe_t = []
        model.train()
        for batch_idx,data in tqdm(enumerate(train_loader)):
            data=np.asarray(data)



            batch_size, seq_len, point_num, dim = data[1][:, length_size_total - length_size:, :].shape
            #print("dim:",dim)
            data_ti_in = np.reshape(data[1][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_ti = torch.tensor(data_ti_in, dtype=torch.float32, device=device).squeeze()

            # print("ti2:",data[1].shape)
            batch_size, seq_len, point_num, dim = data[2][:, length_size_total - length_size:, :].shape
            data_ti_in2 = np.reshape(data[2][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_ti2 = torch.tensor(data_ti_in2, dtype=torch.float32, device=device).squeeze()

            batch_size, seq_len, dim, x, y = data[0][:, length_size_total - length_size:, :].shape
            data_rgb_in = np.reshape(data[0][:, length_size_total - length_size:, :, :, :], (batch_size * seq_len, dim, x, y))
            data_rgb = torch.tensor(data_rgb_in, dtype=torch.float32, device=device)

            batch_size, seq_len, dim, x, y = data[9][:, length_size_total - length_size:, :].shape
            data_rgb_in2 = np.reshape(data[9][:, length_size_total - length_size:, :, :, :], (batch_size * seq_len, dim,x,y))
            data_rgb2 = torch.tensor(data_rgb_in2, dtype=torch.float32, device=device)

            batch_size, seq_len, point_num, dim = data[6][:, length_size_total - length_size:, :].shape
            data_key_in = np.reshape(data[6][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key = torch.tensor(data_key_in, dtype=torch.float32, device=device)

            data_key_in2 = np.reshape(data[7][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key2 = torch.tensor(data_key_in2, dtype=torch.float32, device=device)

            data_key_in3 = np.reshape(data[8][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key3 = torch.tensor(data_key_in3, dtype=torch.float32, device=device)
            data[3] = np.asarray(data[3], dtype=int)
            data[4] = np.asarray(data[4], dtype=int)
            data[5] = np.asarray(data[5], dtype=int)

            optimizer.zero_grad()
            h0 = torch.zeros((6, batchsize, 64), dtype=torch.float32, device=device)
            c0 = torch.zeros((6, batchsize, 64), dtype=torch.float32, device=device)
            #kp-reid-ptnet-lstm
            h1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
            c1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)

            rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2, key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self2,rgb_l2,rgb_l_dif,ti_dif = \
                model(data_rgb,data_rgb2, data_ti, data_ti2, h0, c0, batchsize, length_size)



            #loss_metric_h = loss_fn(rgb_h,ti_h,ti_h2)
            loss_metric_l = loss_fn(rgb_l, ti_l, ti_l2)
            loss_metric_all = loss_fn(output1, output2, output3)

            loss_metric_l0 = loss_fn(ti_l, rgb_l, rgb_l2)
            loss_metric_l_dif1 = loss_fn(rgb_l_dif, ti_l2, ti_l)
            #allfusion2
            loss_metric_l_dif2 = loss_fn(ti_dif, rgb_l, rgb_l2)
            loss_metric_l_dif3 = loss_fn(ti_dif, rgb_l, rgb_l_dif)
            loss_metric_l_dif4 = loss_fn(rgb_l, ti_l, ti_dif)
            loss_metric_l_dif5 = loss_fn(ti_l, rgb_l, rgb_l_dif)
            loss_metric_l_dif = loss_metric_l_dif1



            loss_s1_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s2_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s1_mpjpe = torch.mean(loss_s1_part)
            loss_s2_mpjpe = torch.mean(loss_s2_part)
            loss_s3_mpjpe = torch.mean(loss_s3_part)
            loss_keypoint = (loss_s1_mpjpe + loss_s2_mpjpe + loss_s3_mpjpe) / 3

            # self attention结果监督
            loss_s1_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb_self[:, :10], key_pre_rgb_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s2_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self[:, :10], key_pre_ti_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self2[:, :10], key_pre_ti_self2[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s1_mpjpe_self = torch.mean(loss_s1_part_self)
            loss_s2_mpjpe_self = torch.mean(loss_s2_part_self)
            loss_s3_mpjpe_self = torch.mean(loss_s3_part_self)
            loss_keypoint =(loss_keypoint + (loss_s1_mpjpe_self+loss_s2_mpjpe_self+loss_s3_mpjpe_self)/3)/2

            #loc-loss
            loss_loc_p1=torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p1-data_key2[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_p2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p2 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n1 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n2 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc = (loss_loc_p1+loss_loc_p2+loss_loc_n1+loss_loc_n2) / 4
            training_loss_loc.append(loss_loc.item())

            training_loss_s1_mpjpe.append(loss_s1_mpjpe.item())
            training_loss_s2_mpjpe.append(loss_s2_mpjpe.item())
            training_loss_s3_mpjpe.append(loss_s3_mpjpe.item())



            # self attention结果监督
            training_loss_s1_mpjpe_self.append(loss_s1_mpjpe_self.item())
            training_loss_s2_mpjpe_self.append(loss_s2_mpjpe_self.item())


            key_rgb_out = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1).view(batch_size, length_size, -1)
            key_ti_out = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1).view(batch_size,length_size,-1)
            key_ti2_out = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1).view(batch_size, length_size, -1)

            key_rgb_out2 = torch.flatten(key_rgb_out, start_dim=1, end_dim=2)
            key_ti_out2 = torch.flatten(key_ti_out, start_dim=1, end_dim=2)
            key_ti2_out2 = torch.flatten(key_ti2_out, start_dim=1, end_dim=2)

            show_rgb = key_pre_rgb.cpu().detach()
            show_ti = key_pre_ti2.cpu().detach()
            show_gt_rgb = data_key.cpu().detach()
            show_gt_ti = data_key3.cpu().detach()

            a = 0.5
            loss = a * loss_keypoint + a*loss_loc +loss_metric_l_dif+ a * loss_metric_l+ a * loss_metric_all
            loss.backward()
            optimizer.step()
            #training_idloss.append(loss_id.item())
            training_loss.append(loss.item())
            for i in range(batchsize):
                train_rgb_gt.append(data[6][:, length_size_total - length_size:, :].cpu().detach().numpy()[i])
                train_ti_gt.append(data[7][:, length_size_total - length_size:, :].cpu().detach().numpy()[i])
                train_rgb_skeleton.append(key_rgb_out.cpu().detach().numpy()[i])
                train_ti_skeleton.append(key_ti_out.cpu().detach().numpy()[i])
                gallery_feature.append(output1.cpu().detach().numpy()[i])
                gallery_feature_kp.append(key_rgb_out2.cpu().detach().numpy()[i])
                # gallery_feature.append(rgb_h.cpu().detach().numpy()[i])
                # gallery_feature.append(rgb_l.cpu().detach().numpy()[i])
                gallery_label.append(np.asarray(data[3])[i])
                # query_feature.append(output2.cpu().detach().numpy()[i])
                query_feature.append(output2.cpu().detach().numpy()[i])
                query_feature_kp.append(key_ti_out2.cpu().detach().numpy()[i])
                # query_feature.append(ti_h.cpu().detach().numpy()[i])
                # query_feature.append(ti_l.cpu().detach().numpy()[i])
                query_label.append(np.asarray(data[4])[i])

            # print("train once")

        training_loss = np.mean(training_loss)
        training_idloss = np.mean(training_idloss)
        loss_total += training_loss

        ap1 = 0.0
        ap1_kp = 0.0
        CMC1 = torch.IntTensor(len(gallery_label)).zero_()
        CMC1_kp = torch.IntTensor(len(gallery_label)).zero_()
        #print("query_label:",len(query_label))
        # 测试标准
        for i in range(len(query_label)):
            #print("query_label:",len(query_label))
            #print("i:",i)
            ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
            # print(CMC_tmp[0])
            # print(query_feature[i])
            # print(gallery_feature[i])

            if CMC_tmp[0] == -1:
                continue
            #if query_label[i] < 15 or query_label[i] > 20:
            CMC1 = CMC1 + CMC_tmp
            ap1 += ap_tmp
            flag_single = flag_single + 1

        CMC1 = CMC1.float()
        CMC1 = CMC1 / flag_single  # average CMC
        #print("query_label:", flag_single)
        CMC1 = np.asarray(CMC1)

        #CMC_kp
        for i in range(len(query_label)):
            ap_tmp_kp, CMC_tmp_kp = evaluate(query_feature_kp[i], query_label[i], gallery_feature_kp, gallery_label)
            if CMC_tmp_kp[0] == -1:
                continue
            CMC1_kp = CMC1_kp + CMC_tmp_kp
            ap1_kp += ap_tmp_kp
            flag_single_kp = flag_single_kp + 1

        CMC1_kp = CMC1_kp.float()
        CMC1_kp = CMC1_kp / flag_single_kp  # average CMC

        CMC1_kp = np.asarray(CMC1_kp)

        writer1.add_scalars('train_Rank-n_single',
                            {'rank1': torch.tensor(CMC1[0], dtype=float),
                             'rank2': torch.tensor(CMC1[1]),
                             'rank3': torch.tensor(CMC1[2]),
                             'rank4': torch.tensor(CMC1[3]),
                             'rank5': torch.tensor(CMC1[4]),
                             'rank6': torch.tensor(CMC1[5]),
                             'rank7': torch.tensor(CMC1[6]),
                             'rank8': torch.tensor(CMC1[7]),
                             'rank9': torch.tensor(CMC1[8]),
                             'rank10': torch.tensor(CMC1[9]),
                             'rank11': torch.tensor(CMC1[10]),
                             'rank12': torch.tensor(CMC1[11]),
                             'rank13': torch.tensor(CMC1[12]),
                             'rank14': torch.tensor(CMC1[13]),
                             'rank15': torch.tensor(CMC1[14])
                             }, epoch)

        writer1.add_scalar(tag='train_mAP_single', scalar_value=ap1 / flag_single, global_step=epoch)
        writer1.add_scalars('train_keypre',
                           {   'train_keypoint_rgb': np.mean(training_loss_s1_mpjpe),
                               'train_keypoint_ti1': np.mean(training_loss_s2_mpjpe),
                               'train_keypoint_ti2': np.mean(training_loss_s3_mpjpe),
                           }, epoch)

        writer1.add_scalars('train_keypre_self',
                            {'train_keypoint_rgb_self': np.mean(training_loss_s1_mpjpe_self),
                             'train_keypoint_ti_self': np.mean(training_loss_s2_mpjpe_self),
                             }, epoch)
        writer1.add_scalar('train_loc',np.mean(training_loss_loc),epoch)


        writer1.add_scalar(tag='train_loss', scalar_value=training_loss, global_step=epoch)
        #writer1.add_scalar(tag='train_idloss', scalar_value=training_idloss, global_step=epoch)
        #writer1.add_scalar(tag='train_mAP_multi', scalar_value=ap2 / flag_multi, global_step=epoch)

        if (epoch + 1) % 5 == 0:
            print('train_loss mean after {} epochs: {}'.format((epoch + 1), loss_total / 5))
            print('train_top1: {}'.format(CMC1[0]))
            print('train_top5: {}'.format(CMC1[4]))
            print('train_mAP: {}'.format(ap1 / len(query_label)))
            loss_total = 0.



        # 更新最优解
        top1_best = max(top1_best, CMC1[0])
        top5_best = max(top1_best, CMC1[4])
        top10_best = max(top1_best, CMC1[9])
        kprgb_best = min(kprgb_best, np.mean(training_loss_s1_mpjpe))
        kpti1_best = min(kpti1_best, np.mean(training_loss_s2_mpjpe))
        kpti2_best = min(kpti2_best, np.mean(training_loss_s3_mpjpe))

    if (epoch+1)%20 != 0 and ifrunonce==0:
        continue

    #测试集
    model.eval()
    eval_loss = []
    eval_idloss = []
    eval_loss_s1 = []
    eval_loss_s2 = []
    eval_loss_s3 = []
    eval_loss_s1_self = []
    eval_loss_s2_self = []
    loss_loc_p1 = []
    loss_loc_p2 = []
    loss_loc_n1 = []
    loss_loc_n2 = []
    eval_loss_loc = []
    eval_loss_frame1 = []
    eval_loss_frame2 = []
    eval_loss_frame3 = []
    eval_loss_t_rgb = []
    eval_loss_t_ti = []
    eval_loss_t_ti2 = []
    eval_loss_tgloc_ti = []
    gallery_feature = []
    gallery_feature_l = []
    gallery_feature_h = []
    query_feature = []
    query_feature_l = []
    query_feature_h = []
    gallery_feature_kp = []
    query_feature_kp = []
    gallery_label = []
    query_label = []
    gallery_label_l = []
    query_label_l = []
    query_feature_kp_singleti = []
    gallery_feature_kp_singleti = []
    query_feature_singleti = []
    gallery_feature_singleti = []
    gallery_label_singleti = []
    query_label_singleti = []
    eval_rgb_gt = []
    eval_ti_gt = []
    eval_rgb_skeleton = []
    eval_ti_skeleton = []
    flag_single = 0
    flag_single_l = 0
    flag_single_h = 0
    flag_single_kp = 0
    flag_single_singleti = 0
    flag_single_kp_singleti = 0
    flag_multi = 0
    correct_ti =0
    correct_rgb = 0
    eval_accu_flag = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            data = np.asarray(data)

            batch_size, seq_len, point_num, dim = data[1][:, length_size_total - length_size:, :].shape
            data_ti_in = np.reshape(data[1][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_ti = torch.tensor(data_ti_in, dtype=torch.float32, device=device).squeeze()

            batch_size, seq_len, point_num, dim = data[2][:, length_size_total - length_size:, :].shape
            data_ti_in2 = np.reshape(data[2][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_ti2 = torch.tensor(data_ti_in2, dtype=torch.float32, device=device).squeeze()

            data_ti_rand = torch.rand(batch_size*seq_len,64,3,device=device)

            batch_size, seq_len, dim, x, y = data[0][:, length_size_total - length_size:, :].shape
            data_rgb_in = np.reshape(data[0][:, length_size_total - length_size:, :, :, :], (batch_size * seq_len, dim, x, y))
            data_rgb = torch.tensor(data_rgb_in, dtype=torch.float32, device=device)

            batch_size, seq_len, dim, x, y = data[9][:, length_size_total - length_size:, :].shape
            data_rgb_in2 = np.reshape(data[9][:, length_size_total - length_size:, :, :, :], (batch_size * seq_len, dim, x, y))
            data_rgb2 = torch.tensor(data_rgb_in2, dtype=torch.float32, device=device)

            batch_size, seq_len, point_num, dim = data[6][:, length_size_total - length_size:, :].shape
            data_key_in = np.reshape(data[6][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key = torch.tensor(data_key_in, dtype=torch.float32, device=device)

            data_key_in2 = np.reshape(data[7][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key2 = torch.tensor(data_key_in2, dtype=torch.float32, device=device)

            data_key_in3 = np.reshape(data[8][:, length_size_total - length_size:, :], (batch_size * seq_len, point_num, dim))
            data_key3 = torch.tensor(data_key_in3, dtype=torch.float32, device=device)

            data[3] = np.asarray(data[3], dtype=int)
            data[4] = np.asarray(data[4], dtype=int)
            data[5] = np.asarray(data[5], dtype=int)

            h0 = torch.zeros((6, batchsize2, 64), dtype=torch.float32, device=device)
            c0 = torch.zeros((6, batchsize2, 64), dtype=torch.float32, device=device)

            h1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
            c1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)



            rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2, key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self2,rgb_l2,rgb_l_dif,ti_dif = \
            model(data_rgb,data_rgb2, data_ti, data_ti2, h0, c0, batchsize, length_size)




            loss_s1_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1) - torch.cat((data_key[:, :10], data_key[:, 12:22]),
                                                                                   dim=1)), dim=-1)), dim=0)
            loss_s2_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)

            loss_s1 = torch.mean(loss_s1_part)
            loss_s2 = torch.mean(loss_s2_part)
            loss_s3 = torch.mean(loss_s3_part)
            eval_loss_s1.append(loss_s1.item())
            eval_loss_s2.append(loss_s2.item())
            eval_loss_s3.append(loss_s3.item())
            # loc-loss
            loss_loc_p1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p1 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            #print("loc_gt:",data_key2[0, 0])
            loss_loc_p2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p2 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n1 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n2 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc = (loss_loc_p1 + loss_loc_p2 + loss_loc_n1 + loss_loc_n2) / 4
            eval_loss_loc.append(loss_loc.item())

            # self attention结果监督
            loss_s1_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb_self[:, :10], key_pre_rgb_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s2_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self[:, :10], key_pre_ti_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s1_mpjpe_self = torch.mean(loss_s1_part_self)
            loss_s2_mpjpe_self = torch.mean(loss_s2_part_self)

            eval_loss_s1_self.append(loss_s1_mpjpe_self.item())
            eval_loss_s2_self.append(loss_s2_mpjpe_self.item())

            eval_loss_frame1.append(loss_s1_part.cpu().detach().numpy())
            eval_loss_frame2.append(loss_s2_part.cpu().detach().numpy())
            eval_loss_frame3.append(loss_s3_part.cpu().detach().numpy())


            loss_keypoint = 0.5*(loss_s1 + loss_s2 + loss_s3) / 3 #+ 0.5*loss_id
            eval_loss.append(loss_keypoint.item())

            key_rgb_out = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1).view(batch_size,length_size,-1)
            key_ti_out = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1).view(batch_size,length_size, -1)
            key_ti2_out =torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1).view(batch_size,length_size,-1)

            # print("key_ti_out1:", key_ti_out.shape)
            key_rgb_out2 = torch.flatten(key_rgb_out, start_dim=1, end_dim=2)
            key_ti_out2 = torch.flatten(key_ti_out, start_dim=1, end_dim=2)
            key_ti2_out2 = torch.flatten(key_ti2_out, start_dim=1, end_dim=2)

            show_rgb = key_pre_rgb.cpu().detach()
            show_ti = key_pre_ti.cpu().detach()
            show_gt_rgb = data_key.cpu().detach()
            show_gt_ti = data_key2.cpu().detach()
            draw3Dpose_frames(show_rgb,show_gt_rgb)
            #print(rgb_l==ti_l)
            for i in range(batchsize2):
                eval_rgb_gt.append(data[6][:, length_size_total - length_size:, :].cpu().detach().numpy()[i])
                eval_ti_gt.append(data[7][:, length_size_total - length_size:, :].cpu().detach().numpy()[i])
                eval_rgb_skeleton.append(key_rgb_out.cpu().detach().numpy()[i])
                eval_ti_skeleton.append(key_ti_out.cpu().detach().numpy()[i])
                gallery_feature.append(output1.cpu().detach().numpy()[i])
                gallery_feature_kp.append(key_rgb_out2.cpu().detach().numpy()[i])
                gallery_feature_l.append(rgb_l.cpu().detach().numpy()[i])
                gallery_label.append(np.asarray(data[3])[i])
                gallery_label_l.append(np.asarray(data[3])[i])
                #全排列
                #gallery_feature.append(output4.cpu().detach().numpy()[i])
                #gallery_feature_h.append(rgb_h2.cpu().detach().numpy()[i])

                if ifallfusion == 1:
                    # rgb_l_dif = ti_p+rgb_n;ti_dif = ti_p+rgb_n
                    # anchor:rgb_l;p:ti_l;n:ti_dif
                    gallery_feature_l.append(rgb_l_dif.cpu().detach().numpy()[i])
                    gallery_label_l.append(np.asarray(data[5])[i])
                    query_feature_l.append(ti_dif.cpu().detach().numpy()[i])
                    query_label_l.append(np.asarray(data[3])[i])

                query_feature.append(output2.cpu().detach().numpy()[i])
                query_feature_l.append(ti_l.cpu().detach().numpy()[i])
                query_feature_kp.append(key_ti_out2.cpu().detach().numpy()[i])
                query_label.append(np.asarray(data[4])[i])
                query_label_l.append(np.asarray(data[4])[i])




            # print("train once")
    eval_loss = np.mean(eval_loss)

    eval_loss_total += eval_loss


    # loss_total += loss.data.cpu().numpy()
    ap1 = 0.0
    ap1_kp = 0.0
    ap1_l = 0.0
    ap1_kp_singleti = 0.0
    ap1_singleti = 0.0
    CMC1 = torch.IntTensor(len(gallery_label)-1).zero_()
    CMC1_l = torch.IntTensor(len(gallery_label_l)-1).zero_()
    CMC1_kp = torch.IntTensor(len(gallery_label)-1).zero_()
    #CMC1_kp = torch.IntTensor(len(gallery_label)).zero_()
    #CMC1_kp_singleti = torch.IntTensor(len(gallery_label_singleti)-1).zero_()
    #CMC1_singleti = torch.IntTensor(len(gallery_label_singleti)-1).zero_()
    # 测试标准
    for i in range(len(query_label)):
        # print("query_label:",len(query_label))
        # print("i:",i)

        clean_gallery_feature = np.delete(gallery_feature, i, axis=0)
        clean_gallery_label = np.delete(gallery_label, i, axis=0)
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], clean_gallery_feature, clean_gallery_label)

        # print(CMC_tmp[0])
        # print(query_feature[i])
        # print(gallery_feature[i])

        if CMC_tmp[0] == -1:
            continue
        # if query_label[i] < 15 or query_label[i] > 20:
        CMC1 = CMC1 + CMC_tmp
        ap1 += ap_tmp
        flag_single = flag_single + 1

    CMC1 = CMC1.float()
    CMC1 = CMC1 / flag_single  # average CMC

    CMC1 = np.asarray(CMC1)
    flag_single_l1 = 0
    flag_single_l2 = 0
    for i in range(len(query_label_l)//2):
        clean_gallery_feature_l = np.delete(gallery_feature_l, i*2, axis=0)
        clean_gallery_label = np.delete(gallery_label_l, i*2, axis=0)
        (ap_tmp, CMC_tmp),score,index = evaluate_l(query_feature_l[i*2], query_label_l[i*2], clean_gallery_feature_l, clean_gallery_label)

        clean_gallery_feature_l = np.delete(gallery_feature_l, i*2+1, axis=0)
        clean_gallery_label = np.delete(gallery_label_l, i*2+1, axis=0)
        (ap_tmp2, CMC_tmp2), score2,index2 = evaluate_l(query_feature_l[i*2+1], query_label_l[i*2+1], clean_gallery_feature_l,
                                              clean_gallery_label)

        #print(score[index[0]])

        if CMC_tmp[0] == -1 or  CMC_tmp2[0] == -1:
            continue
        # if query_label[i] < 15 or query_label[i] > 20:
        if score[index[0]]< score2[index2[0]]:
            CMC1_l = CMC1_l + CMC_tmp
            ap1_l += ap_tmp
            flag_single_l = flag_single_l + 1
            flag_single_l1 = flag_single_l1 + 1
            #print(CMC_tmp)
        else:
            CMC1_l = CMC1_l + CMC_tmp2
            ap1_l += ap_tmp2
            flag_single_l = flag_single_l + 1
            flag_single_l2 = flag_single_l2 + 1
            #print(CMC_tmp)

    CMC1_l = CMC1_l.float()
    CMC1_l = CMC1_l / flag_single_l  # average CMC

    CMC1_l = np.asarray(CMC1_l)



    # CMC_kp
    for i in range(len(query_label)):
        clean_gallery_feature_kp = np.delete(gallery_feature_kp, i, axis=0)
        clean_gallery_label = np.delete(gallery_label, i, axis=0)
        ap_tmp_kp, CMC_tmp_kp = evaluate(query_feature_kp[i], query_label[i], clean_gallery_feature_kp, clean_gallery_label)
        # print(CMC_tmp[0])
        # print(query_feature[i])
        # print(gallery_feature[i])

        if CMC_tmp_kp[0] == -1:
            continue
        # if query_label[i] < 15 or query_label[i] > 20:
        CMC1_kp = CMC1_kp + CMC_tmp_kp
        ap1_kp += ap_tmp_kp
        flag_single_kp = flag_single_kp + 1

    CMC1_kp = CMC1_kp.float()
    CMC1_kp = CMC1_kp / flag_single_kp  # average CMC

    CMC1_kp = np.asarray(CMC1_kp)

    if ifrunonce==1:
        print('eval_top1: {}'.format(CMC1[0]))
        print('eval_top5: {}'.format(CMC1[4]))
        print('eval_mAP: {}'.format(ap1 / flag_single))
        print('eval_top1_l: {}'.format(CMC1_l[0]))
        print('eval_top5_l: {}'.format(CMC1_l[4]))
        print('eval_mAP_single_l: {}'.format(ap1_l / flag_single_l))
        print('eval_top1_kp: {}'.format(CMC1_kp[0]))
        print('eval_top5_kp: {}'.format(CMC1_kp[4]))
        print('eval_mAP_single_kp: {}'.format(ap1_kp / flag_single_kp))
        print('eval_keypoint_rgb: {}'.format(np.mean(eval_loss_s1)))
        print('eval_keypoint_ti1: {}'.format(np.mean(eval_loss_s2)))
        print('eval_keypoint_ti2: {}'.format(np.mean(eval_loss_s3)))
        print('eval_keypoint_loc: {}'.format(np.mean(eval_loss_loc)))
        print('eval_loss_tgloc_ti: {}'.format(np.mean(eval_loss_tgloc_ti)))
        print("flag_single_l1:",flag_single_l1)
        print("flag_single_l2:",flag_single_l2)
        if self_atten == 1:
            print('eval_keypoint_rgb_self: {}'.format(np.mean(eval_loss_s1_self)))
            print('eval_keypoint_ti1_self: {}'.format(np.mean(eval_loss_s2_self)))

        np.save("./res/pred_skeletn_gt_divided/train_rgb_skeleton.npy",train_rgb_skeleton)
        np.save("./res/pred_skeletn_gt_divided/train_ti_skeleton.npy", train_ti_skeleton)
        np.save("./res/pred_skeletn_gt_divided/train_rgb_gt.npy", train_rgb_gt)
        np.save("./res/pred_skeletn_gt_divided/train_ti_gt.npy", train_ti_gt)

        np.save("./res/pred_skeletn_gt_divided/eval_rgb_gt.npy", eval_rgb_gt)
        np.save("./res/pred_skeletn_gt_divided/eval_ti_gt.npy", eval_ti_gt)
        np.save("./res/pred_skeletn_gt_divided/eval_rgb_skeleton.npy", eval_rgb_skeleton)
        np.save("./res/pred_skeletn_gt_divided/eval_ti_skeleton.npy", eval_ti_skeleton)
        np.save("./res/pred_skeletn_gt_divided/eval_label.npy", gallery_label)

        if ifst==1:
            print('eval_keypoint_rgb_t: {}'.format(np.mean(eval_loss_t_rgb)))
            print('eval_keypoint_ti1_t: {}'.format(np.mean(eval_loss_t_ti)))
            print('eval_keypoint_ti2_t: {}'.format(np.mean(eval_loss_t_ti2)))

        # 只保存最优解
    if epoch > 200 and CMC1_l[0] > top1_best_test_l:
        save_models("best")
        # 更新最优解
    top1_best_test = max(top1_best_test, CMC1[0])
    top5_best_test = max(top5_best_test, CMC1[4])
    top10_best_test = max(top10_best_test, CMC1[9])
    map_best_test = max(map_best_test, ap1 / flag_single)
    top1_best_test_l = max(top1_best_test_l, CMC1_l[0])
    top5_best_test_l = max(top5_best_test_l, CMC1_l[4])
    top10_best_test_l = max(top10_best_test_l, CMC1_l[9])
    map_best_test_l = max(map_best_test_l, ap1_l / flag_single_l)


    kprgb_best_test = min(kprgb_best_test, np.mean(eval_loss_s1))
    kpti1_best_test = min(kpti1_best_test, np.mean(eval_loss_s2))
    kpti2_best_test = min(kpti2_best_test, np.mean(eval_loss_s3))

    if iftrain == 1:
        writer1.add_scalars('eval_keypre',
                            {'eval_keypoint_rgb': np.mean(eval_loss_s1),
                             'eval_keypoint_ti1': np.mean(eval_loss_s2),
                             'eval_keypoint_ti2': np.mean(eval_loss_s3),
                             }, epoch)
        writer1.add_scalar('eval_loc', np.mean(eval_loss_loc), epoch)
        writer1.add_scalars('eval_keypre_t',
                            {'rgb': np.mean(eval_loss_t_rgb),
                             'ti1': np.mean(eval_loss_t_ti),
                             'ti2': np.mean(eval_loss_t_ti2),
                             }, epoch)
        writer1.add_scalars('eval_Rank-n_single',
                            {'rank1': torch.tensor(CMC1[0], dtype=float),
                             'rank2': torch.tensor(CMC1[1]),
                             'rank3': torch.tensor(CMC1[2]),
                             'rank4': torch.tensor(CMC1[3]),
                             'rank5': torch.tensor(CMC1[4]),
                             'rank6': torch.tensor(CMC1[5]),
                             'rank7': torch.tensor(CMC1[6]),
                             'rank8': torch.tensor(CMC1[7]),
                             'rank9': torch.tensor(CMC1[8]),
                             'rank10': torch.tensor(CMC1[9]),
                             'rank11': torch.tensor(CMC1[10]),
                             'rank12': torch.tensor(CMC1[11]),
                             'rank13': torch.tensor(CMC1[12]),
                             'rank14': torch.tensor(CMC1[13]),
                             'rank15': torch.tensor(CMC1[14])
                             }, epoch)

        writer1.add_scalars('eval_Rank-n_single_l',
                            {'rank1': torch.tensor(CMC1_l[0], dtype=float),
                             'rank2': torch.tensor(CMC1_l[1]),
                             'rank3': torch.tensor(CMC1_l[2]),
                             'rank4': torch.tensor(CMC1_l[3]),
                             'rank5': torch.tensor(CMC1_l[4]),
                             'rank6': torch.tensor(CMC1_l[5]),
                             'rank7': torch.tensor(CMC1_l[6]),
                             'rank8': torch.tensor(CMC1_l[7]),
                             'rank9': torch.tensor(CMC1_l[8]),
                             'rank10': torch.tensor(CMC1_l[9]),
                             'rank11': torch.tensor(CMC1_l[10]),
                             'rank12': torch.tensor(CMC1_l[11]),
                             'rank13': torch.tensor(CMC1_l[12]),
                             'rank14': torch.tensor(CMC1_l[13]),
                             'rank15': torch.tensor(CMC1_l[14])
                             }, epoch)

        scipy.io.savemat(matname,
                         mdict={'top1_best_test': top1_best_test, 'top5_best_test': top5_best_test,
                                'top10_best_test': top10_best_test, 'map_best_test': map_best_test,
                                'top1_best_test_l': top1_best_test_l, 'top5_best_test_l': top5_best_test_l,
                                'top10_best_test_l': top10_best_test_l, 'map_best_test_l': map_best_test_l,
                                'kprgb_best_test': kprgb_best_test,
                                'kpti1_best_test': kpti1_best_test, 'kpti2_best_test': kpti2_best_test,
                                'top1_best': top1_best, 'top5_best': top5_best,
                                'top10_best': top10_best, 'kprgb_best': kprgb_best,
                                'kpti1_best': kpti1_best, 'kpti2_best': kpti2_best,
                                })

    writer1.add_scalar(tag='eval_mAP_single', scalar_value=ap1 / flag_single, global_step=epoch)
    writer1.add_scalar(tag='eval_mAP_single_l', scalar_value=ap1_l / flag_single_l, global_step=epoch)
    writer1.add_scalar(tag='eval_idloss', scalar_value=np.mean(eval_idloss), global_step=epoch)



    #writer1.add_scalar(tag='eval_mAP_multi', scalar_value=ap2 / flag_multi, global_step=epoch)
    if (epoch + 1) % 5 == 0:
        print('eval_top1: {}'.format(CMC1[0]))
        print('eval_top5: {}'.format(CMC1[4]))
        print('eval_mAP: {}'.format(ap1 / len(query_label)))
        print('eval_top1_kp: {}'.format(CMC1_kp[0]))
        print('eval_top5_kp: {}'.format(CMC1_kp[4]))
        print('eval_mAP_kp: {}'.format(ap1_kp / len(query_label)))
        eval_loss_total = 0.
    if ifrunonce == 1:
        break



