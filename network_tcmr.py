import numpy as np
import torch
import torch.nn as nn
from smpl_utils_extend import SMPL
import Resnet
import torch.nn.functional as F
from collections import OrderedDict
import os
import scipy.io as scio
import h5py
from torch.utils.data import Dataset
from torch.autograd import Variable
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torchvision.models as models
import os.path as osp
#from lib.models.spin import Regressor
from lib.models.spin import hmr
from lib.models.spin import hmr_atten
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_crop
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_crop_targetrandom
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_crop_targetrandom_rgbgt
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_crop_targetrandom_gt2
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_full_targetrandom_gt2
from dataset_me_rgb import SiamesePC_20_nonormalization_rgb_full_gt2_kp2d
from torch.utils.tensorboard import SummaryWriter
from lib.models.loss import ContrastiveLoss
import imageio
from lib.models.spin import hmr_atten_14
#可视化
def draw3Dpose_frames_anchor(anchor,ti):
    # 绘制连贯的骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.ion()
    i = 0
    j=0
    anchor_show = anchor.cpu().detach()
    ti_show = ti.cpu().detach()
    while i < anchor.shape[0]:
        ax.view_init(0, 90)
        ax.scatter(anchor_show[i,:, 0], anchor_show[i,:, 1], anchor_show[i,:, 2], c=['green'])
        ax.scatter(ti_show[i, :, 0], ti_show[i, :, 1], ti_show[i, :, 2], c=['red'])
        plt.pause(1.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1
        if i==anchor.shape[0]:
            #i=0
            j+=1
        if j==2:
            break

    plt.ioff()
    plt.show()

#tcmr
class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            seq_len=16,
            hidden_size=2048
    ):
        super(TemporalEncoder, self).__init__()

        self.gru_cur = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers
        )
        self.gru_bef = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.gru_aft = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.mid_frame = int(seq_len/2)
        self.hidden_size = hidden_size

        self.linear_cur = nn.Linear(hidden_size * 2, 2048)
        self.linear_bef = nn.Linear(hidden_size, 2048)
        self.linear_aft = nn.Linear(hidden_size, 2048)

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        # NTF -> TNF
        #print("x:", x.shape)
        y, state = self.gru_cur(x.permute(1,0,2))  # y: Tx N x (num_dirs x hidden size)
        #print("y:", y.shape)

        x_bef = x[:, :self.mid_frame]
        x_aft = x[:, self.mid_frame+1:]
        x_aft = torch.flip(x_aft, dims=[1])
        y_bef, _ = self.gru_bef(x_bef.permute(1,0,2))
        #print("y_bef:",y_bef.shape)
        y_aft, _ = self.gru_aft(x_aft.permute(1,0,2))

        # y_*: N x 2048
        y_cur = self.linear_cur(F.relu(y[self.mid_frame]))
        y_bef = self.linear_bef(F.relu(y_bef[-1]))
        y_aft = self.linear_aft(F.relu(y_aft[-1]))
        '''
        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)

        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        #print("out",out.shape)
        out = torch.sum(out, dim=1)  # N x 2048
        '''

        scores=0
        out_y=y.permute(1,0,2)
        #print("out_y",out_y.shape)
        if not is_train:
            #return out, scores
            return out_y, scores
        else:
            y = torch.cat((y[:, 0:1], y[:, 2:], out_y[:, None, :]), dim=1)
            return y, scores

class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores

class TCMR(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            pretrained=osp.join('/lib/models/pretrained/base_data', 'spin_model_checkpoint.pth.tar'),
    ):

        super(TCMR, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = \
            TemporalEncoder(
                seq_len=seqlen,
                n_layers=n_layers,
                hidden_size=hidden_size
            )

        #regressor can predict cam, pose and shape params in an iterative way
        '''
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    '''
    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        #print("tcmr_input:", input.shape)
        feature, scores = self.encoder(input, is_train=is_train)
        #print("tcmr_f:",feature.shape)
        #feature = feature.reshape(-1, feature.size(-1))


        #不需要回归
        '''
        
        smpl_output = self.regressor(feature, is_train=is_train, J_regressor=J_regressor)

        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores

        else:
            repeat_num = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)
                s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3)
                s['scores'] = scores
        '''
        return feature, scores

def batch_orthographic_projection(kp3d, camera):
    """computes reprojected 3d to 2d keypoints
    Args:
        kp3d:   [batch x K x 3]
        camera: [batch x 3]
    Returns:
        kp2d: [batch x K x 2]
    """
    camera = torch.reshape(camera, (-1, 1, 3))
    # print("camera:", camera.shape)
    kp_trans = kp3d[:, :, :2] + camera[:, :, 1:]
    # print("camera[:, :, 1:]:",camera[:, :, 1:].shape)
    # print("kp3d[:, :, :2]:", kp3d[:, :, :2].shape)
    shape = kp_trans.shape
    # print("shape:",  kp_trans.shape)
    kp_trans = torch.reshape(kp_trans, (shape[0], -1))
    # print("kp_trans:", kp_trans.shape)
    kp2d = camera[:, :, 0] * kp_trans
    # print("kp2d:", kp2d.shape)
    return torch.reshape(kp2d, shape)

#Anchor Module
def AnchorInit(x_min=-0.3, x_max=0.3, x_interval=0.3, y_min=-0.3, y_max=0.3, y_interval=0.3, z_min=-1.2, z_max=1.2, z_interval=0.3):#[z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    x_size=round((x_max-x_min)/x_interval)+1
    y_size=round((y_max-y_min)/y_interval)+1
    z_size=round((z_max-z_min)/z_interval)+1
    device=torch.device('cuda:%d' % (0) if torch.cuda.is_available() else 'cpu')
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32).to(device)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx=x_min+x_no*x_interval
                ly=y_min+y_no*y_interval
                lz=z_min+z_no*z_interval
                centroids[z_no, y_no, x_no, 0]=lx
                centroids[z_no, y_no, x_no, 1]=ly
                centroids[z_no, y_no, x_no, 2]=lz
    return centroids

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def point_ball_set(nsample, xyz, new_xyz):
    """
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3]
        new_xyz: anchor points [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    _, sort_idx = torch.sort(sqrdists)
    sort_idx=sort_idx[:,:,:nsample]
    batch_idx=torch.arange(B, dtype=torch.long).to(device).view((B,1,1)).repeat((1,S,nsample))
    centroids_idx=torch.arange(S, dtype=torch.long).to(device).view((1,S,1)).repeat((B,1,nsample))
    return group_idx[batch_idx, centroids_idx, sort_idx]

def AnchorGrouping(anchors, nsample, xyz, points):
    """
    Input:
        anchors: [B, 9*3*3, 3], npoint=9*3*3
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    _, S, _ = anchors.shape
    idx = point_ball_set(nsample, xyz, anchors)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_anchors=anchors.view(B, S, 1, C).repeat(1,1,nsample,1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors #anchors.view(B, S, 1, C)

    grouped_points = index_points(points, idx)
    new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+C+D]
    return new_points

class AnchorPointNet(nn.Module):
    def __init__(self):
        super(AnchorPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x))) #(Batch, feature, frame_point_number)

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class AnchorVoxelNet(nn.Module):
    def __init__(self):
        super(AnchorVoxelNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(0,0,0))
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(5, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 1, 1))
        self.cb3 = nn.BatchNorm3d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size=x.size()[0]
        x=x.permute(0, 4, 1, 2, 3)

        x=self.caf1(self.cb1(self.conv1(x)))
        x=self.caf2(self.cb2(self.conv2(x)))
        x=self.caf3(self.cb3(self.conv3(x)))

        x=x.view(batch_size, 64)
        return x

class AnchorRNN(nn.Module):
    def __init__(self):
        super(AnchorRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x, (h0, c0))
        return a_vec, hn, cn

class AnchorRNN_bidirectional(nn.Module):
    def __init__(self):
        super(AnchorRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x, (h0, c0))
        return a_vec, hn, cn

class AnchorModule(nn.Module):
    def __init__(self):
        super(AnchorModule, self).__init__()
        self.template_point=AnchorInit()
        self.z_size, self.y_size, self.x_size, _=self.template_point.shape
        #print(self.template_point.shape)
        self.anchor_size=self.z_size*self.y_size*self.x_size
        self.apointnet=AnchorPointNet()
        self.avoxel=AnchorVoxelNet()
        self.arnn=AnchorRNN()

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        g_loc=g_loc.view(batch_size*length_size, 1, 2).repeat(1,self.anchor_size,1)
        anchors=self.template_point.view(1, self.anchor_size, 3).repeat(batch_size*length_size, 1, 1)
        anchors[:,:,:2]+=g_loc
        grouped_points=AnchorGrouping(anchors, nsample=8, xyz=x[..., :3], points=x[..., 3:])
        #print("self.anchor_size:",self.anchor_size)
        grouped_points=grouped_points.view(batch_size*length_size*self.anchor_size, 8, 3+feature_size)
        voxel_points, attn_weights=self.apointnet(grouped_points)
        voxel_points=voxel_points.view(batch_size*length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec=self.avoxel(voxel_points)
        voxel_vec=voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn=self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn

class AnchorModule_bidirectional(nn.Module):
    def __init__(self):
        super(AnchorModule_bidirectional, self).__init__()
        self.template_point=AnchorInit()
        self.z_size, self.y_size, self.x_size, _=self.template_point.shape
        #print(self.template_point.shape)
        self.anchor_size=self.z_size*self.y_size*self.x_size
        self.apointnet=AnchorPointNet()
        self.avoxel=AnchorVoxelNet()
        self.arnn=AnchorRNN_bidirectional()

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        g_loc=g_loc.view(batch_size*length_size, 1, 2).repeat(1,self.anchor_size,1)
        anchors=self.template_point.view(1, self.anchor_size, 3).repeat(batch_size*length_size, 1, 1)
        #print("anchors", anchors.shape)
        #print("anchors", anchors.device)
        #print("g_loc", g_loc.shape)

        anchors[:,:,:2]+=g_loc
        #draw3Dpose_frames_anchor(anchors, x)
        grouped_points=AnchorGrouping(anchors, nsample=8, xyz=x[..., :3], points=x[..., 3:])
        #print("grouped_points:",grouped_points.shape)
        grouped_points=grouped_points.view(batch_size*length_size*self.anchor_size, 8, 3+feature_size)
        voxel_points, attn_weights=self.apointnet(grouped_points)
        voxel_points=voxel_points.view(batch_size*length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec=self.avoxel(voxel_points)
        voxel_vec=voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn=self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn

#mmMesh
class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3,   out_channels=8,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)
        x = torch.cat((in_mat[:,:,:4], x), -1)

        return x

class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)
        #print("x:",x.shape)
        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        #print("attn_weights:", attn_weights.shape)
        #print("before atten:",x.shape)
        attn_vec=torch.sum(x*attn_weights, dim=1)
        #print("after atten:", attn_vec.shape)
        return attn_vec, attn_weights

class GlobalRNN(nn.Module):
    def __init__(self):
        super(GlobalRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)
        self.fc1 = nn.Linear(64, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn

class GlobalRNN_bidirectional(nn.Module):
    def __init__(self):
        super(GlobalRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(128, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn

class GlobalModule_bidirectional(nn.Module):
    def __init__(self):
        super(GlobalModule_bidirectional, self).__init__()
        self.gpointnet=GlobalPointNet()
        self.grnn=GlobalRNN_bidirectional()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn

class RgbRNN_bidirectional(nn.Module):
    def __init__(self):
        super(RgbRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=128, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(256, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn

class GlobalModule(nn.Module):
    def __init__(self):
        super(GlobalModule, self).__init__()
        self.gpointnet=GlobalPointNet()
        self.grnn=GlobalRNN()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn

class CombineModule_ti(nn.Module):
    def __init__(self):
        super(CombineModule_ti, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        #print("回归特征长度：",x.shape)
        b = x

        return b

class CombineModule_rgb(nn.Module):
    def __init__(self):
        super(CombineModule_rgb, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 9 * 6 + 3)

    def forward(self, t_vec, batch_size, length_size):
        x = self.fc1(t_vec)
        x = self.faf1(x)
        x = self.fc2(x)
        q = x[:, :, :9 * 6].reshape(batch_size * length_size * 9, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x = tmp_x.view(batch_size, length_size, 9, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 9, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 9, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)

        # translation vector
        t = x[:, :, 9 * 6:9 * 6 + 3]
        # print("回归特征长度：",x.shape)


        return q, t

class CombineModule_rgb_smpl(nn.Module):
    def __init__(self):
        super(CombineModule_rgb_smpl, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(1024,256)
        self.faf2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 9 * 6 + 3 + 10 + 1)
        self.faf3 = nn.ReLU()

    def forward(self, t_vec, batch_size, length_size):
        x = self.fc1(t_vec)
        x = self.faf1(x)
        x = self.fc2(x)
        x = self.faf2(x)
        x = self.fc3(x)
        x = self.faf3(x)
        # print("回归特征长度：",x.shape)
        q = x[:, :, :9 * 6].reshape(batch_size * length_size * 9, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x = tmp_x.view(batch_size, length_size, 9, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 9, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 9, 3, 1)

        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)

        # translation vector
        t = x[:, :, 9 * 6:9 * 6 + 3]
        # shape vector
        b = x[:, :, 9 * 6 + 3:9 * 6 + 3 + 10]
        # gender
        g = x[:, :, 9 * 6 + 3 + 10:]
        return q, t, b, g

class SMPLModule(nn.Module):
    def __init__(self):
        super(SMPLModule, self).__init__()
        # gupid更改！
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.blank_atom = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, requires_grad=False,
                                       device=self.device)
        self.smpl_model_m = SMPL('m')
        self.smpl_model_f = SMPL('f')

        #self.smpl_model_n = SMPL('n')
    def forward(self, q, t, b):  # b: (10,)
        batch_size = q.size()[0]
        length_size = q.size()[1]
        q = q.view(batch_size * length_size, 9, 3, 3)
        t = t.view(batch_size * length_size, 1, 3)
        #print(t.shape)
        b = b.view(batch_size * length_size, 10)
        #g = g.view(batch_size * length_size)
        q_blank = self.blank_atom.repeat(batch_size * length_size, 1, 1, 1)
        pose = torch.cat((q_blank,
                          q[:, 1:3, :, :],
                          q_blank,
                          q[:, 3:5, :, :],
                          q_blank.repeat(1, 10, 1, 1),
                          q[:, 5:9, :, :],
                          q_blank.repeat(1, 4, 1, 1)), 1)
        rotmat = q[:, 0, :, :]


        smpl_vertice = torch.zeros((batch_size * length_size, 6890, 3), dtype=torch.float32, requires_grad=False,
                                   device=self.device)
        smpl_skeleton = torch.zeros((batch_size * length_size, 24, 3), dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        smpl_vertice, smpl_skeleton = self.smpl_model_f(b, pose,torch.zeros((batch_size* length_size, 3),
                                                                                dtype=torch.float32,
                                                                                requires_grad=False,
                                                                                device=self.device))
        #加入了平移回归
        #smpl_vertice, smpl_skeleton = self.smpl_model_f(b, pose,t)
        smpl_vertice = torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_vertice, 1, 2)), 1, 2) + t
        smpl_skeleton = torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_skeleton, 1, 2)), 1, 2) + t
        smpl_vertice = smpl_vertice.view(batch_size, length_size, 6890, 3)
        smpl_skeleton = smpl_skeleton.view(batch_size, length_size, 24, 3)
        return smpl_vertice, smpl_skeleton

class CombineModule_nosmpl_bidirectional(nn.Module):
    def __init__(self):
        super(CombineModule_nosmpl_bidirectional, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3  )

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)
        #print("key_pre:",key_pre.shape)
        # translation vector

        return key_pre

class CombineModule_tcmr(nn.Module):
    def __init__(self):
        super(CombineModule_tcmr, self).__init__()
        self.fc1 = nn.Linear(256+256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3 )

    def forward(self, g_vec, a_vec,t_vec, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        #新加入normalization策略
        ti_vec = torch.cat((g_vec, a_vec),-1)
        #ti_vec = F.normalize(ti_vec)
        #t_vec = F.normalize(t_vec)

        x = torch.cat((ti_vec,t_vec), -1)
        x = F.normalize(x)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_tcmr_lime(nn.Module):
    def __init__(self):
        super(CombineModule_tcmr_lime, self).__init__()
        self.fc1 = nn.Linear(256+256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        print("x_in:",x.shape)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        print("x_out:", x.shape)
        key_pre=x

        return key_pre

class CombineModule_tcmr_kp2d(nn.Module):
    def __init__(self):
        super(CombineModule_tcmr_kp2d, self).__init__()
        self.fc1 = nn.Linear(256+256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3+3 )

    def forward(self, g_vec, a_vec,t_vec, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = torch.cat((g_vec, a_vec,t_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size*length_size, 24, 3)
        cams = x[:, :, 24 * 3:].view(batch_size*length_size, 3)
        joints_2d = batch_orthographic_projection(key_pre, cams)
        key_pre = x[:, :, :24 * 3].view(batch_size , length_size, 24, 3)
        cams = x[:, :, 24 * 3:].view(batch_size , length_size, 3)
        joints_2d = joints_2d.view(batch_size , length_size, 24,2)
        joints_2d_crop = torch.cat((joints_2d[:,:,1:3, :], joints_2d[:,:,4:6, :],joints_2d[:,:,7:9 :],
                                    joints_2d[:,:,10:13, :],joints_2d[:,:,15:22, :]),dim=2)
        return key_pre,joints_2d_crop

class CombineModule_tcmr_single_rgb(nn.Module):
    def __init__(self):
        super(CombineModule_tcmr_single_rgb, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3 )

    def forward(self, t_vec, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        #x = torch.cat((g_vec, a_vec,t_vec), -1)
        x = self.fc1(t_vec)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_tcmr_single_rgb_kp2d(nn.Module):
    def __init__(self):
        super(CombineModule_tcmr_single_rgb_kp2d, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3+3 )

    def forward(self, t_vec, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        #x = torch.cat((g_vec, a_vec,t_vec), -1)
        x = self.fc1(t_vec)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre = x[:, :, :24 * 3].view(batch_size * length_size, 24, 3)
        cams = x[:, :, 24 * 3:].view(batch_size * length_size, 3)
        joints_2d = batch_orthographic_projection(key_pre, cams)
        key_pre = x[:, :, :24 * 3].view(batch_size, length_size, 24, 3)
        cams = x[:, :, 24 * 3:].view(batch_size, length_size, 3)
        joints_2d = joints_2d.view(batch_size, length_size, 24, 2)
        joints_2d_crop = torch.cat((joints_2d[:, :, 1:3, :], joints_2d[:, :, 4:6, :], joints_2d[:, :, 7:9:],
                                    joints_2d[:, :, 10:13, :], joints_2d[:, :, 15:22, :]), dim=2)

        return key_pre,joints_2d_crop


#rgb网络
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w), *args, **kwargs)
        input_size = x.size()
        output_size = [n, s] + [*input_size[1:]]
        return x.view(*output_size)

class RgbModule(nn.Module):
    def __init__(self):
        super(RgbModule, self).__init__()
        self.Backbone = nn.Sequential(OrderedDict([
            ('BC1', BasicConv2d(3, 32, 7, 1, 2)),
            # ('cb1', nn.BatchNorm2d(32)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('BC2', BasicConv2d(32, 32, 7, 1, 2)),
            # ('cb2', nn.BatchNorm2d(32)),
            ('relu2', nn.LeakyReLU(inplace=True)),
            ('M1', nn.MaxPool2d(kernel_size=7, stride=3)),
            ('FC1', FocalConv2d(32, 64, kernel_size=3, stride=1, padding=1, halving=2)),
            # ('cb3', nn.BatchNorm2d(64)),
            ('relu3', nn.LeakyReLU(inplace=True)),
            ('FC2', FocalConv2d(64, 64, kernel_size=3, stride=1, padding=1, halving=2)),
            # ('cb4', nn.BatchNorm2d(64)),
            ('relu4', nn.LeakyReLU(inplace=True)),
            ('M2', nn.MaxPool2d(kernel_size=7, stride=3))
        ]))
        self.avgpool = nn.AvgPool2d(9, stride=1)
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.fc1 = nn.Linear(3200, 128)
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1,
                           bidirectional=True)

    def forward(self, in_mat):
        #in_mat:(7*13, 50, 6)=(91, 50, 6)
        #in_mat:(25, 64, 6)
        x = self.Backbone(in_mat)
        batch_size=x.size(0)
        lenth = x.size(1)
        x = x.view(lenth*batch_size, x.size(2), x.size(3), x.size(4))
        x = self.avgpool(x)
        x=x.view(batch_size,lenth,-1)
        x=self.fc1(x)
        g_vec, (hn, cn) = self.rnn(x, (h0, c0))


        return g_vec

class mmWaveModel_ti_Anchor_nosmpl_bidirectional(nn.Module):
    def __init__(self):
        super(mmWaveModel_ti_Anchor_nosmpl_bidirectional, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_nosmpl_bidirectional()

    def forward(self, x,h0, c0, batch_size,length_size):
       # print( x.size())
        out_feature_size = 24 + 3

        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0, c0, batch_size, length_size,
                                                         out_feature_size)


        #print("a_vec:", a_vec)
        key_pre = self.module3(g_vec, a_vec, batch_size, length_size)

        g_loc = g_loc.view(batch_size * length_size, -1)

        #print("g_loc:", g_loc.shape)
        #v=F.normalize(v[0][0][:][0:1])
        return g_vec,a_vec,key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc(nn.Module):
    def __init__(self):
        super(mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_nosmpl_bidirectional()

    def forward(self, x,h0, c0, batch_size,length_size):
        print( "x:",x.size())
        out_feature_size = 24 + 3

        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0, c0, batch_size, length_size,
                                                         out_feature_size)

        key_pre = self.module3(g_vec, a_vec, batch_size, length_size)

        g_loc = g_loc.view(batch_size*length_size,-1)
        #print("g_loc:", g_loc.shape)
        #v=F.normalize(v[0][0][:][0:1])
        return g_vec,a_vec,key_pre,g_loc

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, g_vec, a_vec, f_tcmr,batch_size,length_size):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr
        f_tcmr = f_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre = self.module3(g_vec, a_vec,t_vec, batch_size, length_size)

        return key_pre,t_vec

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr_kp2d(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr_kp2d, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr_kp2d()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, g_vec, a_vec, f_tcmr,batch_size,length_size):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr
        f_tcmr = f_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre,kp2d = self.module3(g_vec, a_vec,t_vec, batch_size, length_size)

        return key_pre,kp2d

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_hmr_notrain_single_rgb_kp2d(nn.Module):
    def __init__(self):
        super(mmWaveModel_hmr_notrain_single_rgb_kp2d, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr_kp2d()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)
        self.module3 = CombineModule_tcmr_single_rgb_kp2d()

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, g_vec, a_vec, f_tcmr, batch_size, length_size):
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre, kp2d = self.module3( t_vec, batch_size, length_size)

        g_vec=0
        a_vec=0

        return key_pre, kp2d

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr_train(nn.Module): 
    def __init__(self):
        super(mmWaveModel_tcmr_train, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr()
        self.module4 = RgbRNN_bidirectional()
        self.fc1 = nn.Linear(2048, 128)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=128, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(128)
        self.caf1 = nn.ReLU()

    def forward(self, x,f_hmr,h0, c0,h0_a, c0_a,h0_r, c0_r, batch_size,length_size):
        #print(x_ti.size())
        pt_size = x.size()[2]
        in_feature_size = x.size()[3]
        out_feature_size = 24 + 3

        x = x.view(batch_size * length_size, pt_size, in_feature_size)
        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0_a, c0_a, batch_size, length_size,
                                                         out_feature_size)
        f_hmr = f_hmr.transpose(1, 2)
        f_hmr = self.caf1(self.cb1(self.conv1(f_hmr)))
        f_hmr = f_hmr.transpose(1, 2)
        #f_hmr=self.fc1(f_hmr)
        rgb_vec, g_loc, hn, cn = self.module4(f_hmr, h0_r, c0_r)



        #print("g_vec:", a_vec.shape)
        #print("a_vec:", a_vec)
        key_pre = self.module3(g_vec, a_vec,rgb_vec, batch_size, length_size)


        #v=F.normalize(v[0][0][:][0:1])
        return g_vec,a_vec,key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_rgb(nn.Module):
    def __init__(self):
        super(mmWaveModel_rgb, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr()
        self.module4 = SMPLModule()
        self.module5 = RgbModule()

    def forward(self,  x, x_bit,h0, c0,h0_a, c0_a,batch_size,length_size):
        #多模态
        pt_size = x.size()[2]
        in_feature_size = x.size()[3]
        out_feature_size = 24 + 3

        x = x.view(batch_size * length_size, pt_size, in_feature_size)
        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0_a, c0_a, batch_size, length_size,
                                                         out_feature_size)
        r_vec=self.module5(x_bit)
        key_pre = self.module3(g_vec, a_vec,r_vec, batch_size, length_size)

        return key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_hmr_train(nn.Module):
    def __init__(self,device2):
        super(mmWaveModel_hmr_train, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr()
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional()
        '''
        self.model_ti.load(
            '.\log\Backbone\Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_tcmr()

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,x,h0, c0,h0_a, c0_a, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # print("feature_tcmr",feature_tcmr.shape)
        g_vec, a_vec, _ = self.model_ti(x, h0, c0, batch_size, length_size)

        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre = self.module3(g_vec, a_vec, t_vec, batch_size, length_size)
        ti_vec = torch.cat((g_vec, a_vec), -1)
        ti_vec = F.normalize(ti_vec)
        t_vec = F.normalize(t_vec)

        return ti_vec,t_vec,key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_hmr_train_single_rgb(nn.Module):
    def __init__(self):
        super(mmWaveModel_hmr_train_single_rgb, self).__init__()
        BASE_DATA_DIR = '/lib/models/pretrained/base_data'
        self.model_hmr = hmr()
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional()
        self.model_ti.load(
            '.\log\Backbone\Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device)
        # print(model)
        pretrained_file = '/lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_tcmr_single_rgb()

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x,x_rgb,h0, c0,h0_a, c0_a,h0_r, c0_r, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, seq_len, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # print("feature_tcmr",feature_tcmr.shape)
        #g_vec, a_vec, _ = self.model_ti(x, h0, c0, h0_a, c0_a, batchsize, length_size)

        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre = self.module3( t_vec, batch_size, length_size)

        g_vec=0
        a_vec=0

        return g_vec,a_vec,key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_hmr_notrain_single_rgb(nn.Module):
    def __init__(self):
        super(mmWaveModel_hmr_notrain_single_rgb, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr()
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.model_hmr.eval()

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional()
        self.model_ti.load(
            '.\log\Backbone\Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)
        self.model_tcmr.eval()

        self.module3 = CombineModule_tcmr_single_rgb()

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x,x_rgb,h0, c0,h0_a, c0_a,h0_r, c0_r, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, seq_len, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # print("feature_tcmr",feature_tcmr.shape)
        #g_vec, a_vec, _ = self.model_ti(x, h0, c0, h0_a, c0_a, batchsize, length_size)

        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        key_pre = self.module3( t_vec, batch_size, length_size)

        g_vec=0
        a_vec=0

        return g_vec,a_vec,key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr_smpl(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr_smpl, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_ti()
        self.module4 = CombineModule_rgb()
        self.module5 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, g_vec, a_vec, f_tcmr,batch_size,length_size):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr
        f_tcmr = f_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        b=self.module3(g_vec, a_vec, batch_size, length_size)
        q, t= self.module4(t_vec, batch_size, length_size)
        _, key_pre = self.module5(q,t,b)

        return key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_hmr_notrain_single_rgb_smpl(nn.Module):
    def __init__(self):
        super(mmWaveModel_hmr_notrain_single_rgb_smpl, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_rgb_smpl()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)


        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, g_vec, a_vec, f_tcmr, batch_size, length_size):
        f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        q, t, b, g = self.module3( t_vec, batch_size, length_size)
        v, s = self.module4(q, t, b)
        g_vec=0
        a_vec=0

        return s

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

'''metric learning'''
class mmWaveModel_metric(nn.Module):
    def __init__(self,device2):
        super(mmWaveModel_metric, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr()
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional()
        #self.model_ti.load('./log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(2999))
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, ti_p,ti_n,x_rgb,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_hmr = feature_hmr.transpose(1, 2)
        t_vec = self.caf1(self.cb1(self.conv1(feature_hmr)))
        t_vec = t_vec.transpose(1, 2)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # print("feature_tcmr",feature_tcmr.shape)

        # mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)


        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        #t_vec = f_tcmr.transpose(1, 2)

        t_vec = torch.flatten(t_vec, start_dim=1, end_dim=2)
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)

        ti_h = F.normalize(ti_h)
        ti_h2 = F.normalize(ti_h2)
        t_vec = F.normalize(t_vec)
        #print("rgb:",t_vec.shape)
        #print("ti_h:", ti_h.shape)


        #print("g_vec:", g_vec.shape)
        #print("a_vec:",a_vec.shape)
        #print("t_vec:", t_vec.shape)
        #print("ti_vec:", ti_vec.shape)

        return ti_h,ti_h2,t_vec

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_metric_single_ti(nn.Module):
    def __init__(self):
        super(mmWaveModel_metric_single_ti, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_nosmpl_bidirectional()

    def forward(self, x,x2,x3,h0, c0,  batch_size,length_size):
       # print( x.size())
        out_feature_size = 24 + 3

        x = self.module0(x)
        x2 = self.module0(x2)
        x3 = self.module0(x3)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0, c0, batch_size, length_size,
                                                         out_feature_size)

        g_vec2, g_loc2, global_weights2, hn_g2, cn_g2 = self.module1(x2, h0, c0, batch_size, length_size)
        a_vec2, anchor_weights2, hn_a2, cn_a2 = self.module2(x2, g_loc2, h0, c0, batch_size, length_size,
                                                         out_feature_size)

        g_vec3, g_loc3, global_weights3, hn_g3, cn_g3 = self.module1(x3, h0, c0, batch_size, length_size)
        a_vec3, anchor_weights3, hn_a3, cn_a3 = self.module2(x3, g_loc2, h0, c0, batch_size, length_size,
                                                             out_feature_size)

        #print("g_vec:", a_vec.shape)
        #print("a_vec:", a_vec)

        g_vec = torch.flatten(g_vec, start_dim=1, end_dim=2)
        a_vec = torch.flatten(a_vec, start_dim=1, end_dim=2)
        ti_vec = torch.cat((a_vec, g_vec), dim=1)

        g_vec2 = torch.flatten(g_vec2, start_dim=1, end_dim=2)
        a_vec2 = torch.flatten(a_vec2, start_dim=1, end_dim=2)
        ti_vec2 = torch.cat((a_vec2, g_vec2), dim=1)

        g_vec3 = torch.flatten(g_vec3, start_dim=1, end_dim=2)
        a_vec3 = torch.flatten(a_vec3, start_dim=1, end_dim=2)
        ti_vec3 = torch.cat((a_vec3, g_vec3), dim=1)


        #v=F.normalize(v[0][0][:][0:1])
        return ti_vec,ti_vec2,ti_vec3

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_metric_single_ti_ptnet(nn.Module):
    def __init__(self):
        super(mmWaveModel_metric_single_ti_ptnet, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_nosmpl_bidirectional()

    def forward(self, x,x2,x3,h0, c0,  batch_size,length_size):
        out_feature_size = 24 + 3

        x = self.module0(x)
        x2 = self.module0(x2)
        x3 = self.module0(x3)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0, c0, batch_size, length_size,
                                                         out_feature_size)

        g_vec2, g_loc2, global_weights2, hn_g2, cn_g2 = self.module1(x2, h0, c0, batch_size, length_size)
        a_vec2, anchor_weights2, hn_a2, cn_a2 = self.module2(x2, g_loc2, h0, c0, batch_size, length_size,
                                                             out_feature_size)

        g_vec3, g_loc3, global_weights3, hn_g3, cn_g3 = self.module1(x3, h0, c0, batch_size, length_size)
        a_vec3, anchor_weights3, hn_a3, cn_a3 = self.module2(x3, g_loc2, h0, c0, batch_size, length_size,
                                                             out_feature_size)

        # print("g_vec:", a_vec.shape)
        # print("a_vec:", a_vec)

        g_vec = torch.flatten(g_vec, start_dim=1, end_dim=2)
        a_vec = torch.flatten(a_vec, start_dim=1, end_dim=2)
        ti_vec = torch.cat((a_vec, g_vec), dim=1)

        g_vec2 = torch.flatten(g_vec2, start_dim=1, end_dim=2)
        a_vec2 = torch.flatten(a_vec2, start_dim=1, end_dim=2)
        ti_vec2 = torch.cat((a_vec2, g_vec2), dim=1)

        g_vec3 = torch.flatten(g_vec3, start_dim=1, end_dim=2)
        a_vec3 = torch.flatten(a_vec3, start_dim=1, end_dim=2)
        ti_vec3 = torch.cat((a_vec3, g_vec3), dim=1)
        g_vec = F.normalize(g_vec)
        g_vec2 = F.normalize(g_vec2)
        g_vec3 = F.normalize(g_vec3)

        # v=F.normalize(v[0][0][:][0:1])
        return g_vec, g_vec2, g_vec3


    def save(self, name=None):
            """
            保存模型，默认使用“模型名字+时间”作为文件名
            """
            if name is None:
                prefix = 'checkpoints/'
                name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            torch.save(self.state_dict(), name)
            return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_metric_single_rgb(nn.Module):
    def __init__(self,device2):
        super(mmWaveModel_metric_single_rgb, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr()
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.fc1 = nn.Linear(2048, 256)
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,x_rgb2,x_rgb3, batch_size,seq_len):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, seq_len, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)

        feature_hmr2 = self.model_hmr.feature_extractor(x_rgb2)
        feature_hmr2 = feature_hmr2.view(batch_size, seq_len, 2048)
        feature_tcmr2, _ = self.model_tcmr(feature_hmr2)

        feature_hmr3 = self.model_hmr.feature_extractor(x_rgb3)
        feature_hmr3 = feature_hmr3.view(batch_size, seq_len, 2048)
        feature_tcmr3, _ = self.model_tcmr(feature_hmr3)
        # print("feature_tcmr",feature_tcmr.shape)
        # g_vec, a_vec, _ = self.model_ti(x, h0, c0, h0_a, c0_a, batchsize, length_size)

        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_vec = torch.flatten(t_vec, start_dim=1, end_dim=2)

        f_tcmr2 = feature_tcmr2.transpose(1, 2)
        f_tcmr2 = self.caf1(self.cb1(self.conv1(f_tcmr2)))
        t_vec2 = f_tcmr2.transpose(1, 2)
        rgb_vec2 = torch.flatten(t_vec2, start_dim=1, end_dim=2)

        f_tcmr3 = feature_tcmr3.transpose(1, 2)
        f_tcmr3 = self.caf1(self.cb1(self.conv1(f_tcmr3)))
        t_vec3 = f_tcmr3.transpose(1, 2)
        rgb_vec3 = torch.flatten(t_vec3, start_dim=1, end_dim=2)

        return rgb_vec,rgb_vec2,rgb_vec3

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

'''lime'''
class mmWaveModel_tcmr_lime(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr_lime, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr_lime()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, g_vec, a_vec, f_tcmr):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr
        f_tcmr = f_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        #t_vec = t_vec.squeeze()
        #ti_vec = torch.cat((a_vec, g_vec), dim=1)
        #ti_vec = ti_vec.squeeze()
        #x = torch.cat((ti_vec, t_vec), dim=1)
        #key_pre = self.module3(x, 1, 10)
        #key_pre = torch.mean(key_pre, dim=1)
        key_pre = 0
        return key_pre,t_vec

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr_lime2(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr_lime2, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr_lime()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, x):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr

        key_pre = self.module3(x, 1, 10)
        key_pre = torch.mean(key_pre, dim=1)
        return key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mmWaveModel_tcmr_lime_singleframe(nn.Module):
    def __init__(self):
        super(mmWaveModel_tcmr_lime_singleframe, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        self.module3 = CombineModule_tcmr_lime()
        self.module4 = SMPLModule()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)

    def forward(self, x):
        #多模态
        #print("f_tcmr",f_tcmr.shape)
        #t_vec=self.fc1(f_tcmr)
        #t_vec=f_tcmr

        key_pre = self.module3(x, 1000, 1)
        key_pre = torch.mean(key_pre,dim=1)
        print("key_pre:",key_pre.shape)
        return key_pre

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

'''mid-modal'''
class CombineModule_mid_modal(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_mid_modal_2d(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_2d, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 20 * 2 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:20*2].view(batch_size, length_size, 20, 2)

        return key_pre

class CombineModule_mid_modal_512(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_512, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_mid_modal_t(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_t, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:1*3].view(batch_size, length_size, 1, 3)

        return key_pre

class CombineModule_mid_modal_singlem(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_singlem, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:,:,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_mid_modal_fusionm(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_fusionm, self).__init__()
        self.fc1 = nn.Linear(256+128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:,:,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class mid_modal_hmr_train(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l2 = F.normalize(ti_l2, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#single rgb
class mid_modal_hmr_train_singlergb(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_singlergb, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,rgb_p,rgb_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)

        feature_hmr_p = self.model_hmr.feature_extractor(rgb_p)
        feature_hmr_p = feature_hmr_p.view(batch_size, length_size, 2048)
        feature_tcmr_p, _ = self.model_tcmr(feature_hmr_p)

        feature_hmr_n = self.model_hmr.feature_extractor(rgb_n)
        feature_hmr_n = feature_hmr_n.view(batch_size, length_size, 2048)
        feature_tcmr_n, _ = self.model_tcmr(feature_hmr_n)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        f_tcmr_p = feature_tcmr_p.transpose(1, 2)
        f_tcmr_p = self.caf1(self.cb1(self.conv1(f_tcmr_p)))
        t_vec_p = f_tcmr_p.transpose(1, 2)
        rgb_h_p = t_vec_p[:, :, 256:]
        rgb_l_p = t_vec_p[:, :, :256]

        f_tcmr_n = feature_tcmr_n.transpose(1, 2)
        f_tcmr_n = self.caf1(self.cb1(self.conv1(f_tcmr_n)))
        t_vec_n = f_tcmr_n.transpose(1, 2)
        rgb_h_n = t_vec_n[:, :, 256:]
        rgb_l_n = t_vec_n[:, :, :256]

        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l2 = F.normalize(ti_l2, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_rgb_p = self.module3(rgb_l_p, batch_size, length_size)
        key_pre_rgb_n = self.module3(rgb_l_n, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_rgb_p = key_pre_rgb_p.view(batch_size * length_size, 24, 3)
        key_pre_rgb_n = key_pre_rgb_n.view(batch_size * length_size, 24, 3)
        #总输出
        rgb_h_p = torch.flatten(rgb_h_p, start_dim=1, end_dim=2)
        rgb_l_p = torch.flatten(rgb_l_p, start_dim=1, end_dim=2)
        rgb_h_n = torch.flatten(rgb_h_n, start_dim=1, end_dim=2)
        rgb_l_n = torch.flatten(rgb_l_n, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((rgb_h_p, rgb_l_p), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((rgb_h_n, rgb_l_n), dim=1)
        output3 = F.normalize(output3)

        rgb_h_p = F.normalize(rgb_h_p)
        rgb_l_p = F.normalize(rgb_l_p)
        rgb_h_n = F.normalize(rgb_h_n)
        rgb_l_n = F.normalize(rgb_l_n)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,rgb_h_p,rgb_h_n,key_pre_rgb,key_pre_rgb_p,key_pre_rgb_n,output1,output2,output3,rgb_l,rgb_l_p,rgb_l_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#NLN原版
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 相当于计算value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        print("phi_x:", phi_x.shape)
        print("theta_x:", theta_x.shape)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        f_div_C = F.softmax(f, dim=1)
        f_rgb = f[:,:20,:]
        f_ti = f[:, 20:, :]
        f_rgb = F.softmax(f_rgb, dim=-1)
        f_ti = F.softmax(f_ti, dim=-1)
        print("f_rgb:",f_rgb.shape)


        # 绘制attention map
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(4):
            plt.figure(figsize=(24, 12))
            plot = sns.heatmap(f_ti[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
            # plt.pause(1.3)
            # print(ax.lines)
            plt.show()


        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x
        #print("W_y:",W_y.mean())
        #print("x:", x.mean())
        #print("g_x:", g_x.mean())
        if return_nl_map:
            return z, f_div_C
        return z

class _NonLocalBlockND_S2MA(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_S2MA, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.Image_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.Ptcloud_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        #self mutual atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.F_g = nn.Sequential(self.F_g, max_pool_layer)
            self.F_phi = nn.Sequential(self.F_phi, max_pool_layer)
            self.R_g = nn.Sequential(self.R_g, max_pool_layer)
            self.R_phi = nn.Sequential(self.R_phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)
        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)


            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea, alpha, selfImage, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        if selfImage:
            selfNonLocal_fea = self.Image_bnRelu(self_fea)
            mutualNonLocal_fea = self.Ptcloud_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.F_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)

            f = torch.matmul(theta_x, phi_x)
            #print("f_image:",f[0])

            # using self feature to generate attention
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha * f + self_f, dim=-1)
            '''
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            # plt.ion()
            for i in range(4):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".2f")
                # plt.pause(1.3)
                # print(ax.lines)
                plt.show()
                # plt.clf()
            # plt.ioff()
            '''
            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.R_W(y)
            z = W_y + self_fea
            return z

        else:
            selfNonLocal_fea = self.Ptcloud_bnRelu(self_fea)
            mutualNonLocal_fea = self.Image_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.R_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.R_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            '''
            #绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            plot = sns.heatmap(f[0].cpu().detach(),linewidths=0.8,annot=True,fmt=".2f")
            plt.show()
            print("f_ptcloud:", f[0])
            '''

            # using self feature to generate attention
            self_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.F_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha * f + self_f, dim=-1)
            '''
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            #plt.ion()
            for i in range(4):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".2f")
                #plt.pause(1.3)
                # print(ax.lines)
                plt.show()
                #plt.clf()
            #plt.ioff()
            '''


            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea


        if return_nl_map:
            return z, f_div_C
        return z

#0429修改：使用两个nln且分开计算qkv
#0501修改：只映射最终结果，即self和mutual的结果相加后映射
#0502修改：v的映射网络加入bn防止两个模态映射不一致
class _NonLocalBlockND_2modules(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.Image_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.Ptcloud_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))

        #self mutual atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))


        if sub_sample:
            self.F_g = nn.Sequential(self.F_g, max_pool_layer)
            self.F_phi = nn.Sequential(self.F_phi, max_pool_layer)
            self.R_g = nn.Sequential(self.R_g, max_pool_layer)
            self.R_phi = nn.Sequential(self.R_phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)
        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)


            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea,  return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """


        selfNonLocal_fea = self.Image_bnRelu(self_fea)
        mutualNonLocal_fea = self.Ptcloud_bnRelu(mutual_fea)

        batch_size = selfNonLocal_fea.size(0)

        # using self feature to generate attention
        self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_g_x = self_g_x.permute(0, 2, 1)
        #self_g_x = F.normalize(self_g_x, dim=2)
        #print("self_g_x:", torch.mean(self_g_x))
        self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_theta_x = self_theta_x.permute(0, 2, 1)
        self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_f = torch.matmul(self_theta_x, self_phi_x)

        #self_f = torch.rand(batch_size , 20, 20, device= 'cuda:0')
        self_f_div_C = F.softmax(self_f, dim=-1)
        '''
        print("self_attention:")
        # 绘制attention map
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(4):
            plt.figure(figsize=(24, 12))
            plot = sns.heatmap(self_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
            # plt.pause(1.3)
            # print(ax.lines)
            plt.show()
        '''
        self_y = torch.matmul(self_f_div_C, self_g_x)
        self_y = self_y.permute(0, 2, 1).contiguous()
        #print("self_y:",self_y.shape)
        self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
        #只映射最终结果，即self和mutual的结果相加后映射
        #self_W_y = self.R_W(self_y)


        # using mutual feature to generate attention
        mutual_g_x = self.R_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_g_x = mutual_g_x.permute(0, 2, 1)
        #print("mutual_g_x:", mutual_g_x.shape)
        #mutual_g_x = F.normalize(mutual_g_x,dim=2)
        #print("mutual_g_x:",torch.mean(mutual_g_x))
        mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
        mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
        #mutual_f = torch.rand(batch_size, 20, 20, device='cuda:0')
        mutual_f_div_C = F.softmax(mutual_f, dim=-1)
        '''
        print("mutual_attention:")
        # 绘制attention map
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(4):
            plt.figure(figsize=(24, 12))
            plot = sns.heatmap(mutual_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
            # plt.pause(1.3)
            # print(ax.lines)
            plt.show()
'''
        mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
        mutual_y = mutual_y.permute(0, 2, 1).contiguous()
        mutual_y = mutual_y.view(batch_size, self.inter_channels, *mutualNonLocal_fea.size()[2:])
        #只映射最终结果，即self和mutual的结果相加后映射
        #mutual_W_y = self.F_W(mutual_y)
        #print("mutual_W_y:", mutual_W_y)
        #print("f_image:",f[0])
        z = mutual_y + self_y
        #print("mutual_y:",torch.mean(mutual_y))
        #print("self_y:", torch.mean(self_y))
        z = self.F_W(z)



        if return_nl_map:
            return z, self_f_div_C, mutual_f_div_C
        return z

#0503修改：self和mutual相加改为拼接
class _NonLocalBlockND_2modules_0503(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_0503, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.Image_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.Ptcloud_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))
        '''
        #self mutual atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))
        '''

        if sub_sample:
            self.F_g = nn.Sequential(self.F_g, max_pool_layer)
            self.F_phi = nn.Sequential(self.F_phi, max_pool_layer)
            self.R_g = nn.Sequential(self.R_g, max_pool_layer)
            self.R_phi = nn.Sequential(self.R_phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

            # 拼接后进行映射
            self.C_W = nn.Sequential(
                conv_nd(in_channels=(self.in_channels) * 2, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.C_W[1].weight, 0)
            nn.init.constant_(self.C_W[1].bias, 0)

        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)


            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea, return_nl_map=False):
            """
            :param x: (b, c, t, h, w)
            :param return_nl_map: if True return z, nl_map, else only return z.
            :return:
            """

            selfNonLocal_fea = self.Image_bnRelu(self_fea)
            mutualNonLocal_fea = self.Ptcloud_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            # using self feature to generate attention
            self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_g_x = self_g_x.permute(0, 2, 1)
            # self_g_x = F.normalize(self_g_x, dim=2)
            # print("self_g_x:", torch.mean(self_g_x))
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            self_f_div_C = F.softmax(self_f, dim=-1)

            '''
            print("self_attention:")
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            for i in range(4):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(self_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
                # plt.pause(1.3)
                # print(ax.lines)
                plt.show()
            '''
            #print("self_f_div_C:",self_f_div_C.shape)
            #print("self_g_x:", self_g_x.shape)
            self_y = torch.matmul(self_f_div_C, self_g_x)
            self_y = self_y.permute(0, 2, 1).contiguous()
            #print("self_y:",self_y.shape)
            self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            self_W_y = self.R_W(self_y)

            # using mutual feature to generate attention
            mutual_g_x = self.R_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_g_x = mutual_g_x.permute(0, 2, 1)
            # print("mutual_g_x:", mutual_g_x.shape)
            # mutual_g_x = F.normalize(mutual_g_x,dim=2)
            # print("mutual_g_x:",torch.mean(mutual_g_x))
            mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
            mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)

            mutual_f_div_C = F.softmax(mutual_f, dim=-1)
            #print("mutual_f_div_C:", mutual_f_div_C.shape)

            '''
            print("mutual_attention:")
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            for i in range(4):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(mutual_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
                # plt.pause(1.3)
                # print(ax.lines)
                plt.show()
            '''
            mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
            mutual_y = mutual_y.permute(0, 2, 1).contiguous()
            mutual_y = mutual_y.view(batch_size, self.inter_channels, *mutualNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            mutual_W_y = self.F_W(mutual_y)
            # print("mutual_W_y:", mutual_W_y.shape)
            # print("f_image:",f[0])
            '''
            #0502修改
            z = mutual_y + self_y
            z = self.F_W(z)
            '''
            # 0503修改，拼接做法
            z = torch.cat((self_W_y, mutual_W_y), dim=1)
            #0505修改：拼接后直接返回
            z = self.C_W(z)
            # print("z:", z.shape)

            if return_nl_map:
                return z, self_f_div_C, mutual_f_div_C
            return z

class _NonLocalBlockND_2modules_0505(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_0505, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.Image_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.Ptcloud_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))
        '''
        #self mutual atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))
        '''

        if sub_sample:
            self.F_g = nn.Sequential(self.F_g, max_pool_layer)
            self.F_phi = nn.Sequential(self.F_phi, max_pool_layer)
            self.R_g = nn.Sequential(self.R_g, max_pool_layer)
            self.R_phi = nn.Sequential(self.R_phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

            # 拼接后进行映射
            self.C_W = nn.Sequential(
                conv_nd(in_channels=(self.in_channels) * 2, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.C_W[1].weight, 0)
            nn.init.constant_(self.C_W[1].bias, 0)

        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)


            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea, return_nl_map=False):
            """
            :param x: (b, c, t, h, w)
            :param return_nl_map: if True return z, nl_map, else only return z.
            :return:
            """

            selfNonLocal_fea = self.Image_bnRelu(self_fea)
            mutualNonLocal_fea = self.Ptcloud_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            # using self feature to generate attention
            self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_g_x = self_g_x.permute(0, 2, 1)
            # self_g_x = F.normalize(self_g_x, dim=2)
            # print("self_g_x:", torch.mean(self_g_x))
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            self_f_div_C = F.softmax(self_f, dim=-1)

            '''
            print("self_attention:")
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            for i in range(1):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(self_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
                # plt.pause(1.3)
                # print(ax.lines)
                plt.show()
'''
            self_y = torch.matmul(self_f_div_C, self_g_x)
            self_y = self_y.permute(0, 2, 1).contiguous()
            # print("self_y:",self_y.shape)
            self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            self_W_y = self.R_W(self_y)

            # using mutual feature to generate attention
            mutual_g_x = self.R_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_g_x = mutual_g_x.permute(0, 2, 1)
            # print("mutual_g_x:", mutual_g_x.shape)
            # mutual_g_x = F.normalize(mutual_g_x,dim=2)
            # print("mutual_g_x:",torch.mean(mutual_g_x))
            mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
            mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
            mutual_f_div_C = F.softmax(mutual_f, dim=-1)

            '''
            print("mutual_attention:")
            # 绘制attention map
            import matplotlib.pyplot as plt
            import seaborn as sns
            for i in range(1):
                plt.figure(figsize=(24, 12))
                plot = sns.heatmap(mutual_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
                # plt.pause(1.3)
                # print(ax.lines)
                plt.show()
'''
            mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
            mutual_y = mutual_y.permute(0, 2, 1).contiguous()
            mutual_y = mutual_y.view(batch_size, self.inter_channels, *mutualNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            mutual_W_y = self.F_W(mutual_y)
            # print("mutual_W_y:", mutual_W_y.shape)
            # print("f_image:",f[0])
            '''
            #0502修改
            z = mutual_y + self_y
            z = self.F_W(z)
            '''
            # 0503修改，拼接做法
            z = torch.cat((self_W_y, mutual_W_y), dim=1)
            #0505修改：拼接后直接返回
            #z = self.C_W(z)
            # print("z:", z.shape)

            if return_nl_map:
                return z, self_f_div_C, mutual_f_div_C
            return z,self_W_y

class _NonLocalBlockND_2modules_0507(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_0507, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.Image_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.Ptcloud_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.F_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))

        #self mutual atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                         kernel_size=1, stride=1, padding=0)
        '''
        self.R_g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),bn(self.inter_channels))


        if sub_sample:
            self.F_g = nn.Sequential(self.F_g, max_pool_layer)
            self.F_phi = nn.Sequential(self.F_phi, max_pool_layer)
            self.R_g = nn.Sequential(self.R_g, max_pool_layer)
            self.R_phi = nn.Sequential(self.R_phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)
        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)


            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea,  return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """


        selfNonLocal_fea = self.Image_bnRelu(self_fea)
        mutualNonLocal_fea = self.Ptcloud_bnRelu(mutual_fea)

        batch_size = selfNonLocal_fea.size(0)

        # using self feature to generate attention
        self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_g_x = self_g_x.permute(0, 2, 1)
        #self_g_x = F.normalize(self_g_x, dim=2)
        #print("self_g_x:", torch.mean(self_g_x))
        self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_theta_x = self_theta_x.permute(0, 2, 1)
        self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        self_f = torch.matmul(self_theta_x, self_phi_x)
        self_f_div_C = F.softmax(self_f, dim=-1)

        '''
        print("self_attention:")
        # 绘制attention map
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(4):
            plt.figure(figsize=(24, 12))
            plot = sns.heatmap(self_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
            # plt.pause(1.3)
            # print(ax.lines)
            plt.show()
        '''
        self_y = torch.matmul(self_f_div_C, self_g_x)
        self_y = self_y.permute(0, 2, 1).contiguous()
        #print("self_y:",self_y.shape)
        self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
        #只映射最终结果，即self和mutual的结果相加后映射
        #self_W_y = self.R_W(self_y)


        # using mutual feature to generate attention
        mutual_g_x = self.R_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_g_x = mutual_g_x.permute(0, 2, 1)
        #print("mutual_g_x:", mutual_g_x.shape)
        #mutual_g_x = F.normalize(mutual_g_x,dim=2)
        #print("mutual_g_x:",torch.mean(mutual_g_x))
        mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
        mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
        mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
        mutual_f_div_C = F.softmax(mutual_f, dim=-1)

        '''
        print("mutual_attention:")
        # 绘制attention map
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(4):
            plt.figure(figsize=(24, 12))
            plot = sns.heatmap(mutual_f_div_C[i].cpu().detach(), linewidths=0.8, annot=True, fmt=".3f")
            # plt.pause(1.3)
            # print(ax.lines)
            plt.show()
'''
        mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
        mutual_y = mutual_y.permute(0, 2, 1).contiguous()
        mutual_y = mutual_y.view(batch_size, self.inter_channels, *mutualNonLocal_fea.size()[2:])
        #只映射最终结果，即self和mutual的结果相加后映射
        #mutual_W_y = self.F_W(mutual_y)
        #print("mutual_W_y:", mutual_W_y)

        z = mutual_y + self_y
        z = self.F_W(z)



        if return_nl_map:
            return z, self_f_div_C, mutual_f_div_C
        return z

class mid_modal_hmr_train_nln(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_nln, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        #2个regressor
        self.module4 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        #mutual attention
        self.nl = _NonLocalBlockND(in_channels=256)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # mutual attention2
        '''
        
        self.nl2 = _NonLocalBlockND(in_channels=256)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()
        '''

        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)
        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        rgb_l_ma = rgb_l.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3).view(batch_size*length_size,256,512)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))
        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out))
        rgb_ma = torch.sum(nl_out * attn_weights1, dim=1)  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out))
        #print("attn_weights1:",attn_weights1.shape)
        #print(attn_weights1==attn_weights2)
        ti_ma = torch.sum(nl_out * attn_weights2, dim=1)  # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #print(rgb_ma == ti_ma)
        #reconstruction
        key_pre_rgb = self.module3(rgb_ma, batch_size, length_size)
        key_pre_ti = self.module4(ti_ma, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_ma, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_ma, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#不训练reid 只包含重建网络
class mid_modal_hmr_train_nln_woreid(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_nln_woreid, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

        # mutual attention
        self.nl = _NonLocalBlockND(in_channels=256)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # mutual attention2
        '''

        self.nl2 = _NonLocalBlockND(in_channels=256)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()
        '''

        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)
        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_l, a_vec_l, _ = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_l2, a_vec_l2, _ = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_l = t_vec

        # mutual attention
        rgb_l_ma = rgb_l.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        # print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3).view(batch_size * length_size, 256, 512)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))
        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out))
        rgb_ma = torch.sum(nl_out * attn_weights1, dim=1)  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out))
        # print("attn_weights1:",attn_weights1.shape)
        # print(attn_weights1==attn_weights2)
        ti_ma = torch.sum(nl_out * attn_weights2, dim=1)  # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size, -1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        # print(rgb_ma == ti_ma)
        # reconstruction
        key_pre_rgb = self.module3(rgb_ma, batch_size, length_size)
        key_pre_ti = self.module3(ti_ma, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        ti_l = torch.flatten(ti_ma, start_dim=1, end_dim=2)

        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)

        rgb_l = torch.flatten(rgb_ma, start_dim=1, end_dim=2)


        ti_l = F.normalize(ti_l)
        ti_l2 = F.normalize(ti_l2)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''
        rgb_h = []
        ti_h = []
        ti_h2 = []
        output1 = []
        output2 = []
        output3 = []
        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#2nln分开
class mid_modal_hmr_train_2nln(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_2nln, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        #nln模块
        self.nl = _NonLocalBlockND(in_channels=256)
        self.nl2 = _NonLocalBlockND(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        rgb_l_ma = rgb_l_ma.transpose(1, 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        nl_out_rgb = self.nl(rgb_l_ma)
        nl_out_rgb = nl_out_rgb.transpose(1, 2).contiguous()
        nl_out_ti = self.nl2(ti_l_ma)
        nl_out_ti = nl_out_ti.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out_rgb))
        rgb_ma = nl_out_rgb * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out_ti))
        ti_ma = nl_out_ti * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_fusion = ti_ma+rgb_l
        ti_fusion = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#监督g_loc
class mid_modal_hmr_train_2nln_loc(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_2nln_loc, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        #nln模块
        self.nl = _NonLocalBlockND(in_channels=256)
        self.nl2 = _NonLocalBlockND(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_l = F.normalize(ti_l)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        rgb_l = F.normalize(rgb_l)
        ti_l = ti_l.view(batch_size,length_size,  256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)

        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)


        rgb_l_ma = rgb_l_ma.transpose(1, 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        nl_out_rgb = self.nl(rgb_l_ma)
        nl_out_rgb = nl_out_rgb.transpose(1, 2).contiguous()
        nl_out_ti = self.nl2(ti_l_ma)
        nl_out_ti = nl_out_ti.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out_rgb))
        rgb_ma = nl_out_rgb * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out_ti))
        ti_ma = nl_out_ti * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_fusion = ti_ma+rgb_l
        ti_fusion = rgb_ma+ti_l

        #reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2, g_loc_p1 ,g_loc_p2,g_loc_n1,g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#0423更正nln模块
class mid_modal_hmr_train_nln_loc_0423(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_nln_loc_0423, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_l = F.normalize(ti_l)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        rgb_l = F.normalize(rgb_l)
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)

        rgb_l_ma = rgb_l.contiguous().view(batch_size, length_size, 256)
        ti_l_ma = ti_l.contiguous().view(batch_size, length_size, 256)
        # print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        #nln拼接后split
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 1)
        #print("feature_ma:",feature_ma.shape)
        feature_ma = feature_ma.transpose(1,2)
        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2)
        rgb_fusion = nl_out[:,:length_size,:]
        ti_fusion = nl_out[:, length_size:, :]


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mid_modal_hmr_train_nln_loc_S2MA(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_nln_loc_S2MA, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_S2MA(in_channels=256)
        self.affinityAttConv = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=2, kernel_size=1),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),
        )

        self.image_bn_relu = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.ptcloud_bn_relu = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        rgb_l = rgb_l.permute(0, 2, 1)
        ti_l = ti_l.permute(0, 2, 1)
        affinityAtt = F.softmax(self.affinityAttConv(torch.cat([rgb_l, ti_l], dim=1)))

        alphaD = affinityAtt[:, 0, :].reshape([batch_size, length_size, 1])
        alphaR = affinityAtt[:, 1, :].reshape([batch_size, length_size, 1])
        #print("alphaD:", alphaD[0])
        #print("alphaR:", alphaR[0])
        alphaD = alphaD.expand([batch_size, length_size, length_size])

        alphaR = alphaR.expand([batch_size, length_size, length_size])

        ImageAfterAtt1 = self.nl(rgb_l, ti_l, alphaD, selfImage=True)
        DepthAfterAtt1 = self.nl(ti_l, rgb_l, alphaR, selfImage=False)

        rgb_fusion = self.image_bn_relu(ImageAfterAtt1)
        ti_fusion = self.ptcloud_bn_relu(DepthAfterAtt1)


        # reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#0507修改:fusion和specific回归器分离
class mid_modal_hmr_train_2nln_loc_0429(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0429, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion = self.nl(rgb_l_ma,ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)


        # reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_train_2nln_loc_0503(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0503, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0503(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0503(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        print("rgb_nln:")
        rgb_fusion = self.nl(rgb_l_ma,ti_l_ma)
        print("mmwave_nln:")
        ti_fusion = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0505修改：self监督+拼接后直接输出
#0508修改：fusion和specific回归器分离+self监督
class mid_modal_hmr_train_2nln_loc_0505(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0505, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_512()


        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0505(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0505(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        print("rgb_nln:")
        rgb_fusion,rgb_self = self.nl(rgb_l_ma,ti_l_ma)
        print("mmwave_nln:")
        ti_fusion,ti_self = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_self = rgb_self.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb_self = self.module3(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0510修改：fusion和specific2模态回归器分离+self监督+hmr only
class mid_modal_hmr_train_2nln_loc_0510(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0510, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_512()
        self.module5 = CombineModule_mid_modal()


        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0505(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0505(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        #hmr only
        feature_tcmr = feature_hmr
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        print("rgb_nln:")
        rgb_fusion,rgb_self = self.nl(rgb_l_ma,ti_l_ma)
        print("mmwave_nln:")
        ti_fusion,ti_self = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_self = rgb_self.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0511修改：所有回归器都分离+self监督
class mid_modal_hmr_train_2nln_loc_0511(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0511, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_512()
        self.module5 = CombineModule_mid_modal_512()
        self.module6 = CombineModule_mid_modal()


        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0505(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0505(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion,rgb_self = self.nl(rgb_l_ma,ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion,ti_self = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_self = rgb_self.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb_self = self.module6(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module5(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0516修改：使用像素级attention
class mid_modal_hmr_train_pixelatten_loc(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr_atten().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_512()
        self.module5 = CombineModule_mid_modal()


        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0505(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0505(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        #hmr only
        feature_tcmr = feature_hmr
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        print("rgb_nln:")
        rgb_fusion,rgb_self = self.nl(rgb_l_ma,ti_l_ma)
        print("mmwave_nln:")
        ti_fusion,ti_self = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_self = rgb_self.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_feature_extract(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_feature_extract, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion = self.nl(rgb_l_ma,ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)



        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_train_2nln_atten(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_2nln_atten, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        #nln模块
        self.nl = _NonLocalBlockND(in_channels=256)
        self.nl2 = _NonLocalBlockND(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        rgb_l_ma = rgb_l_ma.transpose(1, 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        nl_out_rgb = self.nl(rgb_l_ma)
        nl_out_rgb = nl_out_rgb.transpose(1, 2).contiguous()
        nl_out_ti = self.nl2(ti_l_ma)
        nl_out_ti = nl_out_ti.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out_rgb))
        rgb_ma = nl_out_rgb * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out_ti))
        ti_ma = nl_out_ti * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_fusion = ti_ma+rgb_l
        ti_fusion = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)


        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#不适用nln，直接使用attention
class mid_modal_hmr_train_wonln(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_wonln, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), map_location=torch.device('cuda:0'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        attn_weights1 = self.softmax1(self.attn1(rgb_l_ma))
        rgb_ma = rgb_l_ma * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(ti_l_ma))
        ti_ma = ti_l_ma * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_final = ti_ma+rgb_l
        ti_final = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_final, batch_size, length_size)
        key_pre_ti = self.module3(ti_final, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#额外估计位移t
class mid_modal_hmr_train_wonln_t(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_wonln_t, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), map_location=torch.device('cuda:0'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_t()
        self.module5 = CombineModule_mid_modal_t()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        attn_weights1 = self.softmax1(self.attn1(rgb_l_ma))
        rgb_ma = rgb_l_ma * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(ti_l_ma))
        ti_ma = ti_l_ma * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_final = ti_ma+rgb_l
        ti_final = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_final, batch_size, length_size)
        key_pre_ti = self.module3(ti_final, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        key_pre_rgb_t = self.module4(rgb_final, batch_size, length_size)
        key_pre_ti_t = self.module4(ti_final, batch_size, length_size)
        key_pre_ti2_t = self.module5(ti_l2, batch_size, length_size)
        key_pre_rgb_t = key_pre_rgb_t.view(batch_size * length_size, 1, 3)
        key_pre_ti_t = key_pre_ti_t.view(batch_size * length_size, 1, 3)
        key_pre_ti2_t = key_pre_ti2_t.view(batch_size * length_size, 1, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2,key_pre_rgb_t,key_pre_ti_t,key_pre_ti2_t

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mid_modal_hmr_train_wonln_atten(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_wonln_atten, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), map_location=torch.device('cuda:0'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        attn_weights1 = self.softmax1(self.attn1(rgb_l_ma))
        rgb_ma = rgb_l_ma * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(ti_l_ma))
        ti_ma = ti_l_ma * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_final = ti_ma+rgb_l
        ti_final = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_final, batch_size, length_size)
        key_pre_ti = self.module3(ti_final, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#高维低维rgb-backbone分开
class mid_modal_2hmr_train_2nln(nn.Module):
    def __init__(self,device2):
        super(mid_modal_2hmr_train_2nln, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr2 = hmr().to(device2)
        self.model_hmr2.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.model_tcmr2 = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        self.model_tcmr2.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

        #nln模块
        self.nl = _NonLocalBlockND(in_channels=256)
        self.nl2 = _NonLocalBlockND(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        feature_hmr2 = self.model_hmr2.feature_extractor(x_rgb)
        feature_hmr2 = feature_hmr2.view(batch_size, length_size, 2048)
        feature_tcmr2, _ = self.model_tcmr2(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        f_tcmr2 = feature_tcmr2.transpose(1, 2)
        f_tcmr2 = self.caf1(self.cb1(self.conv1(f_tcmr2)))
        t_vec2 = f_tcmr2.transpose(1, 2)
        rgb_h = t_vec
        rgb_l = t_vec2

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        rgb_l_ma = rgb_l_ma.transpose(1, 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        nl_out_rgb = self.nl(rgb_l_ma)
        nl_out_rgb = nl_out_rgb.transpose(1, 2).contiguous()
        nl_out_ti = self.nl2(ti_l_ma)
        nl_out_ti = nl_out_ti.transpose(1, 2).contiguous()
        attn_weights1 = self.softmax1(self.attn1(nl_out_rgb))
        rgb_ma = nl_out_rgb * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(nl_out_ti))
        ti_ma = nl_out_ti * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_fusion = ti_ma+rgb_l
        ti_fusion = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module3(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#对重建结果使用ptnet后进行reid
#ptnet
class BasePointKinectNet(nn.Module):
    def __init__(self):
        super(BasePointKinectNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(6)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(12)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        #in_mat:(7*13, 50, 6)=(91, 50, 6)
        #in_mat:(8*25,512,3)=(200,512,3)
        x = in_mat.transpose(1,2)   #转置       # x:(91, 6, 50) point(x,y,z,range,intensity,velocity)

        x = self.caf1(self.cb1(self.conv1(x)))  # x:(91, 8, 50)
        x = self.caf2(self.cb2(self.conv2(x)))  # x:(91, 16, 50)
        x = self.caf3(self.cb3(self.conv3(x)))  # x:(91, 24, 50)

        x = x.transpose(1,2)  # x:(91, 50, 24)
        x = torch.cat((in_mat[:,:,:3], x), -1)    # x:(91, 50, 28)  拼接了x,y,z,range

        return x

class GlobalPointKinectNet(nn.Module):
    def __init__(self):
        super(GlobalPointKinectNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        # x:(91, 50, 28)
        x = x.transpose(1,2)   # x:(91, 28, 50)

        x = self.caf1(self.cb1(self.conv1(x)))   # x:(91, 32, 50)
        x = self.caf2(self.cb2(self.conv2(x)))   # x:(91, 48, 50)
        x = self.caf3(self.cb3(self.conv3(x)))   # x:(91, 64, 50)

        x = x.transpose(1,2)   # x:(91, 50, 64)

        attn_weights=self.softmax(self.attn(x))   # attn_weights:(91, 50, 1)
        attn_vec=torch.sum(x*attn_weights, dim=1)  # attn_vec:(91, 64)   * 点乘
        return attn_vec, attn_weights

class GlobalKinectRNN(nn.Module):
    def __init__(self):
        super(GlobalKinectRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)
        #self.rnn = nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.3,bidirectional=False)
        self.fc1 = nn.Linear(64, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        # x:[7, 13, 64]    h0:[3, 7, 64]   c0:[3, 7, 64]
        g_vec, (hn, cn)=self.rnn(x, (h0, c0)) # g_vec:[7, 13, 64] hn:[3, 7, 64] cn:[3, 7, 64]
        g_loc=self.fc1(g_vec) # g_vec:[7, 13, 16]
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc) # g_vec:[7, 13, 2]
        return g_vec, g_loc, hn, cn

class GlobalKinectModule(nn.Module):
    def __init__(self):
        super(GlobalKinectModule, self).__init__()
        self.bpointnet = BasePointKinectNet()
        self.gpointnet=GlobalPointKinectNet()
        self.grnn=GlobalKinectRNN()
        #self.cb1=nn.BatchNorm1d(1600)

    def forward(self, x, h0, c0,  batch_size, length_size):
        x=self.bpointnet(x)
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        g_vec = torch.flatten(g_vec, start_dim=1, end_dim=2)
        #g_vec = torch.flatten(g_vec, start_dim=1, end_dim=2)
        #g_vec = self.cb1(g_vec)
        return g_vec

class mid_modal_hmr_train_wonln_kp(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_wonln_kp, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), map_location=torch.device('cuda:0'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        #对重建结果进行ptnet表征
        self.module6 = GlobalKinectModule()

        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0,h1, c1, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #mutual attention
        #nln模块
        rgb_l_ma = rgb_l.contiguous().view(batch_size,length_size,  256)
        ti_l_ma = ti_l.contiguous().view(batch_size,length_size, 256)
        #print("cat:",torch.cat([rgb_l_ma, ti_l_ma], 3).shape)

        attn_weights1 = self.softmax1(self.attn1(rgb_l_ma))
        rgb_ma = rgb_l_ma * attn_weights1  # * 点乘
        attn_weights2 = self.softmax2(self.attn2(ti_l_ma))
        ti_ma = ti_l_ma * attn_weights2 # * 点乘
        rgb_ma = rgb_ma.view(batch_size, length_size,-1)
        ti_ma = ti_ma.view(batch_size, length_size, -1)
        #mutual attention结果+原始结果
        rgb_final = ti_ma+rgb_l
        ti_final = rgb_ma+ti_l


        #reconstruction
        key_pre_rgb = self.module3(rgb_final, batch_size, length_size)
        key_pre_ti = self.module3(ti_final, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        key_pre_rgb_19pts = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1)
        key_pre_ti_19pts = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1)
        key_pre_ti2_19pts = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = self.module6(key_pre_rgb_19pts, h1, c1, batch_size, length_size)
        output2 = self.module6(key_pre_ti_19pts, h1, c1, batch_size, length_size)
        output3 = self.module6(key_pre_ti2_19pts, h1, c1, batch_size, length_size)

        output1 = F.normalize(output1)
        output2 = F.normalize(output2)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#online triplet loss
class mid_modal_onlinetri(nn.Module):
    def __init__(self,device2):
        super(mid_modal_onlinetri, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l2 = F.normalize(ti_l2, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,key_pre_rgb,key_pre_ti,output1,output2,rgb_l,ti_l

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mid_modal_notcmr_onlinetri(nn.Module):
    def __init__(self,device2):
        super(mid_modal_notcmr_onlinetri, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), map_location=torch.device('cuda:0'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.hmrrnn = HMRRNN_bidirectional().to(device2)


        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_hmr = feature_hmr.transpose(1, 2)
        feature_hmr = self.caf1(self.cb1(self.conv1(feature_hmr)))
        feature_hmr = feature_hmr.transpose(1, 2)
        t_vec = self.hmrrnn(feature_hmr)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        #print("self.model_ti",self.model_ti.device)
        g_vec_h, a_vec_h, _ = self.model_ti(ti, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)



        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #midmodal预测
        #print("rgb_l:",rgb_l.shape)
        #print("ti_l:", ti_l.shape)
        #print("ti_h:", ti_h.shape)
        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,key_pre_rgb,key_pre_ti,output1,output2,rgb_l,ti_l

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname,device):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location=torch.device('cuda:%d' % (device))))

#加入时序attention
class mid_modal_hmr_train_atten(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_atten, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        # 步态周期attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)
        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)


    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l2 = F.normalize(ti_l2, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        attn_weights_ti_h = self.softmax1(self.attn1(ti_h))
        attn_weights_ti_l = self.softmax2(self.attn2(ti_l))
        attn_weights_ti_h2 = self.softmax1(self.attn1(ti_h2))
        attn_weights_ti_l2 = self.softmax2(self.attn2(ti_l2))
        attn_weights_rgb_l = self.softmax3(self.attn3(rgb_l))
        attn_weights_rgb_h = self.softmax4(self.attn4(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        # print("rgb_l:", rgb_l.shape)

        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#加入idloss
class mid_modal_hmr_train_idloss(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_idloss, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.fc_ti = nn.Linear(5120, 128)
        self.faf1 = nn.ReLU()
        self.fc_ti2 = nn.Linear(128, 20)
        self.fc_rgb = nn.Linear(5120, 128)
        self.faf2 = nn.ReLU()
        self.fc_rgb2 = nn.Linear(128, 20)

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l2 = F.normalize(ti_l2, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)


        #在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''
        #print("output2:",output2.shape)
        label_ti = F.softmax(self.fc_ti2(self.faf1(self.fc_ti(output2))))
        label_ti2 = F.softmax(self.fc_ti2(self.faf1(self.fc_ti(output3))))
        label_rgb = F.softmax(self.fc_rgb2(self.faf2(self.fc_rgb(output1))))
        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2\
            ,label_ti,label_ti2,label_rgb

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#不使用TCMR
class HMRRNN_bidirectional(nn.Module):
    def __init__(self):
        super(HMRRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.gru_cur = nn.GRU(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            num_layers=3
        )
        self.fc1 = nn.Linear(512, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        g_vec, state = self.gru_cur(x.permute(1,0,2))
        g_vec = g_vec.permute(1,0,2)

        return g_vec#, g_loc, hn, cn

class mid_modal_notcmr_train(nn.Module):
    def __init__(self,device2):
        super(mid_modal_notcmr_train, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.hmrrnn = HMRRNN_bidirectional().to(device2)


        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_hmr = feature_hmr.transpose(1, 2)
        feature_hmr = self.caf1(self.cb1(self.conv1(feature_hmr)))
        feature_hmr = feature_hmr.transpose(1, 2)
        t_vec = self.hmrrnn(feature_hmr)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        #print("self.model_ti",self.model_ti.device)
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #midmodal预测
        #print("rgb_l:",rgb_l.shape)
        #print("ti_l:", ti_l.shape)
        #print("ti_h:", ti_h.shape)
        # 重建数据归一化
        #rgb_l = F.normalize(rgb_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        #ti_l = F.normalize(ti_l, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)

        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mid_modal_notcmr_train_atten(nn.Module):
    def __init__(self, device2):
        super(mid_modal_notcmr_train_atten, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)
        self.model_hmr.eval()
        self.hmrrnn = HMRRNN_bidirectional().to(device2)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

        #步态周期attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)
        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_hmr = feature_hmr.transpose(1, 2)
        feature_hmr = self.caf1(self.cb1(self.conv1(feature_hmr)))
        feature_hmr = feature_hmr.transpose(1, 2)
        t_vec = self.hmrrnn(feature_hmr)
        # feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        # print("self.model_ti",self.model_ti.device)
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # 取单独一帧的低特征做判断
        rgb_l_single_frame = rgb_l[:, 5, :]
        ti_l_single_frame = ti_l[:, 5, :]
        ti_l2_single_frame = ti_l2[:, 5, :]
        # midmodal预测
        # print("rgb_l:",rgb_l.shape)
        # print("ti_l:", ti_l.shape)
        # print("ti_h:", ti_h.shape)
        # 重建数据归一化
        # rgb_l = F.normalize(rgb_l, dim=2)
        # ti_l = F.normalize(ti_l, dim=2)
        # ti_l = F.normalize(ti_l, dim=2)
        key_pre_rgb = self.module3(rgb_l, batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出
        attn_weights_ti_h = self.softmax1(self.attn1(ti_h))
        attn_weights_ti_l = self.softmax2(self.attn2(ti_l))
        attn_weights_ti_h2 = self.softmax1(self.attn1(ti_h2))
        attn_weights_ti_l2 = self.softmax2(self.attn2(ti_l2))
        attn_weights_rgb_l = self.softmax3(self.attn3(rgb_l))
        attn_weights_rgb_h = self.softmax4(self.attn4(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        #print("rgb_l:", rgb_l.shape)

        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#使用单模态两个回归器+fusion回归器
class mid_modal_hmr_train_2regressors(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_2regressors, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.fc_ti = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc_rgb = nn.Linear(256, 128)
        self.faf2 = nn.ReLU()
        self.fc_fusion = nn.Linear(256, 128)
        self.faf3 = nn.ReLU()

        self.module_regressor_ti = CombineModule_mid_modal()
        self.module_regressor_rgb = CombineModule_mid_modal()
        self.module_regressor_fusion = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        rgb_h = t_vec[:, :, 256:]
        #重建数据归一化
        rgb_l = F.normalize(rgb_h,dim=2)
        ti_l = F.normalize(ti_l, dim=2)
        ti_l2 = F.normalize(ti_l2, dim=2)

        ti_l_fusion = self.faf3(self.fc_fusion(ti_l))
        ti_l2_fusion = self.faf3(self.fc_fusion(ti_l2))
        rgb_l_fusion = self.faf3(self.fc_fusion(rgb_l))

        ti_l =  self.faf1(self.fc_ti(ti_l))
        ti_l2 = self.faf1(self.fc_ti(ti_l2))
        rgb_l = self.faf2(self.fc_rgb(rgb_l))




        key_pre_rgb = self.module_regressor_rgb(rgb_l, batch_size, length_size)
        key_pre_ti = self.module_regressor_ti(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module_regressor_ti(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        key_pre_rgb_fusion = self.module_regressor_fusion(rgb_l_fusion, batch_size, length_size)
        key_pre_ti_fusion = self.module_regressor_fusion(ti_l_fusion, batch_size, length_size)
        key_pre_ti2_fusion = self.module_regressor_fusion(ti_l2_fusion, batch_size, length_size)
        key_pre_rgb_fusion = key_pre_rgb_fusion.view(batch_size * length_size, 24, 3)
        key_pre_ti_fusion = key_pre_ti_fusion.view(batch_size * length_size, 24, 3)
        key_pre_ti2_fusion = key_pre_ti2_fusion.view(batch_size * length_size, 24, 3)

        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,\
               rgb_l, ti_l, ti_l2,key_pre_rgb_fusion,key_pre_ti_fusion,key_pre_ti2_fusion

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#sPecfic+sHared feature
class mid_modal_hmr_train_PH(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_PH, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)

        self.model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.conv_ti = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.cb_ti = nn.BatchNorm1d(128)
        self.caf_ti = nn.ReLU()
        self.conv_rgb = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.cb_rgb = nn.BatchNorm1d(128)
        self.caf_rgb = nn.ReLU()
        self.conv_fusion = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.cb_fusion = nn.BatchNorm1d(128)
        self.caf_fusion = nn.ReLU()

        self.module_regressor_ti = CombineModule_mid_modal_singlem()
        self.module_regressor_rgb = CombineModule_mid_modal_singlem()
        self.module_regressor_fusion = CombineModule_mid_modal_fusionm()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_l = t_vec[:, :, 256:]
        rgb_h = t_vec[:, :, :256]

        #midmodal预测

        #重建数据归一化
        rgb_l = F.normalize(rgb_l,dim=2)
        ti_l = F.normalize(ti_l, dim=2)
        ti_l2 = F.normalize(ti_l2, dim=2)

        #print("ti_l:",ti_l.shape)

        ti_l_fusion = self.caf_fusion(self.cb_fusion(self.conv_fusion(ti_l.transpose(1, 2))))
        rgb_l_fusion = self.caf_fusion(self.cb_fusion(self.conv_fusion(rgb_l.transpose(1, 2))))
        ti_l =  self.caf_ti(self.cb_ti(self.conv_ti(ti_l.transpose(1, 2))))
        ti_l2 = self.caf_ti(self.cb_ti(self.conv_ti(ti_l2.transpose(1, 2))))
        rgb_l =self.caf_rgb(self.cb_rgb(self.conv_rgb(rgb_l.transpose(1, 2))))
        ti_l_fusion = ti_l_fusion.transpose(1, 2)
        rgb_l_fusion = rgb_l_fusion.transpose(1, 2)
        ti_l = ti_l.transpose(1, 2)
        ti_l2 = ti_l2.transpose(1, 2)
        rgb_l = rgb_l.transpose(1, 2)


        #print("rgb_l:",rgb_l.shape)
        key_pre_rgb = self.module_regressor_rgb(rgb_l, batch_size, length_size)
        key_pre_ti = self.module_regressor_ti(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module_regressor_ti(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        rgb_fusion = torch.cat((rgb_l, rgb_l_fusion,ti_l), dim=2)
        ti_fusion = torch.cat((rgb_l, ti_l_fusion, ti_l), dim=2)
        key_pre_rgb_fusion = self.module_regressor_fusion(rgb_fusion, batch_size, length_size)
        key_pre_ti_fusion = self.module_regressor_fusion(ti_fusion, batch_size, length_size)
        key_pre_rgb_fusion = key_pre_rgb_fusion.view(batch_size * length_size, 24, 3)
        key_pre_ti_fusion = key_pre_ti_fusion.view(batch_size * length_size, 24, 3)

        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,\
               rgb_l, ti_l, ti_l2,key_pre_rgb_fusion,key_pre_ti_fusion

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class mid_modal_hmr_train_len25(nn.Module):
    def __init__(self,device2):
        super(mid_modal_hmr_train_len25, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
                '''
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device2)
        '''
        self.model_ti2.load(
            './log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(
                2999))
        '''
        self.model_tcmr = TCMR(
            seqlen=25,
            n_layers=2,
            hidden_size=1024).to(device2)
        # print(model)
        pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
        ckpt = torch.load(pretrained_file)
        print(f"Load pretrained weights from \'{pretrained_file}\'")
        ckpt = ckpt['gen_state_dict']
        self.model_tcmr.load_state_dict(ckpt, strict=False)

        self.module3 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

    def forward(self, x_rgb,ti_p,ti_n,h0, c0, batch_size,length_size):
        #print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        feature_tcmr, _ = self.model_tcmr(feature_hmr)
        #mmwave网络
        g_vec_h, a_vec_h, _ = self.model_ti(ti_p, h0, c0,  batch_size, length_size)
        ti_h=torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _ = self.model_ti2(ti_p, h0, c0,  batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _ = self.model_ti(ti_n, h0, c0,  batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _ = self.model_ti2(ti_n, h0, c0,  batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        #rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        #取单独一帧的低特征做判断
        rgb_l_single_frame = rgb_l[:,5,:]
        ti_l_single_frame = ti_l[:, 5, :]
        ti_l2_single_frame = ti_l2[:, 5, :]
        #midmodal预测
        #print("rgb_l:",rgb_l.shape)
        #print("ti_l:", ti_l.shape)
        #print("ti_h:", ti_h.shape)
        key_pre_rgb = self.module3(t_vec[:,:,:256], batch_size, length_size)
        key_pre_ti = self.module3(ti_l, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        #总输出
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        #print("output1:", output1.shape)
        #print("output2:", output2.shape)

        return rgb_h,ti_h,ti_h2,key_pre_rgb,key_pre_ti,key_pre_ti2,output1,output2,output3,rgb_l,ti_l,ti_l2

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#crossmodalmidmodal重建后直接判断
class mid_modal_discriminator(nn.Module):
    def __init__(self,device2,ifrnn=0):
        super(mid_modal_discriminator, self).__init__()
        if ifrnn==1:
            self.fc1 = nn.Linear(24*3*10*2, 256)
            self.cb1 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 64)
            self.cb2 = nn.BatchNorm1d(64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 2)
            self.relu3 = nn.Softmax()
            self.caf1 = nn.ReLU()
            self.rnn = nn.LSTM(input_size=72, hidden_size=72, num_layers=3, batch_first=True, dropout=0.1,
                               bidirectional=False)
        else:
            self.fc1 = nn.Linear(24 * 3 * 10*2, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 2)
            self.relu3 = nn.Softmax()
            self.caf1 = nn.ReLU()

    def forward(self, x1,x2,batchsize,h0, c0,ifrnn=0):
        if ifrnn==1:
            x1 = x1.view(batchsize, 10, 24*3)
            x2 = x2.view(batchsize, 10, 24*3)
            x1, (hn, cn) = self.rnn(x1, (h0, c0))
            x2, (hn, cn) = self.rnn(x2, (h0, c0))
            #print("x1:",x1.shape)
            #print("x2:",x1.shape)

            x = torch.cat((x1,x2),dim=1)
            x = x.view(batchsize,24*3*10*2)
            x =  self.relu1( self.cb1(self.fc1(x)))
            x = self.relu2(self.cb2(self.fc2(x)))
            out = self.relu3(self.fc3(x))
        else:
            x = torch.cat((x1, x2), dim=1)
            x = x.view(batchsize, 24 * 3 * 10 * 2 )
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            out = self.relu3(self.fc3(x))
        return out

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

#像素级attention
class _NonLocalBlockND_2modules_pixelatten(nn.Module):
    def __init__(self, in_channels, inter_channels=None, selfrgb=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_pixelatten, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.selfrgb = selfrgb

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if selfrgb==1:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2))
            bn = nn.BatchNorm2d
            conv_nd2 = nn.Conv1d
            max_pool_layer2 = nn.MaxPool1d(kernel_size=(2))
            bn2 = nn.BatchNorm1d
        elif selfrgb==0:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
            conv_nd2 = nn.Conv2d
            max_pool_layer2 = nn.MaxPool2d(kernel_size=(2))
            bn2 = nn.BatchNorm2d


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                               stride=1, padding=0)

        self.F_phi = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                                stride=1,
                                padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.self_bnRelu = nn.Sequential(
            bn(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.mutual_bnRelu = nn.Sequential(
            bn2(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

            # 拼接后进行映射
            self.C_W = nn.Sequential(
                conv_nd(in_channels=(self.in_channels) * 2, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.C_W[1].weight, 0)
            nn.init.constant_(self.C_W[1].bias, 0)

        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea,return_nl_map=False):
            """
            :param x: (b, c, t, h, w)
            :param return_nl_map: if True return z, nl_map, else only return z.
            :return:
            """
            #print("self_fea:", self_fea.shape)
            selfNonLocal_fea = self.self_bnRelu(self_fea)
            mutualNonLocal_fea = self.mutual_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            # using self feature to generate attention
            self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            #print("self_g_x:", self_g_x.shape)
            self_g_x = self_g_x.permute(0, 2, 1)
            # self_g_x = F.normalize(self_g_x, dim=2)
            # print("self_g_x:", torch.mean(self_g_x))
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            self_f_div_C = F.softmax(self_f, dim=-1)
            #print("self_f_div_C:",self_f_div_C.shape)

            '''
            print("self_attention:")
            print("self_f_div_C:",self_f_div_C.shape)
            if self.selfrgb == 1:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    plt.figure(figsize=(12, 12))
                    plot = sns.heatmap(mutual_f_div_C[i][j ].cpu().detach().reshape(14,14).detach(), linewidths=0.8, annot=True, fmt=".3f")
                    # plt.pause(1.3)
                    # print(ax.lines)
                    plt.show()
'''
            self_y = torch.matmul(self_f_div_C, self_g_x)
            #print("self_f_div_C:", self_f_div_C.shape)
            #print("self_g_x:", self_g_x.shape)
            #print("self_y:", self_y.shape)
            self_y = self_y.permute(0, 2, 1).contiguous()
            #print("self_y:",self_y.shape)
            self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            #print("self_y:", self_y.shape)
            # 只映射最终结果，即self和mutual的结果相加后映射
            self_W_y = self.R_W(self_y)

            # using mutual feature to generate attention
            mutual_g_x = self.F_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_g_x = mutual_g_x.permute(0, 2, 1)
            # print("mutual_g_x:", mutual_g_x.shape)
            # mutual_g_x = F.normalize(mutual_g_x,dim=2)
            # print("mutual_g_x:",torch.mean(mutual_g_x))
            mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
            mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
            mutual_f_div_C = F.softmax(mutual_f, dim=-1)
            '''
            print("mutual_attention:")
            print("mutual_f_div_C:", mutual_f_div_C.shape)
            if self.selfrgb==0:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    for j in range(1):
                        plt.figure(figsize=(12, 12))
                        #plot = sns.heatmap(mutual_f_div_C[i][j+30].cpu().detach().unsqueeze(1), linewidths=0.8, annot=True, fmt=".3f")
                        plot = sns.heatmap(mutual_f_div_C[i][j+10 ].cpu().detach().reshape(14,14), linewidths=0.8,
                                           annot=True, fmt=".3f")
                        # plt.pause(1.3)
                        # print(ax.lines)
                        plt.show()
'''
            #print("mutual_f_div_C:", mutual_f_div_C.shape)
            #print("mutual_g_x:", mutual_g_x.shape)
            mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
            mutual_y = mutual_y.permute(0, 2, 1).contiguous()
            mutual_y = mutual_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            mutual_W_y = self.F_W(mutual_y)
            #print("mutual_W_y:", mutual_W_y.shape)
            # print("f_image:",f[0])
            '''
            #0502修改
            z = mutual_y + self_y
            z = self.F_W(z)
            '''
            # 0503修改，拼接做法
            z = torch.cat((self_W_y, mutual_W_y), dim=1)
            #0505修改：拼接后直接返回
            z = self.C_W(z)
            #print("z:", z.shape)

            if return_nl_map:
                return z, self_f_div_C, mutual_f_div_C
            return z,self_W_y,mutual_f_div_C


#使用残差结构
class _NonLocalBlockND_2modules_pixelatten_res(nn.Module):
    def __init__(self, in_channels, inter_channels=None, selfrgb=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_pixelatten_res, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.selfrgb = selfrgb

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if selfrgb==1:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2))
            bn = nn.BatchNorm2d
            conv_nd2 = nn.Conv1d
            max_pool_layer2 = nn.MaxPool1d(kernel_size=(2))
            bn2 = nn.BatchNorm1d
        elif selfrgb==0:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
            conv_nd2 = nn.Conv2d
            max_pool_layer2 = nn.MaxPool2d(kernel_size=(2))
            bn2 = nn.BatchNorm2d


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                               stride=1, padding=0)

        self.F_phi = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                                stride=1,
                                padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.self_bnRelu = nn.Sequential(
            bn(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.mutual_bnRelu = nn.Sequential(
            bn2(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

            # 拼接后进行映射
            self.C_W = nn.Sequential(
                conv_nd(in_channels=(self.in_channels) * 2, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.C_W[1].weight, 0)
            nn.init.constant_(self.C_W[1].bias, 0)

        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea,return_nl_map=False):
            """
            :param x: (b, c, t, h, w)
            :param return_nl_map: if True return z, nl_map, else only return z.
            :return:
            """
            #print("self_fea:", self_fea.shape)
            selfNonLocal_fea = self.self_bnRelu(self_fea)
            mutualNonLocal_fea = self.mutual_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            # using self feature to generate attention
            self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            #print("self_g_x:", self_g_x.shape)
            self_g_x = self_g_x.permute(0, 2, 1)
            # self_g_x = F.normalize(self_g_x, dim=2)
            # print("self_g_x:", torch.mean(self_g_x))
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            self_f_div_C = F.softmax(self_f, dim=-1)
            #print("self_f_div_C:",self_f_div_C.shape)

            '''
            print("self_attention:")
            print("self_f_div_C:",self_f_div_C.shape)
            if self.selfrgb == 1:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    plt.figure(figsize=(12, 12))
                    plot = sns.heatmap(mutual_f_div_C[i][j ].cpu().detach().reshape(14,14).detach(), linewidths=0.8, annot=True, fmt=".3f")
                    # plt.pause(1.3)
                    # print(ax.lines)
                    plt.show()
'''
            self_y = torch.matmul(self_f_div_C, self_g_x)
            #print("self_f_div_C:", self_f_div_C.shape)
            #print("self_g_x:", self_g_x.shape)
            #print("self_y:", self_y.shape)
            self_y = self_y.permute(0, 2, 1).contiguous()
            #print("self_y:",self_y.shape)
            self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            #print("self_y:", self_y.shape)
            # 只映射最终结果，即self和mutual的结果相加后映射
            self_W_y = self.R_W(self_y)

            # using mutual feature to generate attention
            mutual_g_x = self.F_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_g_x = mutual_g_x.permute(0, 2, 1)
            # print("mutual_g_x:", mutual_g_x.shape)
            # mutual_g_x = F.normalize(mutual_g_x,dim=2)
            # print("mutual_g_x:",torch.mean(mutual_g_x))
            mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
            mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
            mutual_f_div_C = F.softmax(mutual_f, dim=-1)
            '''
            print("mutual_attention:")
            print("mutual_f_div_C:", mutual_f_div_C.shape)
            if self.selfrgb==0:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    for j in range(1):
                        plt.figure(figsize=(12, 12))
                        #plot = sns.heatmap(mutual_f_div_C[i][j+30].cpu().detach().unsqueeze(1), linewidths=0.8, annot=True, fmt=".3f")
                        plot = sns.heatmap(mutual_f_div_C[i][j+10 ].cpu().detach().reshape(14,14), linewidths=0.8,
                                           annot=True, fmt=".3f")
                        # plt.pause(1.3)
                        # print(ax.lines)
                        plt.show()
            '''
            #print("mutual_f_div_C:", mutual_f_div_C.shape)
            #print("mutual_g_x:", mutual_g_x.shape)
            mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
            mutual_y = mutual_y.permute(0, 2, 1).contiguous()
            mutual_y = mutual_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            mutual_W_y = self.F_W(mutual_y)
            #print("mutual_W_y:", mutual_W_y.shape)
            # print("f_image:",f[0])
            '''
            #0502修改
            z = mutual_y + self_y
            z = self.F_W(z)
            '''
            # 0503修改，拼接做法
            z = torch.cat((self_W_y, mutual_W_y), dim=1)
            #0505修改：拼接后直接返回
            z = self.C_W(z)
            #print("z:", z.shape)
            z = z+self_fea
            if return_nl_map:
                return z, self_f_div_C, mutual_f_div_C
            return z,self_W_y,mutual_f_div_C

#0517修改：0510+仅训练重建网络
class mid_modal_hmr_train_2nln_loc_0510(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_2nln_loc_0510, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_512()
        self.module5 = CombineModule_mid_modal()


        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(512)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0505(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0505(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr = feature_hmr.view(batch_size, length_size, 2048)
        #hmr only
        feature_tcmr = feature_hmr
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        f_tcmr = feature_tcmr.transpose(1, 2)
        print("feature_tcmr:",f_tcmr.shape)
        f_tcmr = self.caf1(self.cb1(self.conv1(f_tcmr)))
        t_vec = f_tcmr.transpose(1, 2)
        rgb_h = t_vec[:, :, 256:]
        rgb_l = t_vec[:, :, :256]

        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_l.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)

        #2nln,每个nln都计算self和mutual
        print("rgb_nln:")
        rgb_fusion,rgb_self = self.nl(rgb_l_ma,ti_l_ma)
        print("mmwave_nln:")
        ti_fusion,ti_self = self.nl2(ti_l_ma, rgb_l_ma)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_self = rgb_self.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)


        '''
        # nln模块更正0423
        rgb_l_ma = rgb_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        ti_l_ma = ti_l_ma.view(batch_size, length_size, 1, 256).repeat(1, 1, 256, 1)
        feature_ma = torch.cat([rgb_l_ma, ti_l_ma], 3)
        feature_ma = feature_ma.view(batch_size * length_size, 256, -1)
        feature_ma = feature_ma.transpose(1, 2)
        feature_ma = self.caf2(self.cb2(self.conv2(feature_ma)))

        nl_out = self.nl(feature_ma)
        nl_out = nl_out.transpose(1, 2).contiguous()
        attn_weights_rgb = self.softmax1(self.attn1(nl_out))
        attn_weights_ti = self.softmax2(self.attn2(nl_out))
        rgb_fusion = torch.sum(nl_out * attn_weights_rgb, dim=1)
        ti_fusion = torch.sum(nl_out * attn_weights_ti, dim=1)
        '''

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)
        # 总输出

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_l))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_l))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_l * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_l * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0628修改：mutual attention
class mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14_1regressor(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14_1regressor, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        #ti_fusion = self.caf5(self.cb5(self.conv5(ti_fusion)))
        #rgb_fusion = self.caf6(self.cb6(self.conv6(rgb_fusion)))
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_fusion = rgb_fusion.transpose(1, 2)

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#0628修改：mutual attention
class mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_2regressor_featuremap14, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, _, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_fusion = rgb_fusion.transpose(1, 2)
        #ti_fusion = self.caf5(self.cb5(self.conv5(ti_fusion)))
        #rgb_fusion = self.caf6(self.cb6(self.conv6(rgb_fusion)))
        ti_fusion = ti_fusion.transpose(1, 2)
        rgb_fusion = rgb_fusion.transpose(1, 2)

        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_l2))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_l2 * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#nln使用残差结构
class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)

        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)

        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#全排列组合
class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_skeleton(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_skeleton, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        #不同id融合结果
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_rgb_dif = self.module4(rgb_fusion_n_dif, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_rgb_dif = key_pre_rgb_dif.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)
        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)



        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif,key_pre_rgb_dif

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#输出全排列骨架
class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        #不同id融合结果
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)

        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_woh(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_woh, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        #print(x_rgb.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)

        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        #不同id融合结果
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)

        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)
        # 在高低维特征norm前整体norm
        output1 = rgb_l
        output2 = ti_l
        output3 = ti_l2

        rgb_h = 0

        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_noatten(nn.Module):
    def __init__(self, device2):
        super(mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_noatten, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_h = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_h.load_state_dict(checkpoint['model'], strict=False)

        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr_h.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)

        # 总输出
        '''
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_fusion, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_fusion_n, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_fusion, start_dim=1, end_dim=2)


        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#stgcn
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    # def __init__(self, in_channels, num_class, graph_args,
    #              edge_importance_weighting, **kwargs):
    def __init__(self, in_channels, graph_args,
                     edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        #self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        #x = self.fcn(x)
        x = x.view(x.size(0), -1)
        #x = F.normalize(x)
        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class mid_modal_st_gcn(nn.Module):
    def __init__(self, device2,ifst):
        super(mid_modal_st_gcn, self).__init__()
        #是否分开统计s、t
        self.ifst = ifst
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)
        #不训练hmr
        #self.model_hmr_l.eval()

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

        #st_gcn
        in_channels = 3
        # graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        graph_args = {'layout': 'kinect-19', 'strategy': 'spatial'}
        edge_importance_weighting = True
        self.mmwave_model = Model(in_channels, graph_args, edge_importance_weighting).to(device2)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        #print(x_rgb.size())
        #with torch.no_grad():
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)

        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        #不同id融合结果
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        if self.ifst==1:
            trans_rgb = key_pre_rgb[:, 0].unsqueeze(1).repeat(1, 24, 1)
            trans_ti = key_pre_ti[:, 0].unsqueeze(1).repeat(1, 24, 1)
            trans_ti2 = key_pre_ti2[:, 0].unsqueeze(1).repeat(1, 24, 1)
            key_pre_rgb = key_pre_rgb - trans_rgb
            key_pre_ti = key_pre_ti - trans_ti
            key_pre_ti2 = key_pre_ti2 - trans_ti2

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)

        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)
        # 在高低维特征norm前整体norm
        output1 = rgb_l
        output2 = ti_l
        output3 = ti_l2

        rgb_h = 0

        joint_indices = [0, 3, 6, 10, 11, 14, 16, 18, 12, 15, 17, 19, 1, 4, 7, 2, 5, 8, 13]
        data_key_a = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_p = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_n = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1)[:, joint_indices, :]

        N, C, T, V, M = batch_size, 3, length_size, len(joint_indices), 1
        data_key_a = data_key_a.view(N, T, V, C, M)
        data_key_a = data_key_a.permute(0, 3, 1, 2, 4).contiguous()
        data_key_p = data_key_p.view(N, T, V, C, M)
        data_key_p = data_key_p.permute(0, 3, 1, 2, 4).contiguous()
        data_key_n = data_key_n.view(N, T, V, C, M)
        data_key_n = data_key_n.permute(0, 3, 1, 2, 4).contiguous()
        r_a = self.mmwave_model(data_key_a)
        r_p = self.mmwave_model(data_key_p)
        r_n = self.mmwave_model(data_key_n)

        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif,r_a,r_p,r_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#RGB估计二维姿态，融合后升至三维
class mid_modal_RGB2D(nn.Module):
    def __init__(self, device2,ifst):
        super(mid_modal_RGB2D, self).__init__()
        #是否分开统计s、t
        self.ifst = ifst
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr_l = hmr().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)
        #不训练hmr
        #self.model_hmr_l.eval()

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)


        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal_2d()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=20 * 2, out_channels=512, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(512)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(512)
        self.caf3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0503(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0503(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

        #st_gcn
        in_channels = 3
        # graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        graph_args = {'layout': 'kinect-19', 'strategy': 'spatial'}
        edge_importance_weighting = True
        self.mmwave_model = Model(in_channels, graph_args, edge_importance_weighting).to(device2)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        #print(x_rgb.size())
        #with torch.no_grad():
        feature_hmr_l = self.model_hmr_l.feature_extractor(x_rgb)
        feature_hmr_l = feature_hmr_l.view(batch_size, length_size, 2048)
        feature_hmr_l = feature_hmr_l.transpose(1, 2)
        feature_hmr_l_n= self.model_hmr_l.feature_extractor(x_rgb_n)
        feature_hmr_l_n = feature_hmr_l_n.view(batch_size, length_size, 2048)
        feature_hmr_l_n = feature_hmr_l_n.transpose(1, 2)


        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        #print("feature_hmr_l:", feature_hmr_l.shape)
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        key_pre_rgb_2d = self.module5(rgb_l, batch_size, length_size)
        key_pre_rgb_2d = key_pre_rgb_2d.view(batch_size , length_size, 20*2)
        rgb_2d_feature = key_pre_rgb_2d.transpose(1, 2)

        key_pre_rgb_2d_n = self.module5(rgb_l_n, batch_size, length_size)
        key_pre_rgb_2d_n = key_pre_rgb_2d_n.view(batch_size, length_size, 20 * 2)
        rgb_2d_feature_n = key_pre_rgb_2d_n.transpose(1, 2)

        #print("rgb_2d_feature:",rgb_2d_feature.shape)
        rgb_2d_feature = self.caf2(self.cb2(self.conv2(rgb_2d_feature)))
        rgb_2d_feature = self.caf3(self.cb3(self.conv3(rgb_2d_feature)))
        rgb_2d_feature = self.caf4(self.cb4(self.conv4(rgb_2d_feature)))

        rgb_2d_feature_n = self.caf2(self.cb2(self.conv2(rgb_2d_feature_n)))
        rgb_2d_feature_n = self.caf3(self.cb3(self.conv3(rgb_2d_feature_n)))
        rgb_2d_feature_n = self.caf4(self.cb4(self.conv4(rgb_2d_feature_n)))

        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_2d_feature.view(batch_size, length_size, 256)
        ti_l_n = ti_l2.view(batch_size, length_size, 256)
        rgb_l_n = rgb_2d_feature_n.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)
        ti_l_ma_n = ti_l_n.transpose(1, 2)
        rgb_l_ma_n = rgb_l_n.transpose(1, 2)

        # 2nln,每个nln都计算self和mutual
        rgb_fusion = self.nl(rgb_l_ma, ti_l_ma)
        ti_fusion = self.nl2(ti_l_ma, rgb_l_ma)
        ti_fusion_dif = self.nl2(ti_l_ma, rgb_l_ma_n)
        rgb_fusion_dif = self.nl(rgb_l_ma, ti_l_ma_n)
        rgb_fusion_n = self.nl(rgb_l_ma_n, ti_l_ma_n)
        ti_fusion_n = self.nl2(ti_l_ma_n, rgb_l_ma_n)
        rgb_fusion_n_dif = self.nl(rgb_l_ma_n, ti_l_ma)

        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        rgb_fusion_dif = rgb_fusion_dif.transpose(1, 2)
        rgb_fusion_n = rgb_fusion_n.transpose(1, 2)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        rgb_fusion_n_dif = rgb_fusion_n_dif.transpose(1, 2)



        # reconstruction
        #key_pre_rgb_self = self.module5(rgb_l, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_l, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        # key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        key_pre_rgb_2d = key_pre_rgb_2d.view(batch_size * length_size, 20, 2)



        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)


        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)
        # 在高低维特征norm前整体norm
        output1 = rgb_l
        output2 = ti_l
        output3 = ti_l2

        rgb_h = 0

        joint_indices = [0, 3, 6, 10, 11, 14, 16, 18, 12, 15, 17, 19, 1, 4, 7, 2, 5, 8, 13]
        data_key_a = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_p = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_n = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1)[:, joint_indices, :]

        N, C, T, V, M = batch_size, 3, length_size, len(joint_indices), 1
        data_key_a = data_key_a.view(N, T, V, C, M)
        data_key_a = data_key_a.permute(0, 3, 1, 2, 4).contiguous()
        data_key_p = data_key_p.view(N, T, V, C, M)
        data_key_p = data_key_p.permute(0, 3, 1, 2, 4).contiguous()
        data_key_n = data_key_n.view(N, T, V, C, M)
        data_key_n = data_key_n.permute(0, 3, 1, 2, 4).contiguous()
        r_a = self.mmwave_model(data_key_a)
        r_p = self.mmwave_model(data_key_p)
        r_n = self.mmwave_model(data_key_n)

        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_2d,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif,r_a,r_p,r_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class mid_modal_kp2d_hmr(nn.Module):
    def __init__(self, device2,ifst):
        super(mid_modal_kp2d_hmr, self).__init__()
        #是否分开统计s、t
        self.ifst = ifst

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)


        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        #self.module5 = CombineModule_mid_modal()

        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=21 * 2, out_channels=512, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(512)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(512)
        self.caf3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_0503(in_channels=256)
        self.nl2 = _NonLocalBlockND_2modules_0503(in_channels=256)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

        #st_gcn
        in_channels = 3
        # graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        graph_args = {'layout': 'kinect-19', 'strategy': 'spatial'}
        edge_importance_weighting = True
        self.mmwave_model = Model(in_channels, graph_args, edge_importance_weighting).to(device2)

    def forward(self, kp2d,kp2d_n, ti_p, ti_n, h0, c0, batch_size, length_size):


        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)


        key_pre_rgb_2d = kp2d.view(batch_size , length_size, 21*2)
        rgb_2d_feature = key_pre_rgb_2d.transpose(1, 2)

        key_pre_rgb_2d_n = kp2d_n.view(batch_size, length_size, 21 * 2)
        rgb_2d_feature_n = key_pre_rgb_2d_n.transpose(1, 2)

        #print("rgb_2d_feature:",rgb_2d_feature.shape)
        rgb_2d_feature = self.caf2(self.cb2(self.conv2(rgb_2d_feature)))
        rgb_2d_feature = self.caf3(self.cb3(self.conv3(rgb_2d_feature)))
        rgb_2d_feature = self.caf4(self.cb4(self.conv4(rgb_2d_feature)))

        rgb_2d_feature_n = self.caf2(self.cb2(self.conv2(rgb_2d_feature_n)))
        rgb_2d_feature_n = self.caf3(self.cb3(self.conv3(rgb_2d_feature_n)))
        rgb_2d_feature_n = self.caf4(self.cb4(self.conv4(rgb_2d_feature_n)))

        ti_l = ti_l.view(batch_size, length_size, 256)
        rgb_l = rgb_2d_feature.view(batch_size, length_size, 256)
        ti_l_n = ti_l2.view(batch_size, length_size, 256)
        rgb_l_n = rgb_2d_feature_n.view(batch_size, length_size, 256)
        ti_l_ma = ti_l.transpose(1, 2)
        rgb_l_ma = rgb_l.transpose(1, 2)
        ti_l_ma_n = ti_l_n.transpose(1, 2)
        rgb_l_ma_n = rgb_l_n.transpose(1, 2)

        # 2nln,每个nln都计算self和mutual
        rgb_fusion = self.nl(rgb_l_ma, ti_l_ma)
        ti_fusion = self.nl2(ti_l_ma, rgb_l_ma)
        ti_fusion_dif = self.nl2(ti_l_ma, rgb_l_ma_n)
        rgb_fusion_dif = self.nl(rgb_l_ma, ti_l_ma_n)
        rgb_fusion_n = self.nl(rgb_l_ma_n, ti_l_ma_n)
        ti_fusion_n = self.nl2(ti_l_ma_n, rgb_l_ma_n)
        rgb_fusion_n_dif = self.nl(rgb_l_ma_n, ti_l_ma)

        rgb_fusion = rgb_fusion.transpose(1, 2)
        ti_fusion = ti_fusion.transpose(1, 2)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        rgb_fusion_dif = rgb_fusion_dif.transpose(1, 2)
        rgb_fusion_n = rgb_fusion_n.transpose(1, 2)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        rgb_fusion_n_dif = rgb_fusion_n_dif.transpose(1, 2)



        # reconstruction
        #key_pre_rgb_self = self.module5(rgb_l, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_l, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_l2, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        # key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        key_pre_rgb_2d = key_pre_rgb_2d.view(batch_size * length_size, 21, 2)



        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)


        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_l = F.normalize(rgb_l)
        rgb_l2 = F.normalize(rgb_l2)
        rgb_l_dif = F.normalize(rgb_l_dif)
        ti_dif = F.normalize(ti_dif)
        # 在高低维特征norm前整体norm
        output1 = rgb_l
        output2 = ti_l
        output3 = ti_l2

        rgb_h = 0

        joint_indices = [0, 3, 6, 10, 11, 14, 16, 18, 12, 15, 17, 19, 1, 4, 7, 2, 5, 8, 13]
        data_key_a = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_p = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1)[:, joint_indices, :]
        data_key_n = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1)[:, joint_indices, :]

        N, C, T, V, M = batch_size, 3, length_size, len(joint_indices), 1
        data_key_a = data_key_a.view(N, T, V, C, M)
        data_key_a = data_key_a.permute(0, 3, 1, 2, 4).contiguous()
        data_key_p = data_key_p.view(N, T, V, C, M)
        data_key_p = data_key_p.permute(0, 3, 1, 2, 4).contiguous()
        data_key_n = data_key_n.view(N, T, V, C, M)
        data_key_n = data_key_n.permute(0, 3, 1, 2, 4).contiguous()
        r_a = self.mmwave_model(data_key_a)
        r_p = self.mmwave_model(data_key_p)
        r_n = self.mmwave_model(data_key_n)

        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_2d,key_pre_ti_self,key_pre_ti_self_n,rgb_l2,rgb_l_dif,ti_dif,r_a,r_p,r_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#一个hmr拆分
class mid_modal_1hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb(nn.Module):
    def __init__(self, device2):
        super(mid_modal_1hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb, self).__init__()
        BASE_DATA_DIR = './lib/models/pretrained/base_data'
        self.model_hmr = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)



    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # print(x_ti.size())
        feature_hmr_l,_ = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr_l_n, _ = self.model_hmr.feature_extractor(x_rgb_n)
        _,feature_hmr_h = self.model_hmr.feature_extractor(x_rgb)
        feature_hmr_h = feature_hmr_h.view(batch_size, length_size, 2048)
        #feature_tcmr, _ = self.model_tcmr(feature_hmr)
        # mmwave网络
        g_vec_h, a_vec_h, _, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        g_vec_h2, a_vec_h2, _, g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))
        feature_hmr_h = feature_hmr_h.transpose(1, 2)
        #feature_hmr_h:[4, 2048, 20]
        #print("feature_hmr_h:",feature_hmr_h.shape)
        rgb_h = self.caf4(self.cb4(self.conv4(feature_hmr_h)))
        rgb_h = rgb_h.transpose(1, 2)
        #t_vec = f_tcmr.transpose(1, 2)


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)
        n_pts = ti_p.size()[1]
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))

        ti_l2 = ti_l2.view(batch_size, length_size, 256)
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)
        rgb_self = self.avgpool(rgb_self)
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        ti_fusion = ti_fusion.transpose(1, 2)
        ti_self = ti_self.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)

        ti_fusion_n = ti_fusion_n.transpose(1, 2)
        ti_self_n = ti_self_n.transpose(1, 2)
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module5(rgb_self, batch_size, length_size)
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)
        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)
        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_h = self.softmax6(self.attn6(rgb_h))
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        rgb_h = torch.sum(rgb_h * attn_weights_rgb_h, dim=1)
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        '''
        #没有attention
        ti_h = torch.flatten(ti_h, start_dim=1, end_dim=2)
        ti_l = torch.flatten(ti_l, start_dim=1, end_dim=2)
        ti_h2 = torch.flatten(ti_h2, start_dim=1, end_dim=2)
        ti_l2 = torch.flatten(ti_l2, start_dim=1, end_dim=2)
        rgb_h = torch.flatten(rgb_h, start_dim=1, end_dim=2)
        rgb_l = torch.flatten(rgb_l, start_dim=1, end_dim=2)
        '''

        # 在高低维特征norm前整体norm
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output1 = F.normalize(output1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output2 = F.normalize(output2)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        output3 = F.normalize(output3)

        ti_h = F.normalize(ti_h)
        ti_l = F.normalize(ti_l)
        ti_h2 = F.normalize(ti_h2)
        ti_l2 = F.normalize(ti_l2)
        rgb_h = F.normalize(rgb_h)
        rgb_l = F.normalize(rgb_l)
        '''
        #直接使用分别对高低维featurenorm的结果
        output1 = torch.cat((rgb_h, rgb_l), dim=1)
        output2 = torch.cat((ti_h, ti_l), dim=1)
        output3 = torch.cat((ti_h2, ti_l2), dim=1)
        '''

        # print("output1:", output1.shape)
        # print("output2:", output2.shape)

        return rgb_h, ti_h, ti_h2, key_pre_rgb, key_pre_ti, key_pre_ti2, output1, output2, output3, rgb_l, ti_l, ti_l2, g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

#对抗训练
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='nl2.C_W.0'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                #print("param.grad:",param.grad)
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='nl2.C_W.0'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



#3d->2dprojection
def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

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

def draw3Dpose_frames(rgb ,ti_p,data_key_rgb,data_key_tip):
    # 绘制连贯的骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    i = 0
    j = 0
    while i < ti_p.shape[0]:
        draw3Dpose(ti_p[i],rgb[i], data_key_rgb[i],data_key_tip[i], ax)
        plt.pause(0.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1
        if i == ti_p.shape[0]:
            # i=0
            j += 1
        if j == 2:
            break

    plt.ioff()
    plt.show()

if __name__=='__main__':
    cudanum = 0
    if torch.cuda.is_available():
        device = 'cuda:%d' % (cudanum)
    else:
        device = 'cpu'
    model = mid_modal_hmr_train_pixelatten_loc_1regressor_featuremap14_2rgb_nlnres_allfusion_woh(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    fgm = FGM(model)
    emb_name = 'model_ti2.module1.gpointnet'

    for name, param in model.named_parameters():
        print("name:",name)
        #if param.requires_grad and emb_name in name:
            #print("param:",  param.data)
  # 对抗训练
    fgm.attack()  # embedding被修改了
    fgm.restore()





