#NLN网络
import torch
import torch.nn as nn
import torch.nn.functional as F
#from non_local_dot_product import NONLocalBlock1D
import time

class NONLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__()

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
        print("g_x:",g_x.shape)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        print("theta_x:", theta_x.shape)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        print("phi_x:", phi_x.shape)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        print("f", f.shape)
        N = f.size(-1)
        print("N:", N)
        f_div_C = f / N
        print("f_div_C:", f_div_C.shape)

        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        print("y:", y.shape)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class BasePointParsingNet(nn.Module):
    def __init__(self):
        super(BasePointParsingNet, self).__init__()
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
        x = in_mat.transpose(1,2)   #转置       # x:(91, 6, 50) point(x,y,z,range,intensity,velocity)

        x = self.caf1(self.cb1(self.conv1(x)))  # x:(91, 8, 50)
        x = self.caf2(self.cb2(self.conv2(x)))  # x:(91, 16, 50)
        x = self.caf3(self.cb3(self.conv3(x)))  # x:(91, 24, 50)

        x = x.transpose(1,2)  # x:(91, 50, 24)
        x = torch.cat((in_mat[:,:,:3], x), -1)    # x:(91, 50, 28)  拼接了x,y,z,range

        return x

class GlobalPointParsingNet(nn.Module):
    def __init__(self):
        super(GlobalPointParsingNet, self).__init__()

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

class GlobalParsingRNN(nn.Module):
    def __init__(self):
        super(GlobalParsingRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x, h0, c0):
        # x:[7, 13, 64]    h0:[3, 7, 64]   c0:[3, 7, 64]
        g_vec, (hn, cn)=self.rnn(x, (h0, c0)) # g_vec:[7, 13, 64] hn:[3, 7, 64] cn:[3, 7, 64]

        return g_vec, hn, cn

class GlobalParsingModule(nn.Module):
    def __init__(self):
        super(GlobalParsingModule, self).__init__()
        self.bpointnet=BasePointParsingNet()
        self.gpointnet=GlobalPointParsingNet()
        self.grnn=GlobalParsingRNN()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x=self.bpointnet(x)
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, hn, cn=self.grnn(x, h0, c0)
        return g_vec, attn_weights, hn, cn

class BasePointPoseNet(nn.Module):
    def __init__(self):
        super(BasePointPoseNet, self).__init__()
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
        x = in_mat.transpose(1,2)   #转置       # x:(91, 6, 50) point(x,y,z,range,intensity,velocity)

        x = self.caf1(self.cb1(self.conv1(x)))  # x:(91, 8, 50)
        x = self.caf2(self.cb2(self.conv2(x)))  # x:(91, 16, 50)
        x = self.caf3(self.cb3(self.conv3(x)))  # x:(91, 24, 50)

        x = x.transpose(1,2)  # x:(91, 50, 24)
        x = torch.cat((in_mat[:,:,:3], x), -1)    # x:(91, 50, 28)  拼接了x,y,z,range

        return x

class GlobalPointPoseNet(nn.Module):
    def __init__(self):
        super(GlobalPointPoseNet, self).__init__()

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

class GlobalPoseRNN(nn.Module):
    def __init__(self):
        super(GlobalPoseRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x, h0, c0):
        # x:[7, 13, 64]    h0:[3, 7, 64]   c0:[3, 7, 64]
        g_vec, (hn, cn)=self.rnn(x, (h0, c0)) # g_vec:[7, 13, 64] hn:[3, 7, 64] cn:[3, 7, 64]

        return g_vec, hn, cn

class GlobalPoseModule(nn.Module):
    def __init__(self):
        super(GlobalPoseModule, self).__init__()
        self.bpointnet=BasePointPoseNet()
        self.gpointnet=GlobalPointPoseNet()
        self.grnn=GlobalPoseRNN()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x=self.bpointnet(x)
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, hn, cn=self.grnn(x, h0, c0)
        return g_vec, attn_weights, hn, cn

class Parsing_Pose_Net(nn.Module):
    def __init__(self,num_part):
        super(Parsing_Pose_Net, self).__init__()
        self.num_part=num_part
        # parsing
        self.embedding_parsing_net = GlobalParsingModule()
        #parsing
        self.bpointnet = BasePointParsingNet()
        self.conv1=nn.Conv1d(155,64,1) #27+64+64
        self.cb1=nn.BatchNorm1d(64)
        self.caf1=nn.ReLU()

        self.nl=NONLocalBlock1D(in_channels=64)
        self.conv2=nn.Conv1d(64,self.num_part,1)

        #pose
        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)
        self.embedding_pose_net = GlobalPoseModule()
        # self.conv1_pose=nn.Conv1d(64+64,64,1)
        # self.cb1_pose=nn.BatchNorm1d(64)
        # self.caf1_pose=nn.ReLU()
        #self.conv2_pose = nn.Conv1d(64, 64, 1)

        #pose
        self.fc1_pose = nn.Linear(64, 64)
        self.faf1_pose = nn.ReLU()
        self.fc2_pose = nn.Linear(64, 17 * 3)

    def forward(self, x1, h0, c0, num_part, batch_size,length_size):
        n_pts=x1.size()[1]
        #parsing
        g_vec_parsing0, attn_weights, hn, cn = self.embedding_parsing_net(x1, h0, c0, batch_size, length_size)
        #pose
        g_vec_pose0, attn_weights, hn, cn = self.embedding_pose_net(x1, h0, c0, batch_size, length_size)

        #parsing
        # g_vec_parsing, attn_weights, hn, cn = self.embedding_parsing_net(x1, h0, c0, batch_size, length_size)
        print("g_vec_parsing0:",g_vec_parsing0.shape)
        g_vec_parsing=g_vec_parsing0.view(batch_size, length_size, 1,64).repeat(1,1,n_pts,1)
        print("g_vec_parsing:", g_vec_parsing.shape)
        g_vec_pose_add = g_vec_pose0.view(batch_size, length_size, 1,64).repeat(1, 1, n_pts,1)

        bpoint = self.bpointnet(x1)
        bpoint=bpoint.view(batch_size,length_size,n_pts,-1)
        bpoint=torch.cat([g_vec_parsing,g_vec_pose_add,bpoint],3) #每一帧拼接LSTM的输出
        print("bpoint:", bpoint.shape)

        bpoint=bpoint.view(batch_size*length_size,n_pts,-1)
        bpoint = bpoint.transpose(1, 2)
        print("bpoint2:", bpoint.shape)
        bpoint = self.caf1(self.cb1(self.conv1(bpoint)))
        print("nln_in:",bpoint.shape)

        nl_out=self.nl(bpoint)
        print("nl_out:", nl_out.shape)
        x=self.conv2(nl_out)
        print("nl_out2:", x.shape)
        x=x.transpose(2,1).contiguous()
        x=F.log_softmax(x.view(-1,self.num_part),dim=-1)
        out_parsing=x.view(batch_size,length_size,n_pts,self.num_part)

        #pose
        # g_vec_pose = torch.cat([g_vec_pose0, g_vec_parsing0], 2)
        #
        # g_vec_pose = g_vec_pose.transpose(1, 2)
        # out_pose = self.caf1_pose(self.cb1_pose(self.conv1_pose(g_vec_pose)))
        # out_pose = out_pose.transpose(2, 1).contiguous()
        # out_pose=self.conv2_pose(out_pose)
        nl_out=nl_out.transpose(1,2).contiguous()
        print("nl_out:", nl_out.shape)
        attn_weights = self.softmax(self.attn(nl_out))  #
        print("attn_weights:", attn_weights.shape)

        print("nl_out * attn_weights:",(nl_out * attn_weights).shape)
        out_pose = torch.sum(nl_out * attn_weights, dim=1)  # * 点乘
        print("out_pose:", out_pose.shape)
        out_pose = self.fc1_pose(out_pose)
        out_pose = self.faf1_pose(out_pose)
        out_pose = self.fc2_pose(out_pose)

        out_pose = out_pose.view(batch_size, -1)

        return out_parsing,out_pose

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

if __name__=='__main__':
    #batch_size*length_size, pt_size, in_feature_size
    # print('BasePointTiNet:')
    # data=torch.rand((1*25, 64, 3), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape)
    # model=BasePointTiNet()
    # x=model(data)
    # print('\tOutput:', x.shape)
    #
    # print('BasePointKinectNet:')
    # data=torch.rand((1*25, 512, 3), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape)
    # model=BasePointKinectNet()
    # x=model(data)
    # print('\tOutput:', x.shape)
    #
    # print('SiameseNet:')
    # data_ti=torch.rand((1*25, 64, 3), dtype=torch.float32, device='cpu')
    # data_kinect = torch.rand((1 * 25, 512, 3), dtype=torch.float32, device='cpu')
    # #print('\tInput:', data.shape)
    # model=SiameseNet()
    # x=model(data_ti,data_kinect)
    # print('\tOutput:', x[0].shape)

    # print('GlobalTiModule:')
    # data=torch.rand((1*25, 64, 3), dtype=torch.float32, device='cpu')
    # h0=torch.zeros((3, 1, 64), dtype=torch.float32, device='cpu')
    # c0=torch.zeros((3, 1, 64), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape, h0.shape, c0.shape)
    # model=GlobalTiModule()
    # x,l,w,hn,cn=model(data,h0,c0,1,25)
    # print('\tOutput:', x.shape, l.shape, w.shape, hn.shape, cn.shape)

    # print('GlobalKinectModule:')
    # data=torch.rand((1*25, 512, 3), dtype=torch.float32, device='cpu')
    # h0=torch.zeros((3, 1, 64), dtype=torch.float32, device='cpu')
    # c0=torch.zeros((3, 1, 64), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape, h0.shape, c0.shape)
    # model=GlobalKinectModule()
    # x,l,w,hn,cn=model(data,h0,c0,1,25)
    # print('\tOutput:', x.shape, l.shape, w.shape, hn.shape, cn.shape)

    batch_size=16
    print('ParsingNet:')
    data_ti = torch.rand((batch_size * 50, 64, 3), dtype=torch.float32)
    #print('\tInput:', data.shape)
    h0=torch.zeros((3, batch_size, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, batch_size, 64), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape, h0.shape, c0.shape)
    model=Parsing_Pose_Net(num_part=3)
    x=model(data_ti,h0,c0,3,batch_size,50)
    print('\tOutput:', x[0].shape)
    #print(model)

