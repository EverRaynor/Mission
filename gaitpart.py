import torch
import torch.nn as nn
import numpy as np
#from ..base_model import BaseModel
#from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs
import copy
import torch.nn.functional as F
from collections import OrderedDict

#为避免路径问题，将一些调用的小函数直接复制至此模型文件
def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret

class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, s, c, p]
          Output: ret, [n, p, c]
        """
        n, s, c, p = x.size()
        x = x.permute(3, 0, 2, 1).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 0, 2).contiguous()  # [n, p, c]
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

#HP层
class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out

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

class GaitPart(nn.Module):
    def __init__(self, *args, **kargs):
        super(GaitPart, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """
        #手动构建网络
        '''
        self.BC1 = BasicConv2d(1,32,5, 1, 2)
        self.BC2 = BasicConv2d(1, 32, 5, 1, 2)
        self.M1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC1 = FocalConv2d(32,64,kernel_size=3, stride=1, padding=1, halving=2)
        self.FC2 = FocalConv2d(32, 64, kernel_size=3, stride=1, padding=1, halving=2)
        self.M2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC3 = FocalConv2d(64,128,kernel_size=3, stride=1, padding=1, halving=3)
        self.FC4 = FocalConv2d(64, 128, kernel_size=3, stride=1, padding=1, halving=3)
        '''

        self.Backbone = nn.Sequential(OrderedDict([
            ('BC1', BasicConv2d(1,32,5, 1, 2)),
            #('cb1', nn.BatchNorm2d(32)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('BC2', BasicConv2d(32, 32, 5, 1, 2)),
            #('cb2', nn.BatchNorm2d(32)),
            ('relu2', nn.LeakyReLU(inplace=True)),
            ('M1',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('FC1',FocalConv2d(32,64,kernel_size=3, stride=1, padding=1, halving=2)),
            #('cb3', nn.BatchNorm2d(64)),
            ('relu3', nn.LeakyReLU(inplace=True)),
            ('FC2', FocalConv2d(64, 64, kernel_size=3, stride=1, padding=1, halving=2)),
            #('cb4', nn.BatchNorm2d(64)),
            ('relu4', nn.LeakyReLU(inplace=True)),
            ('M2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('FC3', FocalConv2d(64, 128, kernel_size=3, stride=1, padding=1, halving=3)),
            #('cb5', nn.BatchNorm2d(128)),
            ('relu5', nn.LeakyReLU(inplace=True)),
            ('FC4', FocalConv2d(128, 128, kernel_size=3, stride=1, padding=1, halving=3)),
            #('cb6', nn.BatchNorm2d(128)),
            ('relu6', nn.LeakyReLU(inplace=True))
        ]))
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.Head = SeparateFCs(12,128,128)
        self.cb1 = nn.BatchNorm1d(12)
        self.HPP = SetBlockWrapper(
           HorizontalPoolingPyramid(bin_num=[12]))
        #self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=128, parts_num=12))
        self.cb2 = nn.BatchNorm1d(8)

#层定义
    '''
    def build_network(self, model_cfg):


    self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
    head_cfg = model_cfg['SeparateFCs']
    self.Head = SeparateFCs(**model_cfg['SeparateFCs'])
    self.Backbone = SetBlockWrapper(self.Backbone)
    self.HPP = SetBlockWrapper(
        HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
    self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
        in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))
    '''


    def forward(self, inputs, seqL):


        '''
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts
        out = self.Backbone(sils)  # [n, s, c, h, w]
        '''

        out = self.Backbone(inputs)  # [n, s, c, h, w]
        #print(out.shape)
        out = self.HPP(out)  # [n, s, c, p]
        #print(out.shape)
        #out = self.cb1(self.TFA(out, seqL) ) # [n, p, c]
        out = self.TFA(out, seqL)  # [n, p, c]
        #embs = self.cb2(self.Head(out.permute(1, 0, 2).contiguous()) ) # [p, n, c]
        embs = self.Head(out.permute(1, 0, 2).contiguous()) # [p, n, c]
        embs = embs.permute(1, 0, 2).contiguous()  # [n, p, c]


        '''
         n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval
        '''
        return embs


