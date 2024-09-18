import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        # distances = (output2 - output1).pow(2).sum(1) #原来的代码
        distances = (output2 - output1).pow(2).sum(1).mean()  # squared distances
        # losses = 0.5 * (target.float() * distances +
        #                 (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        target=target.cuda()
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


if torch.cuda.is_available():
    device = 'cuda:%d' % (1)
else:
    device = 'cpu'


'''batch hard'''
def euclidean_dist(x,y):
    m,n = x.size(0),y.size(0)
    xx = torch.pow(x,2).sum(1,keepdim=True).expand(m,n)
    yy = torch.pow(y,2).sum(dim=1,keepdim=True).expand(n,m).t()
    xx.to(device)
    yy.to(device)
    dist = xx + yy
    dist.addmm_(1,-2,x,y.t())           #warning修改
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def _batch_hard(mat_distance,mat_similarity,indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-100000.0)*(1 - mat_similarity),dim=1, descending=True)
    hard_p = sorted_mat_distance[:,0]
    #print(hard_p[[0]])
    hard_p_indice = positive_indices[:,0]
    sorted_mat_distance, negative_indices = torch.sort( mat_distance + 100000.0 * mat_similarity,dim = 1,descending=False )
    hard_n = sorted_mat_distance[:,0]
    hard_n_indice = negative_indices[:,0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

def _batch_all(mat_distance,mat_similarity,indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance *mat_similarity,dim=1, descending=True)
    hard_p = torch.sum(sorted_mat_distance,1)
    #hard_p = (sorted_mat_distance[:, 0]+sorted_mat_distance[:, 1])/2
    #print(sorted_mat_distance)
    #print(hard_p[[0]])
    hard_p_indice = positive_indices[:,0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance * (mat_similarity-1),dim = 1,descending=False   )
    hard_n = torch.sum(sorted_mat_distance,1)
    #hard_n = (sorted_mat_distance[:, 0]+sorted_mat_distance[:, 1])/2
    hard_n_indice = negative_indices[:,0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=0.5, normalize_feature = True):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin = margin)
        #self.triplet_selector = triplet_selector

    def forward(self, embeddings_kinect, embeddings_ti,label_kinect,label_ti):

        if self.normalize_feature:
            emb_ti = F.normalize(embeddings_ti)
            emb_kinect = F.normalize(embeddings_kinect)
            #print('emb')
            #print(emb_ti.shape)
            #print(emb_kinect.shape)
            mat_dist = euclidean_dist(emb_ti, emb_kinect)
            mat_dist2 = euclidean_dist(emb_ti, emb_ti)
            mat_dist3 = euclidean_dist(emb_kinect, emb_kinect)
            #print('mat_dist')
            assert mat_dist.size(0) == mat_dist.size(1)
            N = mat_dist.size(0)
            mat_sim = label_ti.expand(N,N).eq(label_kinect.expand(N,N).t()).float()
            #print(mat_sim-1)
            mat_sim2 = label_ti.expand(N, N).eq(label_ti.expand(N, N).t()).float()
            mat_sim3 = label_kinect.expand(N, N).eq(label_kinect.expand(N, N).t()).float()
            #print(mat_dist)
            # #print(mat_sim)
            #batch hard

            dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            dist_ap2, dist_an2 = _batch_hard(mat_dist2, mat_sim2)
            dist_ap3, dist_an3 = _batch_hard(mat_dist3, mat_sim3)
            assert dist_an.size(0) == dist_ap.size(0)
            y = torch.ones_like(dist_ap)
            loss1 = self.margin_loss(dist_an, dist_ap, y)
            assert dist_an2.size(0) == dist_ap2.size(0)
            y2 = torch.ones_like(dist_ap2)
            loss2 = self.margin_loss(dist_an2, dist_ap2, y2)
            assert dist_an3.size(0) == dist_ap3.size(0)
            y3 = torch.ones_like(dist_ap3)
            loss3 = self.margin_loss(dist_an3, dist_ap3, y3)
            # loss=(loss1+loss2+loss3)/3
            loss = self.margin_loss(dist_an, dist_ap, y)
            '''
            #batch all
            dist_ap=mat_dist*mat_sim
            #print((dist_ap != 0.).sum(dim=0))
            dist_ap_mean=dist_ap.sum(dim=0)/((dist_ap != 0.).sum(dim=0))
            #print(dist_ap_mean)
            #dist_ap =torch.mean(dist_ap,1)
            dist_an=mat_dist*(1-mat_sim)
            dist_an_mean = dist_an.sum(dim=0) / ((dist_an != 0.).sum(dim=0))
            #print(dist_an_mean)
            #dist_an = torch.mean(dist_an, 1)
            y = torch.ones_like(dist_ap)
            loss=dist_ap_mean-dist_an_mean+self.margin
            '''
            #valid_triplets = loss[loss > 1e-16]
            #print(valid_triplets.size(0))
            #num_positive_triplets = valid_triplets.size(0)
            #num_valid_triplets = mat_sim.sum()
            #loss=torch.mean(loss)
            #loss = loss.sum() / (num_positive_triplets + 1e-16)
            #loss = torch.mean(loss)
            #print(loss)
            if loss<0:
                loss=torch.tensor(0.0,requires_grad=True)
            #print(dist_ap)

            prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
            return loss, prec


class OnlineTripletLoss_single(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=0.5, normalize_feature = True):
        super(OnlineTripletLoss_single, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin = margin)
        #self.triplet_selector = triplet_selector

    def forward(self,  embeddings_ti,label_ti,batchsize):

        if self.normalize_feature:
            emb_ti = F.normalize(embeddings_ti)
            '''
            emb_ti = emb_ti.cpu().detach().numpy()
            emb_ti2 = emb_ti
            label_ti2 = label_ti
            label_ti = label_ti.cpu().detach().numpy()
            random_state = np.random.RandomState(1) #打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(emb_ti)
            random_state = np.random.RandomState(1) #打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(label_ti)
            emb_ti = torch.from_numpy(emb_ti).cuda()
            label_ti = torch.from_numpy(label_ti).cuda()
            '''
            #print(label_ti)
            label_ti2 = label_ti[0:batchsize//2]
            label_ti = label_ti[batchsize//2:batchsize]
            emb_ti2 = emb_ti[0:batchsize//2]
            emb_ti = emb_ti[batchsize//2:batchsize]
            #print(label_ti)
            #print(label_ti)
            #print(label_ti2)
            #print('emb')
            #print(emb_ti.shape)
            #print(emb_kinect.shape)
            mat_dist = euclidean_dist(emb_ti, emb_ti2)
            mat_dist2 = euclidean_dist(emb_ti, emb_ti)
            mat_dist3 = euclidean_dist(emb_ti2, emb_ti2)
            #print('mat_dist')
            assert mat_dist.size(0) == mat_dist.size(1)
            N = mat_dist.size(0)
            mat_sim = label_ti.expand(N,N).eq(label_ti2.expand(N,N).t()).float()
            #print(mat_sim-1)
            mat_sim2 = label_ti.expand(N, N).eq(label_ti.expand(N, N).t()).float()
            mat_sim3 = label_ti2.expand(N, N).eq(label_ti2.expand(N, N).t()).float()
            #print(mat_dist)
            # #print(mat_sim)
            #batch hard

            dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            dist_ap2, dist_an2 = _batch_hard(mat_dist2, mat_sim2)
            dist_ap3, dist_an3 = _batch_hard(mat_dist3, mat_sim3)
            assert dist_an.size(0) == dist_ap.size(0)
            y = torch.ones_like(dist_ap)
            loss1 = self.margin_loss(dist_an, dist_ap, y)
            assert dist_an2.size(0) == dist_ap2.size(0)
            y2 = torch.ones_like(dist_ap2)
            loss2 = self.margin_loss(dist_an2, dist_ap2, y2)
            assert dist_an3.size(0) == dist_ap3.size(0)
            y3 = torch.ones_like(dist_ap3)
            loss3 = self.margin_loss(dist_an3, dist_ap3, y3)
            # loss=(loss1+loss2+loss3)/3
            loss = self.margin_loss(dist_an, dist_ap, y)
            '''
            #batch all
            dist_ap=mat_dist*mat_sim
            #print((dist_ap != 0.).sum(dim=0))
            dist_ap_mean=dist_ap.sum(dim=0)/((dist_ap != 0.).sum(dim=0))
            #print(dist_ap_mean)
            #dist_ap =torch.mean(dist_ap,1)
            dist_an=mat_dist*(1-mat_sim)
            dist_an_mean = dist_an.sum(dim=0) / ((dist_an != 0.).sum(dim=0))
            #print(dist_an_mean)
            #dist_an = torch.mean(dist_an, 1)
            y = torch.ones_like(dist_ap)
            loss=dist_ap_mean-dist_an_mean+self.margin
            '''
            #valid_triplets = loss[loss > 1e-16]
            #print(valid_triplets.size(0))
            #num_positive_triplets = valid_triplets.size(0)
            #num_valid_triplets = mat_sim.sum()
            #loss=torch.mean(loss)
            #loss = loss.sum() / (num_positive_triplets + 1e-16)
            #loss = torch.mean(loss)
            #print(loss)
            if loss<0:
                loss=torch.tensor(0.0,requires_grad=True)
            #print(dist_ap)

            prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
            return loss, prec