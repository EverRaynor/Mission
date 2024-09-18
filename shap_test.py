import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import matplotlib.pyplot as plt
from network_tcmr import mmWaveModel_tcmr_lime
from network_tcmr import mmWaveModel_tcmr_lime2
from network_tcmr import mmWaveModel_tcmr_lime_singleframe
from network_tcmr import mmWaveModel_ti_Anchor_nosmpl_bidirectional
from network_tcmr import TCMR
from collections import Counter
import torch
from lib.models.spin import hmr
from shapash import SmartExplainer
import os.path as osp
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

import shap
from sklearn.datasets import load_boston
import sklearn.ensemble
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:%d' % (0)
else:
    device = 'cpu'

model=mmWaveModel_tcmr_lime().to(device)
model2=mmWaveModel_tcmr_lime2().to(device)
model_singleframe=mmWaveModel_tcmr_lime_singleframe().to(device)
model.load('F:\SouthEast\Reid\\rgb_ti_smpl\log\Backbone\Anchor_id20_len10_key19_smpl_tcmr_notrain_gt2_full/model_{}.pth'.format(1099))
model2.load('F:\SouthEast\Reid\\rgb_ti_smpl\log\Backbone\Anchor_id20_len10_key19_smpl_tcmr_notrain_gt2_full/model_{}.pth'.format(1099))

model_tcmr = TCMR(
            seqlen=10,
            n_layers=2,
            hidden_size=1024).to(device)
# print(model)
pretrained_file = './lib/models/pretrained/base_data/tcmr_demo_model.pth.tar'
ckpt = torch.load(pretrained_file)
print(f"Load pretrained weights from \'{pretrained_file}\'")
ckpt = ckpt['gen_state_dict']
model_tcmr.load_state_dict(ckpt, strict=False)
# mmMeshBackbone
model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional().to(device)
model_ti.load('./log/Backbone/Anchor_id20_nonormalization_len10_key19_nosmpl_bidirectional/model_{}.pth'.format(2999))
model_ti.eval()
BASE_DATA_DIR = './lib/models/pretrained/base_data'
# hmrBackbone
model_hmr = hmr().to(device)
checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
model_hmr.load_state_dict(checkpoint['model'], strict=False)
model_hmr.eval()

def pre_fn(data):
    seq_len, size = data.shape
    data_tensor = torch.tensor(data,dtype=torch.float32, device=device)
    print("data_tensor:",data_tensor.shape)
    data_ti = data_tensor[:,:64*3].view(seq_len,64,3)
    data_rgb = data_tensor[:, 64 * 3:].view(seq_len, 3,224,224)
    print("data_rgb:", data_rgb.shape)

    with torch.no_grad():
        feature_hmr = model_hmr.feature_extractor(data_rgb)
        feature_hmr = feature_hmr.view(batch_size, seq_len, 2048)
        feature_tcmr, _ = model_tcmr(feature_hmr)
        print("feature_tcmr", feature_tcmr.shape)
        g_vec, a_vec, _ = model_ti(data_ti, h0, c0, batch_size, seq_len)
        ti_vec = torch.cat((a_vec, g_vec), dim=2)
        s, rgb_vec = model(g_vec, a_vec, feature_tcmr)
        x = torch.cat((ti_vec, rgb_vec), dim=2)
        x = x.squeeze()

        key_pred = model2(x)
        key_pred = np.asarray(key_pred.cpu().detach())
    return key_pred

batch_size=1
seq_len=10


list_all_rgb_full = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_rgb_full.npy")

list_all_ti = np.load(
                "F:\SouthEast\Reid\Reid_data\\npydata\SiamesePC_20_nonormalization_rgb\list_all_ti_calibration.npy")


for epoch in range(1):
    print("epoch:",epoch)
    data_ti = list_all_ti[epoch, 25 - seq_len:, :]
    data_rgb = list_all_rgb_full[epoch, 25 - seq_len:, :, :, :]
    data_rgb = np.reshape(data_rgb, (batch_size * seq_len, 224*224*3))
    data_ti=np.reshape(data_ti, (batch_size * seq_len, 64*3))
    print("data_rgb",data_rgb.shape)
    print("data_ti", data_ti.shape)
    data_ti=torch.tensor(data_ti, dtype=torch.float32, device=device).squeeze()
    data_rgb=torch.tensor(data_rgb, dtype=torch.float32, device=device).squeeze()
    data_in = torch.cat((data_ti,data_rgb),dim=1)
    h0 = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
    c0 = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
    '''
    with torch.no_grad():
        feature_hmr = model_hmr.feature_extractor(data_rgb)
        feature_hmr = feature_hmr.view(batch_size, seq_len, 2048)
        feature_tcmr, _ = model_tcmr(feature_hmr)
        print("feature_tcmr",feature_tcmr.shape)
        g_vec, a_vec, _ = model_ti(data_ti, h0, c0,  batch_size, seq_len)
        ti_vec = torch.cat((a_vec, g_vec), dim=2)
        s,rgb_vec = model(g_vec, a_vec, feature_tcmr)
        x = torch.cat((ti_vec,rgb_vec),dim=2)
        x = x.squeeze()
        print("x", x.shape)
        #s = model2(x)
        #s = s.squeeze()
        ti_vec = ti_vec.squeeze()
        rgb_vec = rgb_vec.squeeze()
        print("ti_vec", ti_vec.shape)
        #print("rgb_vec", rgb_vec.shape)
        #print("s", s)
'''


    #explainer = shap.KernelExplainer(pre_fn,np.asarray(data_in.cpu()))
    explainer = shap.GradientExplainer(pre_fn)
    i = 0
    #shap_values = explainer.shap_values(np.asarray(data_in.cpu()))
    shap_values = explainer(np.asarray(data_in[5].cpu()))

    ti_count = Counter(shap_values[:256])
    rgb_count = Counter(shap_values[256:])
    top_three_ti = ti_count.most_common(3)
    top_three_rgb = rgb_count.most_common(3)
    print("ti:",top_three_ti)
    print("rgb:",top_three_rgb)
    print("shapvalue_ti:",np.mean(shap_values[:256]))
    print("shapvalue_rgb:", np.mean(shap_values[256:]))
    #np.save("F:\SouthEast\Reid\Reid_data\\npydata\LIME/result_{}.npy".format(epoch),np.asarray(exp.as_list()))

'''
boston = load_boston()
categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()
print("categorical_features:",categorical_features)
print("feature_names:",boston.feature_names)
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target, train_size=0.80)
print("train:",train.shape)
rf.fit(train, labels_train)
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, verbose=True, mode='regression')
i = 25
exp = explainer.explain_instance(test[i], rf.predict, num_features=1500)
print(test[i])
#exp.show_in_notebook(show_table=True)
print(exp.as_list())

'''
