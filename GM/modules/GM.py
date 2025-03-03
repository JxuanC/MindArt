import torch
import numpy as np
from math import log
from torch import nn
import torch.utils.data
from einops import rearrange
from modules.vit import fMRI_ViT_Encoder
from modules.gnn import AttentionalGNN
import random

class simpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(simpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class fMRIGM(nn.Module):
    def __init__(self, fmri_dim, rois_len, topk, embed_dim, depth, num_heads, clip_features, retrieval_index):
        super(fMRIGM, self).__init__()
        self.clip_features = clip_features
        self.retrieval_index = retrieval_index
        self.topk = topk
        self.fmri_encoder = fMRI_ViT_Encoder(512, rois_len, embed_dim, depth, num_heads)
        self.img_encoder = simpleMLP(512, 256, 512)
        self.retrieval_encoder = fMRI_ViT_Encoder(embed_dim, topk + 1, embed_dim, depth, num_heads)
        self.gnn = AttentionalGNN(512, ['self', 'cross', 'self', 'cross', 'self', 'cross'])
        self.final_proj = nn.Conv1d(512, 512, kernel_size = 1, bias = True)
        self.start_proj = nn.Linear(fmri_dim, 512)
    
    def retrieval(self, fmri):
        D, I = self.retrieval_index.search(fmri.detach().cpu().numpy(), self.topk)
        return self.clip_features[I]
    
    def interpolation(self, x, noise = False):
        fmri_num = x.shape[1]
        selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
        if(selected_no.shape[0] != 1):
            coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0]), dtype = torch.float32).softmax(0)
            mixup_x = torch.sum(x[:, selected_no] * coefficient[None, :, None, None].to(x.device), 1)
            return mixup_x
        return x[:, selected_no, :, :].squeeze()

    def forward(self, fmri, img):
        # x (batch, roi_num, roi_dim)
        x = self.start_proj(fmri)
        if(len(x.shape) == 4):
            x = self.interpolation(x)
        y = self.img_encoder(img)
        #y = self.fmri_encoder(y.view_as(x))
        x = self.fmri_encoder(x)

        x, y = self.gnn(rearrange(x, '(b n) d -> b d n', b = 1), rearrange(y, '(b n) d -> b d n', b = 1))
        x, y = rearrange(x, 'b d n -> (b n) d', b = 1), rearrange(y, 'b d n -> (b n) d', b = 1)
        if(self.topk > 0):
            y = self.retrieval(x)
            y = torch.tensor(y, dtype = torch.float32).to(x.device)
            x, y = self.gnn(rearrange(x, '(b n) d -> b d n', b = 1), rearrange(y, 'n k d -> k d n'))
            return rearrange(self.final_proj(x), 'b d n -> (b n) d')
        else:
            return x.squeeze(), y.squeeze()
        
    def encode_fmri(self, fmri, return_class_embedding = True):
        x = self.start_proj(fmri)
        x = self.fmri_encoder(x, return_class_embedding)
        return x