import sys
sys.path.append('.')
import clip
import faiss
import torch
import numpy as np
import datetime
import logging.config
import torch.nn as nn
import torch.nn.functional as F
from modules.GM import fMRIGM
from data.dataset_augment import get_clip_fmri_dataset
from utils import setup_logging_from_args
from optimal_Transport import log_optimal_transport
import data.configure as config

train_clip_features = np.load(config.CLIPTRFEATURE)
test_clip_features = np.load(config.CLIPTEFEATURE)

def cal_matching_matrix(fmri_categories, img_categories):
    img_categories = np.array(img_categories)
    fmri_categories = np.array(fmri_categories)
    matching_labels = np.array([fmri_categories[i] == img_categories for i in range(len(fmri_categories))])
    img_dustbin = ~np.any(matching_labels, 0)
    fmri_dustbin = np.hstack((~np.any(matching_labels, 1), False))[:,None]
    aug_matching_labels = np.vstack((matching_labels, img_dustbin))
    aug_matching_labels = np.hstack((aug_matching_labels, fmri_dustbin))
    return torch.tensor(matching_labels).to(DEVICE), torch.tensor(aug_matching_labels).to(DEVICE)

def epoch_train(fMRI_GM, optimizer, train_dl):
    running_loss = []
    for idx, (image_clips, fmris, img_categories, fmri_categories) in enumerate(train_dl):
        fMRI_GM.train()
        image_clips = image_clips.float().to(DEVICE)
        fmris = fmris.float().to(DEVICE)
    
        fmri_embedding, virtual_embedding = fMRI_GM(fmris, image_clips)
        fmri_embedding = torch.concat((fmri_embedding, virtual_embedding), 0)
        fmri_embedding = fmri_embedding / fmri_embedding.norm(dim=-1, keepdim=True)
        img_embedding = image_clips / image_clips.norm(dim=-1, keepdim=True)
        cross_similarity = torch.matmul(fmri_embedding, img_embedding.T)
        fmri_similarity = torch.matmul(fmri_embedding, fmri_embedding.T)
        img_similarity = torch.matmul(img_embedding, img_embedding.T)
        cross_similarity = cross_similarity# / 512 **.5
        fmri_categories = fmri_categories + img_categories
        across_labels, aug_across_labels = cal_matching_matrix(fmri_categories, img_categories)
        self_labels, aug_self_labels = cal_matching_matrix(fmri_categories, fmri_categories)
        scores = log_optimal_transport(cross_similarity.unsqueeze(0), 
                                       torch.nn.Parameter(torch.tensor(1.0)).to(DEVICE), iters = 100)
        #ntxent_loss, similarity = ntxent_loss(fmri_embedding, img_embedding)
        loss = torch.mean(-scores[aug_across_labels[None,:,:]])# + 0.1 * ntxent_loss
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        if idx % 10 == 0: # print every 10 mini-batches
            trained_num = idx * image_clips.shape[0]
            data_num = len(train_dl) * image_clips.shape[0]
            percent = int(100. * trained_num / data_num)
            logging.info(f"Epoch: {(epoch + 1):4d} Batch: {(idx + 1):4d} [{(trained_num):5d}/{(data_num):5d} ({(percent):2d}%)]" +
                  f"  Loss: {(np.mean(running_loss)):.4f}")
            running_loss = []
            

if __name__ == '__main__':
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected_rois = ['ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_V4', 'ROI_LOC', 'ROI_FFA', 'ROI_PPA']
    selected_sub = 'sub-3'
    candidate, random_matching = True, True
    train_clip_features = np.vstack((train_clip_features, test_clip_features)) if random_matching else train_clip_features
    log_path = setup_logging_from_args(f'./log/fMRIGM/DIR/{selected_sub}/', 'model')
    vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)#ViT-B/16,ViT-B/32,RN50,RN101
    train_dl, fmri_dim = get_clip_fmri_dataset(selected_sub, selected_rois, [train_clip_features, test_clip_features], 1024, candidate, random_matching)
    retrieval_topk = 0
    fmrigm = fMRIGM(fmri_dim, len(selected_rois), retrieval_topk, 512, 12, 8, None, None).to(DEVICE)
    optimizer = torch.optim.Adam(fmrigm.parameters(), lr = 2e-5)

    for epoch in range(30):
        epoch_train(fmrigm, optimizer, train_dl)