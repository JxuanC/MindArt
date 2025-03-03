import torch
import torch.nn as nn
from optimal_Transport import SinkhornDistance

def ntxent_loss(fmri_embedding, img_embedding):
    DEVICE = fmri_embedding.device
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    similarity_matrix = torch.matmul(fmri_embedding, img_embedding.float().T)
    labels = torch.eye(fmri_embedding.shape[0], dtype = torch.float).to(DEVICE)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)
    logits = torch.cat([positives, negatives], dim = 1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)
    return criterion(logits, labels), similarity_matrix#, logits, labels

def ntxent_loss_with_soft_labels(fmri_embedding, img_embedding, labels = None, factor = 10):
    DEVICE = fmri_embedding.device
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    similarity_matrix = torch.matmul(fmri_embedding, img_embedding.float().T)
    labels = (factor * similarity_matrix).softmax(-1).to(DEVICE) if labels is None else labels
    loss = criterion(similarity_matrix, labels)
    return loss, similarity_matrix

def matching_loss(fmri_embedding, img_embedding, labels):
    DEVICE = fmri_embedding.device
    similarity_matrix = torch.matmul(fmri_embedding, img_embedding.float().T)
    SinkhornDistance(0.01, 100)
