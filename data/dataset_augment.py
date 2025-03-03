import os
import faiss
import bdpy
import json
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data
import data.configure as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio
from GM.modules.GM import fMRIGM

def DIR_sub_without_images(sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois = [], []
    DIR_sub_train = bdpy.BData(os.path.join(config.DIR_dataset_dir, config.DIR_train_subs[sub]))
    DIR_sub_test = bdpy.BData(os.path.join(config.DIR_dataset_dir, config.DIR_test_subs[sub]))

    train_image_index = DIR_sub_train.select('image_index').squeeze().astype(int) - 1
    test_image_index = DIR_sub_test.select('image_index').squeeze().astype(int) - 1

    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    
    MAX_DIM = 0
    for roi in rois:
        train_roi_fMRI = DIR_sub_train.select(roi)
        test_roi_fMRI = DIR_sub_test.select(roi)

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()

    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    for cat in testCatIDs:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs

class CLIP_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, CLIP_features, fMRI, train = True, random_matching = False):
         self.imageIDs = imageIDs
         self.CLIP_features = CLIP_features
         self.fMRI = fMRI
         self.train = train
         self.random_matching = random_matching
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        image_category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image_clip = self.CLIP_features[idx]
        #image = Image.open(self.imageIDs[idx])
        #vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
        fmri_category_id = np.random.choice(list(self.fMRI.keys())) if self.random_matching else image_category_id
        fMRI = self.fMRI[fmri_category_id]
        return image_clip, fMRI, image_category_id, fmri_category_id

class Visual_Text_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, caps, categories, fMRI, tokenizer, mixup = False, train = True, transform = None, 
                  k = 0, template_path = './data/template_retrieval.txt', max_caption_length = 25, clip_features = None, retrieval_caps = None,
                  retrieval_model = None, retrieval_index = None):
        self.tokenizer = tokenizer
        self.imageIDs = imageIDs
        self.sortedimageIDs = np.sort(imageIDs)
        self.fMRI = fMRI
        self.mixup = mixup
        self.train = train
        self.transform = transform
        self.caps = caps
        self.categories = categories
        self.clip_features = clip_features
        self.retrieval_caps = retrieval_caps
        self.k = k
        self.retrieval_model = retrieval_model
        self.retrieval_index = retrieval_index
        self.SIMPLE_PREFIX = "This image shows "
        self.retrieved_caps = None
        self.CAPTION_LENGTH = max_caption_length

        if(self.k > 0):
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                         + max_caption_length * k # retrieved captions
                                         + len(tokenizer.encode(self.template)) # template
                                         + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                         )
        else:
            self.template = self.SIMPLE_PREFIX
            self.max_target_length = (max_caption_length
                                    + len(tokenizer.encode(self.template)))

     def __len__(self):
        return len(self.imageIDs)

     def prep_strings(self, text, tokenizer, retrieved_caps = None): 
        if not self.train:
            padding = False
            truncation = False
        else:
            padding = True 
            truncation = True
        
        if retrieved_caps is not None:
            infix = '\n\n'.join(retrieved_caps) + '.'
            prefix = self.template.replace('||', infix)
        else:
            prefix = self.SIMPLE_PREFIX

        prefix_ids = tokenizer.encode(prefix)
        len_prefix = len(prefix_ids)

        text_ids = tokenizer.encode(text, add_special_tokens = False)
        if truncation:
            text_ids = text_ids[:self.CAPTION_LENGTH]
        input_ids = prefix_ids + text_ids if self.train else prefix_ids

        # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
        label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
        if padding:
            input_ids += [tokenizer.pad_token_id] * (self.max_target_length - len(input_ids))
            label_ids += [-100] * (self.max_target_length - len(label_ids))
        
        if not self.train:
            return input_ids
        else:  
            return input_ids, label_ids
     
     @torch.no_grad()
     def retrieval_caption(self, fMRI):
        fMRI = torch.tensor(fMRI[None,:,:], dtype = torch.float32).cuda()
        fmri_embedding = self.retrieval_model.encode_fmri(fMRI)[None,:]
        fmri_embedding = fmri_embedding / fmri_embedding.norm(dim=-1, keepdim=True)
        dis, nns = self.retrieve_imgs(fmri_embedding.cpu().numpy())
        return [self.caps[str(self.sortedimageIDs[nns[0][k]]).split('/')[-1]] for k in range(self.k)]
        
     def retrieve_imgs(self, image_embedding):
        xq = image_embedding.astype(np.float32)
        D, I = self.retrieval_index.search(xq, self.k) 
        return D, I
     
     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image = Image.open(self.imageIDs[idx])
        cap = self.caps[file_name]
        category = self.categories[category_id]
        #visual_features = self.clip_features[idx]
        visual_features = self.clip_features[str(self.imageIDs[idx])][()]
        if(self.train):
            fmri_num = self.fMRI[category_id].shape[0]
            selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
            if(self.mixup and selected_no.shape[0] != 1):
                coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0])).softmax(0)
                selected_fMRI = torch.tensor(self.fMRI[category_id][selected_no])
                coefficient = coefficient[:, None, None] if len(selected_fMRI.shape) == 3 else coefficient[:, None]
                mixup_fMRI = torch.sum(selected_fMRI * coefficient, 0)
                fMRI = mixup_fMRI.numpy()
            else:
                fMRI = self.fMRI[category_id][selected_no[0]].squeeze()
        else:
            selected = random.randint(0, self.fMRI[category_id].shape[0] - 1)
            fMRI = self.fMRI[category_id][selected]

        image = self.transform(image) if self.transform else image
        k_caption = None
        if(self.k > 0 and self.retrieval_caps is None):
            k_caption = self.retrieval_caption(fMRI)
        decoder_input_ids, labels = self.prep_strings(cap, self.tokenizer, k_caption)
        data = {'encoder_inputs': fMRI.astype(np.float32), 'encoder_labels': visual_features, 
                'decoder_input_ids': np.array(decoder_input_ids), 'decoder_labels': np.array(labels)}
        return data

def get_clip_fmri_dataset(sub, rois, clip_dataset, batch_size, candidate = True, random_matching = True):

    train_cat_rois, _, trainStiIDs, test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    #Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Tr_Aug}/{category}/*.JPEG")) for category in Train_category])
        #test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Te_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Tr_Aug}/*/{image}")) for image in trainStiIDs])
        #test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Te_Aug}/*/{image}")) for image in testStiIDs])
        
    train_images = np.hstack((np.sort(train_images))) if random_matching else np.sort(train_images)
    #test_images = np.sort(test_images)

    train_dataset = CLIP_fMRI_Dataset(train_images, clip_dataset[0], train_cat_rois, True, random_matching)
    #test_dataset = CLIP_fMRI_Dataset(test_images, clip_dataset[1], test_cat_rois, False, random_matching)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    #test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader, fmri_dim

def get_visual_text_fmri_dataset(sub, rois, batch_size, mixup = True, candidate = True, 
                            tokenizer = None, clip_features = None, retrieval_k = 0, retrieval_model_path = None, 
                            retrieval_index_path = None, template_path = './data/template_retrieval.txt'):

    train_cat_rois, _, trainStiIDs,\
    test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)

    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    #Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Train_category])
        #test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in trainStiIDs])
        #test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in testStiIDs])
        
    train_caps = json.load(open(config.smallCap_Kamitani_train))
    #test_caps = json.load(open('database/captioning/smallCap_Kamitani_test.json'))
    classIDs = np.array(pd.read_csv(config.kamitani_sti_text, header = None)[0])
    classTexts = np.array(pd.read_csv(config.kamitani_sti_text, header = None)[1])
    id_text = dict(zip(classIDs, classTexts))

    if(retrieval_model_path is not None and retrieval_index_path is not None):
        fmrigm = fMRIGM(fmri_dim, len(rois), 0, 512, 12, 8, None, None).cuda()
        fmrigm.load_state_dict(torch.load(retrieval_model_path, map_location = 'cpu'))
        retrieval_model = fmrigm.eval()
        retrieval_index = faiss.read_index(retrieval_index_path)
        res = faiss.StandardGpuResources()  
        retrieval_index = faiss.index_cpu_to_gpu(res, 0, retrieval_index)
    else:
        retrieval_model, retrieval_index = None, None

    train_dataset = Visual_Text_fMRI_Dataset(np.sort(train_images), train_caps, id_text, train_cat_rois, tokenizer, mixup, 
                                                True, clip_features = clip_features, k = retrieval_k, retrieval_model = retrieval_model, 
                                                retrieval_index = retrieval_index, template_path = template_path)
    #test_dataset = Visual_Text_fMRI_Dataset(np.sort(test_images), test_caps, id_text, test_cat_rois, tokenizer, mixup, False)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    #test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataset, train_dataloader, fmri_dim#, #test_dataloader, fmri_dim