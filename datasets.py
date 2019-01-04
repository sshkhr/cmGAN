from __future__ import print_function, absolute_import
import glob
import random
import os
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import random
from time import time

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SYSU_triplet_dataset(Dataset):

    def __init__(self, data_folder = 'SYSU-MM01', transforms_list=None, mode='train', search_mode='all'):

        if mode == 'train':
            self.id_file = 'train_id.txt'
        elif mode == 'val':
            self.id_file = 'val_id.txt'
        else:
            self.id_file = 'test_id.txt'

        if search_mode == 'all':
            self.rgb_cameras = ['cam1','cam2','cam4','cam5']
            self.ir_cameras = ['cam3','cam6']
        elif search_mode == 'indoor':
            self.rgb_cameras = ['cam1','cam2']
            self.ir_cameras = ['cam3','cam6']

        file_path = os.path.join(data_folder,'exp',self.id_file)

        with open(file_path, 'r') as file:
            self.ids = file.read().splitlines()

        self.ids = [int(y) for y in self.ids[0].split(',')]
        self.ids.sort()

        self.id_dict = {}

        for index, id in enumerate(self.ids):
            #print(index,id)
            self.id_dict[id] = index

        self.ids = ["%04d" % x for x in self.ids]

        self.transform = transforms.Compose(transforms_list)
        
        self.files_rgb = {}
        self.files_ir = {}

        for id in sorted(self.ids):

            self.files_rgb[id] = [] 
            self.files_ir[id] = []
            
            for cam in self.rgb_cameras:
                img_dir = os.path.join(data_folder,cam,id)
                if os.path.isdir(img_dir):
                    self.files_rgb[id].extend(sorted([img_dir+'/'+i for i in os.listdir(img_dir)]))
            for cam in self.ir_cameras:
                img_dir = os.path.join(data_folder,cam,id)
                if os.path.isdir(img_dir):
                    self.files_ir[id].extend(sorted([img_dir+'/'+i for i in os.listdir(img_dir)]))  
        
        self.all_files = []
        #self.rgb_files = []
        #self.ir_files = []

        for id in sorted(self.ids):
            self.all_files.extend(self.files_rgb[id])
            #self.rgb_files.extend(self.files_rgb[id])
            self.all_files.extend(self.files_rgb[id])
            #self.ir_files.extend(self.files_ir[id]) 
        
    def __getitem__(self, index):

        anchor_file = self.all_files[index]
        '''
        anchor_cam = anchor_file.split('/')[1]

        if anchor_cam in self.ir_cameras:
            target_files = self.files_rgb
            modality = torch.tensor([1,0]).float()
        else:
            target_files = self.files_ir
            modality = torch.tensor([0,1]).float()
        '''

        anchor_id = anchor_file.split('/')[2]

        anchor_rgb = np.random.choice(self.files_rgb[anchor_id])
        positive_rgb =  np.random.choice([x for x in self.files_rgb[anchor_id] if x != anchor_rgb])
        negative_id = np.random.choice([id for id in self.ids if id != anchor_id])
        negative_rgb = np.random.choice(self.files_rgb[negative_id])      
        
        anchor_ir = np.random.choice(self.files_ir[anchor_id])
        positive_ir =  np.random.choice([x for x in self.files_ir[anchor_id] if x != anchor_ir])
        negative_id = np.random.choice([id for id in self.ids if id != anchor_id])
        negative_ir = np.random.choice(self.files_ir[negative_id])      
        
        anchor_label = np.array(self.id_dict[int(anchor_id)])

        #print(anchor_file, positive_file, negative_file, anchor_id)
            
        anchor_rgb = Image.open(anchor_rgb)
        positive_rgb = Image.open(positive_rgb)
        negative_rgb = Image.open(negative_rgb)

        anchor_ir = Image.open(anchor_ir)
        positive_ir = Image.open(positive_ir)
        negative_ir = Image.open(negative_ir)
        
        if self.transform is not None:
            anchor_rgb = self.transform(anchor_rgb)
            positive_rgb = self.transform(positive_rgb)
            negative_rgb = self.transform(negative_rgb)
            
            anchor_ir = self.transform(anchor_ir)
            positive_ir = self.transform(positive_ir)
            negative_ir = self.transform(negative_ir)

        modality_rgb = torch.tensor([1,0]).float()
        modality_ir = torch.tensor([0,1]).float()

        return anchor_rgb, positive_rgb, negative_rgb, anchor_ir, positive_ir, negative_ir, anchor_label, modality_rgb, modality_ir

    def __len__(self):
        return len(self.all_files)



class SYSU_eval_datasets(object):

    def __init__(self, dataset_dir = 'SYSU-MM01', search_mode='all' , data_split='val', **kwargs):
        
        self.data_folder = dataset_dir
        self.train_id_file = 'train_id.txt'
        self.val_id_file = 'val_id.txt'
        self.test_id_file = 'test_id.txt'

        if search_mode == 'all':
            self.rgb_cameras = ['cam1','cam2','cam4','cam5']
            self.ir_cameras = ['cam3','cam6']
        elif search_mode == 'indoor':
            self.rgb_cameras = ['cam1','cam2']
            self.ir_cameras = ['cam3','cam6']

        if data_split == 'train':
            self.id_file = self.train_id_file
        elif data_split == 'val':
            self.id_file = self.val_id_file
        elif data_split == 'test':
            self.id_file = self.test_id_file
            

        random.seed(time)

        query, num_query_pids, num_query_imgs = self._process_query_images(id_file = self.id_file, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_gallery_images(id_file = self.id_file, relabel=False)
        
        num_total_pids = num_query_pids
        num_total_imgs = num_query_imgs + num_gallery_imgs

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.query = query
        self.gallery = gallery

        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
    
    def _process_query_images(self, id_file, relabel=False):
        
        file_path = os.path.join(self.data_folder,'exp',id_file)

        files_rgb = []
        files_ir = []

        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        for id in sorted(ids):
            for cam in self.rgb_cameras:
                img_dir = os.path.join(self.data_folder,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    files_rgb.append(random.choice(new_files))#files_rgb.extend(new_files)
            for cam in self.ir_cameras:
                img_dir = os.path.join(self.data_folder,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    files_ir.append(random.choice(new_files))#files_ir.extend(new_files)

        pid_container = set()
        for img_path in files_ir:
            camid, pid = int(img_path.split('/')[1].split('cam')[1]), int(img_path.split('/')[2])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in files_ir:
            #print(img_path)
            camid, pid = int(img_path.split('/')[1].split('cam')[1]), int(img_path.split('/')[2])
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        #print("query done")
        #input()

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    def _process_gallery_images(self, id_file, relabel=False):
        
        file_path = os.path.join(self.data_folder,'exp',id_file)

        files_rgb = []
        files_ir = []

        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        for id in sorted(ids):
            for cam in self.rgb_cameras:
                img_dir = os.path.join(self.data_folder,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    files_rgb.extend(new_files)
            for cam in self.ir_cameras:
                img_dir = os.path.join(self.data_folder,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)

        pid_container = set()
        for img_path in files_rgb:
            camid, pid = int(img_path.split('/')[1].split('cam')[1]), int(img_path.split('/')[2])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in files_rgb:
            #print(img_path)
            camid, pid = int(img_path.split('/')[1].split('cam')[1]), int(img_path.split('/')[2])
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

class Image_dataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid
