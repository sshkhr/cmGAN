from __future__ import print_function, absolute_import
import os
import glob
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

from utils import mkdir_if_missing, write_json, read_json

"""RGB Image ReID"""

class Eval_Loader(object):

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

        '''
        self.dataset_dir = osp.join(dataset_dir, root)
        print(self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding-box-train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding-box-test')

        self._check_before_run()
        '''

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
