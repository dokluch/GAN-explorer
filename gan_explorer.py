from __future__ import print_function

import os, math, ipyplot
import numpy as np
import torch, torchvision, pickle
import PIL
from PIL import Image
from matplotlib.pyplot import imshow
import IPython.display
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from tqdm.notebook import tqdm

from timeline_controls, render_controls, model_loader, stylegan2_ada_tools import *

class stylegan2_ada_model:
    def __init__(self):
        self.name = ""
        self.path = ""
        self.prefix = ""
        self.model = None

    def update_name_path(self, name, path):
        self.name = name
        self.path = path
        self.model = self.load_model()

    def update_prefix(self, prefix):
        self.prefix = prefix

    def load_model(self):
        with open(self.path, 'rb') as f:
            G = pickle.load(f)['G_ema'].cuda() 
        return(G)  

class seeds_updater:
    def __init__(self):
        self.seed_list = []
        self.imgs_list = []
    def add_seed_img(self, seed, img):
        self.seed_list.append(seed)
        self.imgs_list.append(img)
    def remove_last(self):
        self.seed_list = self.seed_list[:-1]
        self.imgs_list = self.imgs_list[:-1]
    def reset(self):
        self.seed_list = []
        self.imgs_list = []