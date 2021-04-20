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


def get_timeline_controls(model, seeds_updater):
    button_get_random = widgets.Button(description="Get random seed")
    button_prev = widgets.Button(description="<<<")
    button_next = widgets.Button(description=">>>")
    buttons_line_1 = widgets.HBox([button_prev, button_get_random, button_next])

    button_add_seed = widgets.Button(description="Add seed")
    button_remove_last = widgets.Button(description="Remove last seed")
    button_reset = widgets.Button(description="Reset timeline")
    buttons_line_2 = widgets.HBox([button_add_seed, button_remove_last, button_reset])

    output = widgets.Output()
    output2 = widgets.Output()

    def on_save_clicked(b):
        with output2:
            clear_output()
            if(seeds_updater.seed_list):
                if seeds_updater.seed_list[-1] != button_get_random.seed:
                    seeds_updater.add_seed_img(button_get_random.seed, button_get_random.img)
            else:
                seeds_updater.add_seed_img(button_get_random.seed, button_get_random.img)

            print(seeds_updater.seed_list)
            display_seeds_as_imgs()

    def on_remove_last(b):
        with output2:
            clear_output()
            if(seeds_updater.seed_list):
                seeds_updater.remove_last()
                print(seeds_updater.seed_list)
                display_seeds_as_imgs()

    def on_reset(b):
        with output2:
            clear_output()
            seeds_updater.reset()
            display_seeds_as_imgs()

    def display_seeds_as_imgs():
        if seeds_updater.imgs_list:
            ipyplot.plot_images(seeds_updater.imgs_list, labels = seeds_updater.seed_list, img_width=200)

    def on_random_clicked(b):
        with output:
            clear_output()
            seed_gen = np.random.randint(0, 400000)
            print(seed_gen)
            b.img = make_img_from_seed(model.model, seed_gen).resize((256,256))
            display(b.img)
            b.seed = seed_gen
            b.prev_seeds.append(b.seed)
            b.pos = len(b.prev_seeds)
            print(b.prev_seeds)

    def on_prev(b):
        with output:
            if len(button_get_random.prev_seeds) > 1 and button_get_random.pos >= 1:
                button_get_random.pos -= 1
                button_get_random.seed = button_get_random.prev_seeds[button_get_random.pos]
                button_get_random.img = make_img_from_seed(model.model, button_get_random.seed).resize((256,256))
                clear_output()
                print(button_get_random.seed)
                display(button_get_random.img)
                print(button_get_random.prev_seeds)

    def on_next(b):
        with output:
            if len(button_get_random.prev_seeds) > 1 and button_get_random.pos < len(button_get_random.prev_seeds) - 1:
                button_get_random.pos += 1
                button_get_random.seed = button_get_random.prev_seeds[button_get_random.pos]
                button_get_random.img = make_img_from_seed(model.model, button_get_random.seed).resize((256,256))
                clear_output()
                print(button_get_random.seed)
                display(button_get_random.img)
                print(button_get_random.prev_seeds)


    button_add_seed.seeds = []
    button_add_seed.imgs = []
    button_add_seed.on_click(on_save_clicked)
    button_remove_last.on_click(on_remove_last)
    button_reset.on_click(on_reset)

    button_get_random.prev_seeds = []
    button_get_random.on_click(on_random_clicked)
    button_prev.on_click(on_prev)
    button_next.on_click(on_next)
    on_random_clicked(button_get_random)

    return(output, buttons_line_1, buttons_line_2, output2)

def make_img_from_seed(Gs, seed_in = 0):
    torch.manual_seed(seed_in)
    z1 = torch.randn([1, Gs.z_dim]).cuda() 
    c = None #class
    w = Gs.mapping(z1, c, truncation_psi=0.7, truncation_cutoff=8)
    img = Gs.synthesis(w, noise_mode='const', force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

    return(img)

def make_img_from_vec(Gs, w_in):
    c = None
    img = Gs.synthesis(w_in, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

    return(img)
    
def seed2vec(Gs, seed_in = 0):
    torch.manual_seed(seed_in)
    z = torch.randn([1, Gs.z_dim]).cuda()
    w = Gs.mapping(vec_in, c, truncation_psi=0.7, truncation_cutoff=8)
    return w

def generate_image(Gs, w):
    c = None
    w = Gs.mapping(z, c, truncation_psi=0.7, truncation_cutoff=8)
    img = Gs.synthesis(w, noise_mode='const', force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img


def get_render_controls(model, seeds_updater, sequence_folder = "/content/sequence", video_folder = "/content/renders"):
    STEPS = 100
    easy_ease = 1
    loop = True
    SEEDS = [39644, 35189, 4531, 11258, 7987] #MANUANAL
    FPS = 25

    def easing(x, beta):
        b = beta
        return 1 / (1 + math.pow(x / (1 - x + 1e-8), -b))

    def get_normalized_distances(seeds, frames):

        vecs = [seed2vec(model.model, s) for s in seeds]
        dist = []

        for t in range(len(vecs) - 1):
            dist.append(((vecs[t]-vecs[t + 1])**2).sum(axis=1).item()) #Euclidian distance
        
        dist = np.array(dist)
        dist /= np.average(dist)
        factor = len(dist) * frames / sum(dist)
        dist = (factor * dist).astype("int")
        
        return(dist)

    def render_seq_bttn_click(b):
        with output3:
            clear_output()
            assert seeds_updater.seed_list
            seeds = seeds_updater.seed_list
            render_sequence(model, seeds, steps_slider.value, sequence_folder, easing_slider.value, loop_chkbx.value)

    def render_vid_bttn_click(b):
        with output3:
            clear_output()
            assert len(os.listdir(sequence_folder)) != 0
            assert seeds_updater.seed_list
            SEEDS = seeds_updater.seed_list
            create_video(sequence_folder, video_folder, fps_text.value, SEEDS)

    def render_sequence(model, seeds, num_steps, output_folder, easy_ease = 1, loop = True):
        if loop and seeds[-1] != seeds[0]:
            seeds.append(seeds[0])

        distances_norm = get_normalized_distances(seeds, num_steps)

        os.system(f"rm {os.path.join(sequence_folder, '*')}")

        idx = 0
        tqdm_progress = tqdm(range(len(seeds)-1), desc = "", leave=True)

        for i in tqdm_progress:
            w1 = seed2vec(model.model, seeds[i])
            w2 = seed2vec(model.model, seeds[i+1])

            diff = w2 - w1
            step = diff / distances_norm[i]
            current = w1.clone().detach()

            for s, j in enumerate(range(distances_norm[i])):
                tqdm_progress.set_description(f"State: {i + 1}/{len(seeds) - 1} | Frame: {i*distances_norm[i] + s} / {(len(seeds) - 1) * distances_norm[i]}")
                tqdm_progress.refresh()

                now = current + diff * easing((s + 0.01 ) / distances_norm[i], easy_ease)
                img = generate_image(model.model, now)
                img.save(os.path.join(output_folder,f'frame-{idx}.png'))
                idx+=1

        print("Rendering video")
        create_video(sequence_folder, video_folder, fps_text.value, seeds)
        print("Finished rendering")

    def create_video(sequence_folder, output_folder, FPS, seeds):
        seeds_list = "_".join([str(s) for s in seeds])
        input_sequence = os.path.join(sequence_folder, "frame-%d.png")
        img = Image.open(os.path.join(sequence_folder, os.listdir(sequence_folder)[0]))
        output_file = os.path.join(output_folder, f"{model.prefix}_{seeds_list}.mp4")
        os.system(f"ffmpeg -r {FPS} -i {input_sequence} -c:v libx264 -b:v 15M -pix_fmt yuv420p {output_file} -y")
        clear_output()

    steps_slider = widgets.IntSlider(min=10, max=1000, step=10, value = STEPS, description='Frames between seeds')
    easing_slider = widgets.FloatSlider(min=1, max=2, step=0.01, value = easy_ease, description='Easing')
    fps_text = widgets.Dropdown(options=['5', '10', '12', '15', '20', '24', '25', '30'],value=str(FPS),description='FPS',disabled=False)
    loop_chkbx = widgets.Checkbox(value=loop,description='Loop',disabled=False,indent=False)
    sliders = widgets.HBox([steps_slider, easing_slider, fps_text, loop_chkbx])

    render_seq_bttn = widgets.Button(description="Render sequence")
    render_vid_bttn = widgets.Button(description="Compile video")
    bttns = widgets.HBox([render_seq_bttn, render_vid_bttn])

    render_seq_bttn.on_click(render_seq_bttn_click)
    render_vid_bttn.on_click(render_vid_bttn_click)

    output3 = widgets.Output()

    return(sliders, bttns, output3)

def get_model_loader(model, models_folder = "/content/models"):
    models_list = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(models_folder)]
    models_paths = [os.path.join(models_folder,f) for f in os.listdir(models_folder)]
    models_dict = dict(zip(models_list, models_paths))

    def load_model_onclick(b):
        with output_model_select:
            if(models_select.value):
                model.update_name_path(models_select.value, models_dict[models_select.value])
                clear_output()
                print(f"Model {models_select.value} selected")

    models_select = widgets.Dropdown(options=models_list,description='Model',disabled=False)
    load_model_bttn = widgets.Button(description="Load model")
    bttns = widgets.HBox([models_select, load_model_bttn])
    load_model_bttn.on_click(load_model_onclick)

    output_model_select = widgets.Output()

    load_model_onclick(load_model_bttn) #autoclick

    return bttns, output_model_select
