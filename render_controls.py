def get_render_controls(model, seeds_updater):
    STEPS = 100
    easy_ease = 1
    loop = True
    sequence_folder = "/content/sequence"
    video_folder = "/content/renders"
    SEEDS = [39644, 35189, 4531, 11258, 7987] #MANUANAL
    FPS = 25

    def easing(x, beta):
        b = beta
        return 1 / (1 + math.pow(x / (1 - x + 1e-8), -b))

    def get_normalized_distances(seeds, frames):

        vecs = [seed2vec(s) for s in seeds]
        dist = np.array([])

        for t in range(len(vecs) - 1):
            dist.append(((vecs[t]-vecs[t + 1])**2).sum(axis=1).item()) #Euclidian distance
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
            v1 = seed2vec(model.model, seeds[i])
            v2 = seed2vec(model.model, seeds[i+1])

            diff = v2 - v1
            step = diff / distances_norm[i]
            current = v1.clone().detach()

            for s, j in enumerate(range(distances_norm[i])):
                tqdm_progress.set_description(f"State: {i + 1}/{len(seeds) - 1} | Frame: {i*distances_norm[i] + s} / {(len(seeds) - 1) * distances_norm[i]}")
                tqdm_progress.refresh()

                now = current + diff * easing((s + 0.01 ) / distances_norm[i], easy_ease)
                img = generate_image(model.model, now, 1.0)
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
        os.system(f"ffmpeg -r {FPS} -i {input_sequence} -c:v libx264 -b:v 15M -pix_fmt yuv422p10le {output_file} -y")
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