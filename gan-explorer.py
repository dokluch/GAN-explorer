def get_timeline_controls():
    button_get_random = widgets.Button(description="Get random seed")
    button_prev = widgets.Button(description="<<<")
    button_next = widgets.Button(description=">>>")
    buttons_line_1 = widgets.HBox([button_prev, button_get_random, button_next])

    output = widgets.Output()

    button_add_seed = widgets.Button(description="Add seed")
    button_remove_last = widgets.Button(description="Remove last seed")
    buttons_line_2 = widgets.HBox([button_add_seed, button_remove_last])

    output2 = widgets.Output()

    def on_save_clicked(b):
        with output2:
            clear_output()
            if(b.seeds):
                if b.seeds[-1] != button_get_random.seed:
                    b.seeds.append(button_get_random.seed)
                    b.imgs.append(button_get_random.img)
                else:
                    b.seeds.append(button_get_random.seed)
                    b.imgs.append(button_get_random.img)
                    print(b.seeds)
                    display_seeds_as_imgs()

    def on_remove_last(b):
        with output2:
            clear_output()
            if(button_add_seed.seeds):
                button_add_seed.seeds = button_add_seed.seeds[:-1]
                button_add_seed.imgs = button_add_seed.imgs[:-1]
                print(button_add_seed.seeds)
                display_seeds_as_imgs()


    def display_seeds_as_imgs():
        if button_add_seed.imgs:
            ipyplot.plot_images(button_add_seed.imgs, labels = button_add_seed.seeds, img_width=200)

    def on_button_clicked(b):
        with output:
            clear_output()
            seed_gen = np.random.randint(0, 400000)
            print(seed_gen)
            b.img = make_img_from_seed(seed_gen).resize((256,256))
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
                button_get_random.img = make_img_from_seed( button_get_random.seed).resize((256,256))
                clear_output()
                print(button_get_random.seed)
                display(button_get_random.img)
                print(button_get_random.prev_seeds)

    def on_next(b):
        with output:
            if len(button_get_random.prev_seeds) > 1 and button_get_random.pos < len(button_get_random.prev_seeds) - 1:
                button_get_random.pos += 1
                button_get_random.seed = button_get_random.prev_seeds[button_get_random.pos]
                button_get_random.img = make_img_from_seed( button_get_random.seed).resize((256,256))
                clear_output()
                print(button_get_random.seed)
                display(button_get_random.img)
                print(button_get_random.prev_seeds)


    button_add_seed.seeds = []
    button_add_seed.imgs = []
    button_add_seed.on_click(on_save_clicked)
    button_remove_last.on_click(on_remove_last)

    button_get_random.prev_seeds = []
    button_get_random.on_click(on_button_clicked)
    button_prev.on_click(on_prev)
    button_next.on_click(on_next)
    on_button_clicked(button_get_random)

    return(output, buttons_line_1, buttons_line_2, output2)


def get_render_controls():
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


    def render_seq_bttn_click(b):
        with output3:
            clear_output()
            assert button_add_seed.seeds
            seeds = button_add_seed.seeds
            render_sequence(seeds, steps_slider.value, sequence_folder, easing_slider.value, loop_chkbx.value)

    def render_vid_bttn_click(b):
        with output3:
            clear_output()
            assert len(os.listdir(sequence_folder)) != 0
            assert button_add_seed.seeds
            SEEDS = button_add_seed.seeds

    create_video(sequence_folder, video_folder, fps_text.value, SEEDS)

    def render_sequence(seeds, num_steps, output_folder, easy_ease = 1, loop = True):
        STEPS = num_steps

        if loop and seeds[-1] != seeds[0]:
            seeds.append(seeds[0])

            !rm {os.path.join(sequence_folder, "*")}

            idx = 0
            tqdm_progress = tqdm(range(len(seeds)-1), desc = "", leave=True)

            for i in tqdm_progress:
                v1 = seed2vec(seeds[i])
                v2 = seed2vec(seeds[i+1])

                diff = v2 - v1
                step = diff / STEPS
                current = v1.clone().detach()

                for s, j in enumerate(range(STEPS)):
                    tqdm_progress.set_description(f"State: {i + 1}/{len(seeds) - 1} | Frame: {i*STEPS + s} / {(len(seeds) - 1) * STEPS}")
                    tqdm_progress.refresh()

                    now = current + diff * easing((s + 0.01 ) / STEPS, easy_ease)
                    img = generate_image(G, now, 1.0)
                    img.save(os.path.join(output_folder,f'frame-{idx}.png'))
                    idx+=1

                    print("Rendering video")
                    create_video(sequence_folder, video_folder, fps_text.value, seeds)
                    print("Finished rendering")

    def create_video(sequence_folder, output_folder, FPS, seeds):
        seeds_list = "_".join([str(s) for s in seeds])
        input_sequence = os.path.join(sequence_folder, "frame-%d.png")
        img = Image.open(os.path.join(sequence_folder, os.listdir(sequence_folder)[0]))
        output_file = os.path.join(output_folder, f"{prefix}_{seeds_list}.mp4")
        !ffmpeg -r {FPS} -i {input_sequence} -c:v libx264 -b:v 15M -pix_fmt yuv420p {output_file} -y
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