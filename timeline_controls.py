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