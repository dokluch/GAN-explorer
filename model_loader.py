def get_model_loader(model):
    models_list = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("/content/models")]
    models_paths = [os.path.join("/content/models",f) for f in os.listdir("/content/models")]
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
