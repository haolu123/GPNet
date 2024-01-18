import torch.nn as nn

def Hook_register(model, layer_name_list, activation):
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    def get_layer(model, layer_path):
        if isinstance(layer_path, str):
            return model._modules.get(layer_path)
        elif isinstance(layer_path, list):
            for layer in layer_path:
                model = model._modules.get(layer)
            return model
        else:
            raise ValueError("Invalid layer path")
    for layer_name in layer_name_list:
        target_layer = get_layer(model, layer_name)
        if target_layer is not None:
            layer_id = '.'.join(layer_name) if isinstance(layer_name, list) else layer_name
            target_layer.register_forward_hook(get_activation(layer_id))
        else:
            print(f"Layer {layer_name} not found in the model")
    return activation

def run_with_hook(model, features1_count, features2_gene_idx):
    pred, _, _ = model(features1_count, features2_gene_idx)
    return pred