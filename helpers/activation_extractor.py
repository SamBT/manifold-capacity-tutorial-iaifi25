'''
Tool for extracting activations from pytorch models
'''

from collections import OrderedDict
import torch
import numpy as np

# Global storage for extracted activations
extracted_dict = OrderedDict()


def extractor(model, data, labels, layer_nums=None, layer_types=None):
    '''
    Extract model activations on the given data for the specified layers
    
    Args:
        model: Model to extract activations from
        data: Iterable containing batches of inputs to extract activations from
        layer_nums (optional): Numbers of layers ot extract activations for. If None,
            activations from all layers are returned.
        layer_types (optional): Names of layers to extract activations from. If None
            activations from all layers are returned. Only use this or layer_nums

    Returns:
        extracted_dict: Dictionary containing extracted activations. Order matches
            the order of the given data.
    '''
    assert (layer_nums is None or layer_types is None), 'Only specify one of layer_nums or layer_types'
    global extracted_dict
    extracted_dict = OrderedDict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find all the layers that match the specified types
    flat_children = []
    leaf_traverse(model, flat_children)
    add_layer_names(flat_children)
    flat_children = filter_layers(flat_children, layer_types, layer_nums)

    # Register hooks to the found layers
    registered_hooks = register_hooks(flat_children)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        extracted_dict['inputs'] = data.cpu().numpy().reshape(data.shape[0], -1)  # Flatten inputs
        extracted_dict['labels'] = labels.cpu().numpy() if labels is not None else None
        _ = model(data.to(device))
    # Remove the hooks
    deregister_hooks(registered_hooks)

    layer_names = [name for name in extracted_dict.keys() if name != 'labels']

    # Return the activations
    return extracted_dict, layer_names


def activation_extractor(self, input, output):
    '''
    Hook to extract activations
    '''
    global extracted_dict
    layer_name = self.layer_name

    # Store the activations
    #if layer_name not in extracted_dict:
    #    extracted_dict[layer_name] = []
    #extracted_dict[layer_name] += [output.data.cpu().numpy()]
    extracted_dict[layer_name] = output.data.cpu().numpy().reshape(output.shape[0], -1)  # Flatten activations


def leaf_traverse(root, flat_children):
    '''
    Get all the layers of the model
    '''
    if len(list(root.children())) == 0:
        flat_children.append(root)
    else:
        for child in root.children():
            leaf_traverse(child, flat_children)

def register_hooks(flat_children):
    '''
    Register the activation extraction hook on each of the given layers
    '''
    registered_hooks = []
    for child in flat_children:
        hook = child.register_forward_hook(activation_extractor)
        registered_hooks.append(hook)
    return registered_hooks


def deregister_hooks(registered_hooks):
    '''
    Remove hooks
    '''
    for hook in registered_hooks:
        hook.remove()


def add_layer_names(flat_children):
    '''
    Count the layers in the model and add names to them
    '''
    count = 1
    for child in flat_children:
        name = "layer_" + str(count) + "_" + child._get_name()
        child.__setattr__('layer_name', name)
        count += 1


def filter_layers(flat_children, layer_types, layer_nums):
    '''
    Retain layers that match layer_types
    '''
    if layer_types is None and layer_nums is None:
        filtered_children = flat_children
    elif layer_nums is not None:
        filtered_children = []
        for layer in flat_children:
            if int(layer.layer_name.split('_')[1]) in layer_nums:
                filtered_children.append(layer)
    elif layer_types is not None:
        filtered_children = []
        for layer in flat_children:
            if layer._get_name() in layer_types:
                filtered_children.append(layer)
    return filtered_children
