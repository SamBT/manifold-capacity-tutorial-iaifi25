'''
Packaged version of the analysis for pytorch models and datasets
'''

import numpy as np
import torch
from collections import OrderedDict

from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor

def analyze(model, dataset, sampled_classes=50, examples_per_class=50, kappa=0, n_t=300, n_reps=1,
        max_class=None, projection=True, projection_dimension=5000, layer_nums=None, layer_types=None, 
        verbose=True, cuda=True, seed=0, use_tensors=True):
    '''
    Bundled analysis for PyTorch models and datasets. Automatically flattens all features.

    Args:
        model: PyTorch model to analyze.
        dataset: PyTorch style dataset or iterable containing (input, label) pairs.
        sampled_classes: Number of classes to sample in the analysis (Default 50).
        examples_per_class: Number of examples per class to use in the analysis (Default 50).
        kappa: Size of margin to use in analysis (Default 0)
        n_t: Number of t vectors to sample (Default 300)
        n_reps: Number of repititions to use in correlation analysis (Default 1)
        max_class: ID of the largest class to choose in sampling.
        projection: Whether or not to project the data to a lower dimension.
        projection_dimension: Dimension above which data is projected down to projection_dimension.
        layer_nums: Numbers of layers to analyze. Ex: [1, 2, 4]
        layer_types: Types of layers to use in analysis. Ex: ['Conv2d', 'Linear']. Only use if
            layer_nums isn't specified.
        verbose: Give updates on progress (Default True)
        cuda: Whether or not to use a GPU to generate activations (Default True)
        seed: Random seed.
        use_tensors: Whether to use PyTorch tensors throughout computation (Default True)

    Returns:
        results: Dictionary of results for each layer.
    '''
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    # Create the manifold data
    manifold_data = make_manifold_data(dataset, sampled_classes, examples_per_class, seed=seed)
    # Move the model and data to the device
    model = model.to(device)
    manifold_data = [d.to(device) for d in manifold_data]
    # Extract the activations
    activations = extractor(model, manifold_data, layer_nums=layer_nums, layer_types=layer_types, return_tensors=use_tensors)
    # Set the seed for random projections
    if use_tensors:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    # Preprocess activations for analysis
    for layer, data, in activations.items():
        if use_tensors:
            X = [d.reshape(d.shape[0], -1).T for d in data]
        else:
            X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # Optionally project the features to a lower dimension
        if projection and N > projection_dimension:
            # Create a projection matrix
            if use_tensors:
                M = torch.randn(projection_dimension, N, device=device)
                M /= torch.sqrt(torch.sum(torch.square(M), dim=1, keepdim=True))
                # Project the data
                X = [torch.matmul(M, d) for d in X]
            else:
                M = np.random.randn(projection_dimension, N)
                M /= np.sqrt(np.sum(np.square(M), axis=1, keepdims=True))
                # Project the data
                X = [np.matmul(M, d) for d in X]
        activations[layer] = X
    # Create storage for the results
    results = OrderedDict()
    # Run the analysis on each layer that has been selected
    for k, X, in activations.items():
        analyze_layer = False
        if layer_nums is not None and int(k.split('_')[1]) in layer_nums:
            analyze_layer = True
        elif layer_types is not None and k.split('_')[-1] in layer_types:
            analyze_layer = True
        elif layer_nums is None and layer_types is None:
            analyze_layer = True

        if analyze_layer:
            if verbose:
                print('Analyzing {}'.format(k))
            if use_tensors:
                a, r, d, r0, K = manifold_analysis_corr(X, kappa, n_t, n_reps=n_reps, device=device)
                # Convert results to numpy for consistency with original interface
                a = a.cpu().numpy()
                r = r.cpu().numpy()
                d = d.cpu().numpy()
                if isinstance(r0, torch.Tensor):
                    r0 = r0.item()
            else:
                a, r, d, r0, K = manifold_analysis_corr(X, kappa, n_t, n_reps=n_reps)
            # Store the results
            results[k] = {}
            results[k]['capacity'] = a
            results[k]['radius'] = r
            results[k]['dimension'] = d
            results[k]['correlation'] = r0
            results[k]['K'] = K
            results[k]['feature dimension'] = X[0].shape[0]
    return results
