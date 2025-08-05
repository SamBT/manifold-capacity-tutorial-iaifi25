import matplotlib.pyplot as plt
import numpy as np

def network_capacity_summary(alphas, radii, dims, mean_alphas, mean_radii, mean_dims, layer_names, labels):
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # X-axis labels for all plots
    x_labels = layer_names
    x_values = np.arange(len(layer_names))

    # Plot alpha values (a)
    plt.sca(axes[0])
    plt.plot(x_values, mean_alphas, 'o-', color='blue', markersize=8,label='Average')
    for l in labels:
        plt.plot(x_values, alphas[l], 'x-', markersize=4, alpha=0.5, label=f'Label {l}')
    plt.legend(loc='upper left', fontsize=10,ncol=4)
    plt.xticks(x_values, labels=x_labels, rotation=45, fontsize=12)
    plt.title(r"Manifold Capacity $\alpha_M$",fontsize=16)
    plt.ylabel(r"$\alpha_M$", fontsize=22)
    plt.yticks(fontsize=12)

    # Plot R values (a)
    plt.sca(axes[1])
    plt.plot(x_values, mean_radii, 'o-', color='blue', markersize=8)
    for l in labels:
        plt.plot(x_values, radii[l], 'x-', markersize=4, alpha=0.5)
    plt.xticks(x_values, labels=x_labels, rotation=45, fontsize=12)
    plt.title(r"Manifold Radius $R_M$",fontsize=16)
    plt.ylabel(r"$R_M$", fontsize=22)
    plt.yticks(fontsize=12)

    # Plot D values (a)
    plt.sca(axes[2])
    plt.plot(x_values, mean_dims, 'o-', color='blue', markersize=8)
    for l in labels:
        plt.plot(x_values, dims[l], 'x-', markersize=4, alpha=0.5)
    plt.xticks(x_values, labels=x_labels, rotation=45, fontsize=12)
    plt.title(r"Manifold Dimension $D_M$",fontsize=16)
    plt.ylabel(r"$D_M$", fontsize=22)
    _=plt.yticks(fontsize=12)