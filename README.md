# Computing With Neural Manifolds Tutorial
The exercises in this repository are designed to accompany Prof. SueYeon Chung's lectures on Computing with Neural Manifolds at the 2025 [IAIFI Summer School](https://iaifi.org/phd-summer-school.html). The `main` branch contains notebooks with incomplete code, with sections for you to fill in as you work through them. If at any point you need solutions, you can see them in the `solutions` branch.

## Getting set up
Ideally you should clone this repo somewhere where you have access to a GPU, as tutorials 2 and 3 involve training/evaluating some large-ish neural networks. There are two likely scenarios:

1. **Google Colab** (links below) - this should work out of the box, but I haven't tested it thoroughly. **Run the cell at the top of each notebook to do the necessary pip installs and to clone the repo**.

2. **On a cluster/personal machine** - Use a relatively recent version of python (tested with 3.12) and make sure `pytorch` and `torchvision` are installed (install instructions [here](https://pytorch.org/), make sure you get the GPU-compatible versions if working on a GPU node). Install the remaining required packages from `requirements.txt`
```bash
pip install requirements.txt
```

## The tutorials

1. **Tutorial 1** - implementing the mean-field calculations of manifold capacity, radius, and dimension. Doesn't require any GPUs! <a target="_blank" href="https://colab.research.google.com/github/SamBT/manifold-capacity-tutorial-iaifi25/blob/main/Tutorial_1_theory.ipynb"> 
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

2. **Tutorial 2** - training some neural networks on image data and analyzing learned representations using mean-field manifold capacity quantities. <a target="_blank" href="https://colab.research.google.com/github/SamBT/manifold-capacity-tutorial-iaifi25/blob/main/Tutorial_2_neuralNets.ipynb"> 
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
</a>

3. **Tutorial 3** - exploring [MMCR](https://arxiv.org/abs/2303.03307), a self-supervised learning technique using a manifold capacity-inspired loss function. <a target="_blank" href="https://colab.research.google.com/github/SamBT/manifold-capacity-tutorial-iaifi25/blob/main/Tutorial_3_MMCR.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>