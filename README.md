# voronoi-assisted-spherical-classifiers
Course Project for CMSC828W@UMD Fall 2022

Code based on @mariogeiger 's e3nn-based [implementation](https://github.com/e3nn/e3nn/tree/main/examples/s2cnn/mnist) of the spherical CNN from [https://arxiv.org/abs/1801.10130](https://arxiv.org/abs/1801.10130). 

In addition to that, we mask each image in the spherical MNIST dataset and use Voronoi tessellation as a precursor to training a classifier on the degraded data. We show that Voronoi-assisted classifiers outperform unassisted classifiers throughout all mask parameters.

TODO: Update with instructions on how to run and package versions.