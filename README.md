# SKS-ACA Decomposition of 2D Homography

This is the offical implementation of the paper "Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity (SKS) and Affine-Core-Affine (ACA)". SKS and ACA are novel decomposition forms for 2D homography (projective transformation) matrices, which are superior to previous methods (NDLT-SVD, HO-SVD, GPT-LU, RHO-GE) in terms of computational efficiency, geometrical meaning of parameters, and unified management for various planar configurations.

The uploaded codes include the Matlab, C++ (with OpenCV or CUDA library) and Python (with PyTorch library) procedures used in CPU and GPU experiments.

## SKS Decomposition

SKS decomposes a 2D homography into three sub-transformation: 

```math
H=H_{S_2}^{-1}*H_{K}*H_{S_1}

$\alpha$

