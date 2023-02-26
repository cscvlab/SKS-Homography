# SKS-ACA Decomposition of 2D Homography

This is the offical implementation of the paper "Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity (SKS) and Affine-Core-Affine (ACA)". SKS and ACA are novel decomposition forms for 2D homography (projective transformation) matrices, which are superior to previous methods (NDLT-SVD, HO-SVD, GPT-LU, RHO-GE) in terms of computational efficiency, geometrical meaning of parameters, and unified management for various planar configurations.

The uploaded codes include the Matlab, C++ (with OpenCV or CUDA library) and Python (with PyTorch library) procedures used in CPU and GPU experiments.

## SKS Decomposition

SKS decomposes a 2D homography into three sub-transformation: 

```math
\mathbf{H}=\mathbf{H}_{S_2}^{-1}*\mathbf{H}_{K}*\mathbf{H}_{S_1}
```

where $\mathbf{H}_{S_2}$ and $\mathbf{H}_{S_1}$ are similarity transformations induced by two arbitrary points on target plane and source plane, respectively; $\mathbf{H}_{K}$ is the 4-DOF kernel transfromation we defined, which generates projective distortion between two similarity-normalized planes. 

## ACA Decomposition

ACA also decomposes a 2D homography into three sub-transformation: 

```math
\mathbf{H}=\mathbf{H}_{A_2}^{-1}*\mathbf{H}_{C}*\mathbf{H}_{A_1}
```

where $\mathbf{H}_{A_2}$ and $\mathbf{H}_{A_1}$ are affine transformations induced by three arbitrary points on target plane and source plane, respectively; $\mathbf{H}_{C}$ is the 2-DOF core transfromation we defined, which generates projective distortion between two affinity-normalized planes.

## Floating-point operations (FLOPs)

Floating-point operations (FLOPs) of SKS:


fda 

$\alpha$

