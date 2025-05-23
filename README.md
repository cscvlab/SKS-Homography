# SKS & ACA Decomposition of 2D Homography
This repository is the official implementation of the paper (early access for PAMI): 

**Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity and Affine-Core-Affine Transformations**.

__Authors:__ Shen Cai*, Zhanhao Wu, Lingxi Guo, Jiachun Wang, Siyu Zhang, Junchi Yan,  Shuhan Shen*.

Previous four-point homography computation methods exhibit algebraic redundancy (arising from the construction of a sparse linear system) and geometric isolation (no link to other 2D primitives, transformation computations, and minimal vision problems). Our SKS and ACA methods offer distinct advantages in various aspects and demonstrate extreme efficiency. The uploaded codes include the Matlab, C++ (with OpenCV or CUDA library) and Python (with PyTorch library) procedures used in CPU and GPU experiments.

**Links:**  [[Paper]](https://arxiv.org/pdf/2402.18008) 
[[Project Page]](http://www.cscvlab.com/research/SKS-Homography/) 
[[ShortVideo(bilibili)]](https://www.bilibili.com/video/BV1iLZ3YyEnR/)
[[LongVideo(bilibili)]](https://www.bilibili.com/video/BV1f3E7zCE3N/)
[[SV(YouTube)]](https://youtu.be/jVQ-6ub70K0)
[[LV(YouTube)]](https://youtu.be/IFZ9jrLvZok)

## SKS Decomposition
SKS decomposes a 2D homography into three sub-transformations: 
```math
\mathbf{H}=\mathbf{H}_{S_2}^{-1}*\mathbf{H}_{K}*\mathbf{H}_{S_1},
```
where $\mathbf{H}\_{S\_1}$ and $\mathbf{H}\_{S\_2}$ are similarity transformations induced by two arbitrary pairs of corresponding points on source plane and target plane, respectively; $\mathbf{H}\_{K}$ is the 4-DOF kernel transfromation we defined, which generates projective distortion between two similarity-normalized planes. 

## ACA Decomposition
ACA also decomposes a 2D homography into three sub-transformations: 
```math
\mathbf{H}=\mathbf{H}_{A_2}^{-1}*\mathbf{H}_{C}*\mathbf{H}_{A_1},
```
where $\mathbf{H}\_{A\_1}$ and $\mathbf{H}\_{A\_2}$ are affine transformations induced by three arbitrary pairs of corresponding points on source plane and target plane, respectively; $\mathbf{H}\_{C}$ is the 2-DOF core transfromation we defined, which generates projective distortion between two affinity-normalized planes.

## Rich Geometric Meanings
In SKS and ACA, each sub-transformation and even each parameter carry geometric significance. Specifically:
1. In SKS, the projective distortion induced by $\mathbf{H}_K$ ties to hyperbolic similarity transformations.
2. SKS, using two anchor points, offers a unified solution for various 2D primitives, including lines and conics, as well as their hybrid patterns.
3. The stratified geometric transformation extends the existing SAP decomposition and encompasses affine transformations.

## Algebraic Simplicity
SKS and ACA exhibit many unique properties in algebra, some of which are shown below.

### No Need to Construct A Linear System of Equations
Previous 4-point homography methods follow the same way to construct a square system of linear equations, followed by solving it through well-established matrix factorization methods, such as SVD and LU,
```math
\mathbf{A}_{8*9}*\mathbf{h}_{9*1}=\mathbf{0} \quad \mathcal{or} \quad \mathbf{A}_{8*8}*\mathbf{h}_{8*1}=\mathbf{b}_{8*1}.
```
Such approachs are circuitous since the constructed coefficient matrix $\mathbf{A}$ is redundant (including a number of 0 and 1). Conversely, SKS and ACA directly compute the sub-transformations of homography in a stratified way. 

### Division-Free Solver 
ACA is extremely concise in algebra and only requires 85 addtions, subtractions and multiplications of floating-point numbers to compute homographies up to a scale. Among four arithmetic operations, the most complicated division is avoided in ACA. 

### Floating-point Operations (FLOPs)
FLOPs of SKS and ACA for computing 4-point homographies up to a scale are 157 and 85 respectively. With the normalization based on the last element of homography (which costs 12 extra FLOPs), FLOPs of SKS and ACA are 169 and 97 respectively. Compared with commonly used robust 4-point homography solvers NDLT-SVD ($\ge$27K FLOPs) and GPT-LU (~1950 FLOPs), SKS and ACA represent {162x, 12x} and {282x, 20x}, respectively.

### Polynomial Expression of Each Element of Homography
Owing to the extremely simple expression of each sub-transformation, we can represent each element of a homography by the input variables ($16$ coordinates provided by four point correspondences) in 7th to 9th polynomial form.

### Homographies Mapping A Rectangle to A Quadrangle
All previous 4-point offsets based deep homography methods compute the homography mapping a square or rectangle in source image to a general quadrangle in target image. However, the previous methods treat the special rectanlge as a general quadrangle and no simplification is conducted. In SKS and ACA, homographies mapping a rectangle (or square) to a quadrangle are simplified straightforwardly. The complete steps of the tensorized ACA (TensorACA) for a rectangle are illustrated in the following Algorithm with only 15 vector operations (47 FLOPs). Consequently, FLOPs for a source square will be reduced to 14 vector operations (44 FLOPs).

$\mathbf{Note:}$ When calculating homography in a very common computer vision task—QR code detection, FLOPs are further reduced to 29. This drastic reduction lowers energy consumption, given the massive number of QR code scans performed daily.

<!-- <div align="center"> <img src="imgs/ACA-rect.png" width = 60% /> </div> -->

### A Unified Way to Decompose and Compute Affine Transformations
Affine transformations, as one kind of degenerate projective transformations, can also be managed by SKS and ACA in a unified way. Therefore, FLOPs of computing affine transformations with three points is significantly reduced, especially compared to the GPT-LU method (see OpenCV's function 'getAffineTransform').


## Citation

```bibtex
@article{Cai2025SKS,
  title={Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity and Affine-Core-Affine Transformations},
  author={Cai, Shen and Wu, Zhanhao and Guo, Lingxi and Wang, Jiachun and Zhang, Siyu and Yan, Junchi and Shen, Shuhan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3568582}
}
```


