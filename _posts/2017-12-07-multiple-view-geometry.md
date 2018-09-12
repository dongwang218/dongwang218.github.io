---
layout: post
mathjax: true
comments: true
title:  "Multiple View Geometry in Computer Vision"
---
This blog is about Multiple View Geometry in Computer Vision by Richard Hartley and Andrew Zisserman. I will mainly take some notes about the gold standard algorithms in this book. All figures are blatantly copied from the book.

# 3D Space

Concepts:
* Plane is a 3 vector. 3 point define a plane, 3 planes define a point.
* Quadrics is a surface x^TQx = 0, Q is 4x4. its 2d counter part is conic.
* plane at infinity $\pi_\infty = (0, 0, 0, 1)^T$
* abosulte conic $\Omega_\infty$ is conic at infinity, $x_1^2 + x_2^2 + x_3^2 = 0$ and $x_4 = 0$

3D transoforms
![3D Transforms](/assets/mvgtable3.2.png)

# 2D homography

A plane in two views are related by a homography $H_{3x3}$.  Two views if camera is only rotated about its center also has a homography. Given $x_i$ and $x_i^{\prime}$ are 2D correpondence point, use homogenous coordindates. From $x_i^{\prime} = \eta_i H x_i$, thus $x_i^{\prime} \times H x_i = 0$. let $\textbf{x}^{\prime} = [x_i^{\prime}, y_i^{\prime}, w_i^{\prime}]$, $h$ is the ith row of $h$. 4 points give 8 constraints $Ah = 0$, $A$ is $2nx9$. Let $A = UDV^T$, then $h$ is the last column of $V$. This is the **Direct Linear Transform (DLT) algorithm** (Algorithm 4.1 on page 91).

![Homography](/assets/mvgequation4.1.png)
An improved algorithm is Algorithm 4.2 on page 109, it does prenormalization before calling Algorithm 4.1, so that the mean of $x$ are (0, 0) and average distance to origin is $\sqrt{2}$.

Given $H$, the projection of $x$ is $\hat{x}_i^{\prime} = Hx$, for $x^{\prime}$, its projection is $\hat{x}_i = H^{-1} x^{\prime}$. Use optimization to iteratively reduce reprojection error. This is the **gold standard Algorithm 4.3 on page 114 for estimating H from image correspondences**.
![Homography](/assets/mvgalgorithm4.3.png)

RANSAC randomly selects random candidate and measure num of inliners.
* Threshold: sum of squares of distance is a $\xi_m^2$ distribution.
* samples: $(1-w^s)^N = 1 - p$, so $w$ is inlier probability, $s$ is num of points, $N$ is num of selections, $p$ is confidence. When $w$ is unknown, can estimated it from current model.
**Homography using RANSAC** Algorithm 4.6 on page 123.
![RANSAC](/assets/mvgalgorithm4.6.png)

# Camera models

![Pinhole camera](/assets/mvgfigure6.1.png)
![Equation 6.10](/assets/mvgequation6.10.png)
![Equation 6.11](/assets/mvgequation6.11.png)
Image point $x = P X$, $K$ is 3x3 upper-triangular matrix, $\alpha_x, \alpha_y$ is the focal length in pixels. $R$ is a 3x3 rotation matrix whose columns are the directions of the world axes in the camera's reference frame. The vector $\tilde{C}$ is the camera center in world coordinates; the vector $t = -R\tilde{C}$ gives the position of the world origin in camera coordinates according to this [blog](http://ksimek.github.io/2012/08/14/decompose/). Rotation can be  decomposed into rotation along x, y and z axes
![Equation 6.11](/assets/mvgrotation.png)
![P](/assets/mvgtable6.1.png)
Recover intrinsic and extrinsic parameter from $P=[M \mid p_4]$, so $\tilde{C} = -M^{-1}p_4$. $K$ and $R$ can be separated by RQ-decomposition of $M$.

Cameras at infinity is affine camera, with last row of $P$ is (0, 0, 0, 1).
* Orthographic projection map (X,Y,Z,1)^T to (X, Y,1)^T.
![Orthographic 6.11](/assets/mvgequation6.23.png)
* Weak perspecitive projection, is scaled orthographic projection, where $\alpha_x$ and $\alpha_y$ not equal.

# Compute Camera Matrix P

P is $3x4$, similar to the 2d homograph derivation, because $x = PX$, we will have the following constraint
![Equation 7.1](/assets/mvgequation7.1.png)
Similarly to solve $P$ with at least 6 points, this is the Algorithm 7.1 **gold standard algorithm to estimate P**, this is implemented in the popular camera calibration code in opencv with calibration grid. In fact, it also considers radial distortions.
![Algorithm 7.1](/assets/mvgalgorithm7.1.png)

For affine cameras, we only need to estimate the first two rows, this results in the following Algorithm 7.2.
![Algorithm 7.2](/assets/mvgalgorithm7.2.png)

# Single View

Vanishing point: a line in world $X(\lambda) = A + \lambda D$, D = (d^T, 0)^T. The vanishing point is the ray parallel to world line from camera center intersection with image plane, ie $Kd$. Two parallel world lines intersect at vanishing point.

Vanishing line: Intersection of image plane by a plane going through camera center parallel to the plane. $l = K^{-T} n$. Or connecting two vanishing points on the plane. Three equally spaced parallel lines can determine the vanishing line.

Algorithm 8.1 can compute ratio of two scene lines perpendicular to the ground plane.

# Epipolar Geometry

In two views, a point in one view define a epipolar line in the other view on which the corresponding point lies, main concepts:
* epipole is the intersection of image plane by the line (baseline) joining the camera centers.
* All epipolar line intersect at the epipole.
* Fundamental matrix $F$ is $3x3$, rank 2, corresponding image points $x'^T F x = 0$
* $P$ and $P^\prime$ may be computed from $F$ up to a projective ambiguity of 3-space, can be chosen as $P = [I \mid 0]$ and $P^\prime = [[e^\prime]_x F \mid e^\prime]$.
* For image point $x$, normalized coordindates of it $\hat{x} = K^{-1} x$
* Essential matrix $E$, two corresponding normalized coordindates has $\hat{x}' E \hat{x} = 0$.
* $E = K'^T F K$
* $E$ is a essential matrix iff two of its singular values are equal, the third is zero.
* If $E = U diag(1, 1, 0) V^T$, first camera $P = [I \mid 0]$, there are four choices for the second matrix $P'$
$P' = [U W V^T \mid \pm u_3]$ or $[U W^T V^T \mid \pm u_3]$
* If camera is affine, the first 2x2 of the fundamental matrix is 0, only need at least 4 point to compute $F$.

Properties of Fundamental matrix.
![Table 9.1](/assets/mvgtable9.1.png)


If $a=(x, y, z)^T$ then
$$
[a]x = \begin{pmatrix}
0 & -z & y \\
z & 0 & -x \\
y & x & 0
\end{pmatrix}
$$

# Computing Fundamental Matrix
$f$ is the 9 vector from $F$ in row-major order, $x'Fx=0$ become
![Equation 11.3](/assets/mvgequation11.3.png)
![Algorithm 11.1](/assets/mvgalgorithm11.1.png)
The trick is to replace $F = UDV^T$ by $F' = U diag(r,s,0)V^T$.

The **gold standard algorithm to compute fundamental matrix**
![Algorithm 11.3](/assets/mvgalgorithm11.3.png)

## Image Rectification
Identify correspondences $x_i \leftrightarrow x'_i$, compute $F, e, e'$. Compute $H'$ maps $e'$ to $(1, 0, 0)^T$, compute the matching $H$ that minimizes $\sum_i d(Hx_i, H' x'_i)$. The rectified image is $H I $ and $H' I'$. (todo: add the code from class)


# 3D reconstruction

Given $x = P X$ and $x' = P' X$, again using the DLT method, $x \times (P X) = 0$, assuming the last component of $x$ is 1. Doing the same for $x'$, take two equations from the two and combine to become
 $A X = 0$, solving use the SVD method to get the 3d point $X$.
![Equation 12.0](/assets/mvgequation12.0.png)
The actual optimal reconstruction algorithm 12.1 tries to minize the distance of $x$ and $x'$ to the epipolar line. It is much more involved than DLT though.



## Metric reconstruction using direct method
![Algorithm 10.1](/assets/mvgalgorithm10.1.png)

# N-View

## Affine

## Bundle Adjustment