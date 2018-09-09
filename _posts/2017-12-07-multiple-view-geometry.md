---
layout: post
mathjax: true
comments: true
title:  "Multiple View Geometry in Computer Vision"
---
This blog is about Multiple View Geometry in Computer Vision by Richard Hartley and Andrew Zisserman. I will mainly take some notes about the gold standard algorithms in this book. All figures are blatantly copied from the book.

# stuff at infinity
points lines

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