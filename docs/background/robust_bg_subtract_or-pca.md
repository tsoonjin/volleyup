## Robust Background Subtraction via Online Robust PCA

### Understanding how Online RPCA works
1. [Significance of nuclear norm](https://www.quora.com/What-is-the-significance-of-the-nuclear-norm)
 - provides a convex heuristic for low rank minimization problem. used for matrix completion
 - convex envelope (best lower bound for original function)
 - sum of singular values given by SVD of a matrix
2. Norm notation: superscript (squared version)
3. Develop a stochastic optimization algorithm to minimize the empirical cost function one sample
at time

### Methodology
1. Image Decomposition
 - G1 = cv2.GaussianBlur(raw, (5, 5), 2)
 - G2 = cv2.GaussianBlur(G1, (5, 5), 2)
 - L1 = G1 - G2
 - use G2 and L1 for background modelling

2. Background modelling
 - initialize basis L using N video frames with appropriate rank
 - two iterative steps:
  - project input onto basis
  - add input to basis

3. Integration
 - Add matrices generated from Gaussian and Laplacian

4. Choosing parameters
 - for static background, r = 1, lamda2 = (0.01, 0.04]
 - for dynamic background, r = (1, 2, 3], lamda2 = (0.01, 0.03] for Laplacian
 - for dynamic background, r = (r, 8], lamda2 = (lamda2, 0.09] for Gaussian

5. Post processing
 - median filter on foreground and background
