# Beach Volleyball Match Analysis
A python application that analyzes beach volleyball matches to extract important information
for tactical planning

## Requirements
 - [ ] stitch to form panorama of volleyball court with player actions and ball
 - [ ] top-down view of the players' position and ball position on *full* volleyball court
 - [ ] plot position and movement of *player* from top view 
 - [ ] plot position and movement of *ball* from top view
 - [ ] concatenate all 7 videos with the following layout:

 | Original Video                | Full volleyball court                          |
 |-------------------------------|------------------------------------------------|
 | Top-down view - player - ball | Statistics:  - distance run  - number of jumps |

## Getting Started
1. [Installing OpenCV3 on Ubuntu 16.04](http://thaim.hatenablog.jp/entry/2016/07/11/004631)

## Prior knowledge
1. PCA, SVD
 - [PCA, SVD by Alexander Ihler](https://www.youtube.com/watch?v=F-nfsSq42ow)
 - [Eigenface by applying PCA bytefish.de](http://www.bytefish.de/pdf/facerec_python.pdf)
2. Gaussian Mixture Model
 - [Clustering(4): GMM & EM by Alexander Ihler](https://www.youtube.com/watch?v=qMTuMa86NzU)
 - [OpenCV Background Subtraction](http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html)
 - [Jake Vanderplas GMM for astronomy](https://www.youtube.com/watch?v=W0XECm4-3LI)
3. Mean-shift algorithm
 - [UCF Mubarak Shah Mean-shift](https://www.youtube.com/watch?v=M8B3RZVqgOo)
 - [Udacity Mean-shift](https://www.youtube.com/watch?v=DEtx_R1NzPY)
 - [OpenCV Mean-shift tutorial](http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html)
4. Fast Fourier Transform
 - [William Cox: An Intuitive Introduction to the Fourier Transform and FFT](https://www.youtube.com/watch?v=YEwIjyOKFQ4)
 - [FFT for Computer Vision](https://www.cs.unm.edu/~brayer/vision/fourier.html)
 - [2D dimensional dst](https://www.youtube.com/watch?v=YYGltoYEmKo)
 - [Oxford Image Processing and Fourier](http://www.robots.ox.ac.uk/~az/lectures/ia/lect2.pdf)
 - [What is the meaning of Fourier Transform of an image](https://www.quora.com/What-is-the-meaning-of-Fourier-transform-of-an-image-Why-is-it-important-in-image-processing)
5. Correlation vs Convolution
 - [Correlation and Convolution Notes](http://www.cs.umd.edu/~djacobs/CMSC426/Convolution.pdf)
6. Optical Flow
 - [OpenCV Optical Flow tutorial](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
 - [CVFX Lecture 13: Optical Flow](https://www.youtube.com/watch?v=KoMTYnlNNnc)
7. Image morphing
 - [PyImageSearch Panorama Stiching tutorial](http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
 - [PyImageSearch getPerspectiveTransform](http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
 - [Generating bird eye view opencv SO](http://stackoverflow.com/questions/15768651/generating-a-birds-eye-top-view-with-opencv)


## Courses
 - [Udacity: GeorgiaTech Intro to CV](https://classroom.udacity.com/courses/ud810/lessons/3490398568/concepts/47481911650923)

## Credits
1. [philipz](https://github.com/philipz/docker-opencv3) for the Dockerfile with OpenCV 3
