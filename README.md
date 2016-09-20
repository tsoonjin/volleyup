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
 | Top-down view 		 | Statistics					  |
 | - player			 | - distance run				  | 
 | - ball 		         | - number of jumps 			 	  |

## Getting Started
1. [Installing OpenCV3 on Ubuntu 16.04](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)
2. Download and extract [data.zip](https://drive.google.com/open?id=0B8rRUzf-5h4fdHpVdzlLTktCZEk) folder into **volleyup/volleyup** folder to start playing with different algorithms available

## Strategy

### 1. Image mosaicing for video
 - [AKAZE Matching using Brute Force matcher](http://docs.opencv.org/3.1.0/dc/d16/tutorial_akaze_tracking.html)
 - [Mosaics: U.Washington](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect7.pdf)
 - [Recognizing panorama, Lowe, 2003](http://matthewalunbrown.com/papers/iccv2003.pdf)
 - [Image mosaicing with Motion Segmentation Stanford](http://web.stanford.edu/class/ee392j/Winter2002/projects/roman_gilat_report.pdf)
 - [Real-time panorama and image stitching PyImageSearch](http://www.pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/)
 - [OpenCV panorma stitching for 2 images](http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
 - [Using RANSAC with OpenCV SIFT](https://gitlab.com/josemariasoladuran/object-recognition-opencv-python/blob/master/objrecogn.py)
 - [Naive image stitching](http://home.deib.polimi.it/boracchi/teaching/IAS/Stitching/stitch.html)
 - [Automated Panorama Stitching](https://cs.brown.edu/courses/csci1950-g/results/proj6/edwallac/)
 - [Auto Stitching Photo Mosaics CMU](http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj4/www/wwedler/)
 - [Some problems on image stitching SO](http://stackoverflow.com/questions/11134667/some-problems-on-image-stitching-homography)

### 2. Getting bird-eye view of the video
 - [Recovering a comprehensive road appearance mosaic from video](http://www.zemris.fer.hr/~ssegvic/pubs/sikiric10mipro.pdf)
 - [Real Time Distance Determination for an  Automobile Environment using Inverse Perspective Mapping in Open CV ](http://shanetuohy.com/fyp/Images/Shane%20Tuohy%20Thesis.pdf)
 - use perspective transform to warp the perspective by detecting corners of court

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
8. Edge Detection
 - [Canny Edge Computerphile](https://www.youtube.com/watch?v=sRFM5IEqR2w)
9. Features
 - [RANSAC](https://www.youtube.com/watch?v=NKxXGsZdDp8)


## Courses
 - [Udacity: GeorgiaTech Intro to CV](https://classroom.udacity.com/courses/ud810/lessons/3490398568/concepts/47481911650923)

## Credits
1. [philipz](https://github.com/philipz/docker-opencv3) for the Dockerfile with OpenCV 3
