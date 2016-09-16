# Strategy
Tactics derived from various journal papers working on this topic

## 1. Visual Tracking of Athletes in Beach Volleyball Using a Single Camera

### **Abstract** ###
A novel method based on integral histograms, to use a high dimensional model for a particle filter
without drastic increase in runtime. We extend integral histograms to handle rotated objects

### **Details** ###
1. Homography
 - Homography, to compute court positions of players (for details see Hartley & Zisserman, 2002)
 - reconstruct real world court coordinates (e.g. 8x16m for beach volleyball)
2. Color tracking using particle filter
 - Using H, S, V histogram using integral image
 - original tracker divided into 3 parts estimate rotation
3. Including background information
 - integrate backgrund similarity into likelihood measurement of the particles

## 2. Video Object Extraction by Using Background Subtraction Techniques for Sports Applications

### **Abstract** ###
Mixture of Gaussian turns out to be best in reliability of extraction of moving objects, robust
to noise, whereas the conventional algorithms result in noise and poor extraction of objects

### **Details** ###
1. Frame difference
 - cons: similarity between court and platers. must be constantly moving
 - pros: remove noise. highly adaptive
2. Approximate median
 - if |current - bg| > T: bg+1, fg = 1 else bg-1, fg = 0
 - good for handling slow movement
3. Mixture of Gaussian
 - keeps cumulative average of the recent pixel values
4. Mixture of Gaussians produces the best results, while approximate median filtering offers a
simple alternative with competitive performance

## 3. ViBe: A universal background subtraction algorithm for video sequences

### **Abstract** ###
Adapts the model by choosing randomly which values to substitute from the background model.
This approach differs from those based on the classical belief that the oldest values should be
replaced first.

### **Details** ###
1. Use radius R = 20 and cardinality of 2 
2. Randomly select frame to replace

## 4. Tracking of Ball and Players in Beach Volleyball Videos

### **Abstract** ###
Results suggest an improved robustness against player confusion between different particle sets
when tracking with a rigid grid approach

### **Details** ###
1. Homographic transformation from image coordinate to 4 corners of a court
2. Foreground extractionb by differencing with background model and morphological opening
3. Mask weighting using 20-50 particles
4. Movement weighting
5. Color weighting using Bhattacharyya's distance
6. Ball tracking
 - Uses two concenric squares to track a ball by constraining number of pixels in outer square
 - Generates trajectories based on current position of ball

## 5. What Players do with the Ball: A Physically Constrained Interaction Modeling (likely won't be implemented)

## 6. Motion Detection Using an Improved Colour Model

### **Abstract** ###
Our method relies on the abil- ity to represent colours in terms of a 3D-polar coordinate system
having saturation independent of the brightness function; specifically, we build upon an 
Improved Hue, Luminance, and Saturation space (IHLS). 

### **Details** ###
1. Outperforms the HSV model as well as the
photometric colour invariants NRGB and c 1 c 2 c 3 in several challenging sequences
2. Handles hue for low saturation case

## 7. Improved optimal seam selection blending for fast video stitching of videos captured from freely moving devices

### **Details** ###
1. Interest points (IP) using SURF and optical flow
2. Filtering IPs belonging to foreground moving objects
3. Enhanced blending, region-of-difference based
 - distance between pixels and intensity difference

## 8. Fast stitching of videos captured from freely moving devices by exploiting temporal redundancy

### **Abstract** ###
Construct in real-time a panoramic video stream from input video streams captured by freely moving cameras

### **Details** ###
1. Area of overlap from previous frames
2. Use motion vector from first frame to avoid recomputing transformation matrix. Use less descriptors
3. Using LKT optical flow to calculate global 2D motion. Use only background IPs

## 9. Ball tracking and 3D trajectory approximation with applications to tactics analysis from single-camera volleyball sequences






