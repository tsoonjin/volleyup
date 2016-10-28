## Challenges

1. Moving camera and zoom in/out
2. Color of players similar to background
3. Players movement causes appearance model of players to vary
4. Occlusion between players and balls
5. Movement of players erratic. Interact with partners and opponents
6. When player jump leaves ground plane

## Requirements

1. 4 players only on the court
2. Does not need to work online

## Object Detection

1. Background Subtraction
2. Optical Flow
3. Cascade of features
4. Color rectangular feature
5. Color histogram
6. Edge Orientation Histogram (EOH)
7. Feature Descriptor: Shi-Tomasi-Kanade
8. Deformable Part Model (DPM)
9. Integral Channel Feature (ChnDetector)
10. Discriminately Trained Deformable Part Model (DT-DPM)
11. Aggregate Channel Features (ACF)
12. Scale Dependent Pool (SDP)
13. Faster-RCNN
14. 

## Tracking

1. Particle Filter
 - one particle set for each player
 - k-nearest neighbor which link to nearest tracked
2. Camshift
3. Network flow
4. Continuous Energy Minimization (CEM)
5. Similar Multiple Object Tracking (SMOT)
6. JPDA (Joint Probability Data Association)

## Background subtraction

1. Online RPCA (winner)
2. Approximate median differencing
3. GMM
4. Local SVD Binary Pttern
5. PCA
