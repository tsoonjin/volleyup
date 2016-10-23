## Jin's documentations

### Multiple Target Tracking (MTT)

#### Multiple Hypotheses Tracking Revisited

1. Introduction
 - Builds a tree of potential track hypotheses for each candidate target, thereby providing a
systematic solution to the data association problem. BFS search, therefore pruning is essential

2. Related Work
 - network flow restricted to unary and pairwise potential.
 - markov chain monte carlo (mcmc) 

#### Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor

1. Introduction
 - important to have a robust data association model and accurate similarity measure between
 detections. 
 - global trajectory optimization limited to post-video analysis
 - online method prone to error because uses temporal window only

2. Details
 - Aggregated Local Flow Descriptor (ALFD) to measure similarity between detections across time

### MCMC Particle Filter

1. Details
 - add a MRF prior to motion model 
 - uses MCMC sampling rather than traditional importance sampling 

### General tracking algorithm

#### Mutli-camera Probabilistic Occupancy Map

1. Details
 - discretize ground plane to cells 
 - known homography matrix to map to top view 
 - eigenbackground subtraction to generate detections


#### Multiple Object Tracking using K-Shortest Paths Optimization

1. Introduction
 - kalman filter and mean shift works for small number of objects
 - particle filter using mcmc follows in the same class. Grows exponential with window considered
 - boosted particle filter

2. Details
 - consider bigger window when linking to achieve more robust assignment
 - solve linear programming in fast manner


#### Tracking Multiple Players using a Single Camera

1. Introduction
 - tracking-by-detection is the state of the art. Performs independent detections and link them
 - single camera can performs as well with geometric constraint
 - current approach that supersedes particle filter: connect detections into tracklets (short tracks)

2. Details
 - Retraining the Deformable Part Model (DPM)   
 - geometric constraint such as court marking
 - uses ellipse for non-maximum suppression


#### Multi-Commodity Network Flow for Tracking Multiple People

1. Introduction
 - formalize as multiple commodity min-cost max-flow problem

2. Related work
 - kalman filter and gating but prone to identity switches
 - particle filter only works for small batches due to growth in state space
 - operations on tracklets that minimizes a global energy function
 - conditional random field used to model occlusion and motion dependencies between tracklet
 - linear programming and dynamic programming powerful alternative but hard to set the edge cost

3. Background of problems
 - tracking players different than pedastrian because of erratic movement and less predictable
 - run KSP followed by linear programming


### Background subtraction

#### Segmentation by Eigenbackground subtraction

1. Details
 - sample N images to build adaptive eigenspace that models the background
 - compute mean image and covariance matrix and perform eigen-value decomposition


### Volleyball specific

#### Analyzing Volleyball Match Data from the 2014 World Championships Using Machine Learning Techniques

1. Introduction
 - identify team's attacking patterns in volleyball matches that occur frequently in won rallies and 
 infrequently in lost rallies
 - identify attacking pattern used by one team but not the other

2. Background
 - Basic skills: serve, dig, pass, set, spike, block
 - encode courts into grids into 3 x 3 squares
 - relational learning using inductive logic programming (ILP)
