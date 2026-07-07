# Deep flux optique

## Dataset description

The datasets used in this project are from the Cell Tracking Challenge web site (https://celltrackingchallenge.net/) and are all 2D datasets.
Each dataset folder contains 2 videos (01 and 02), one is used for training and the second for testing. 
Here is a description of the different type of annotation used in this project (with N = 1 or 2) :
 - 0N_ERR_SEG : imperfect segmentation masks for sequence of the real dataset
 - 0N_GT : gold truth for the sequence
    - SEG : gold segmentation truth, not complete
    - TRA : gold tracking truth

The different dataset used are : 
 - Fluo-N2DH-GOWT1_train : GFP-GOWT1 mouse stem cells (Dr. E. Bártová. Institute of Biophysics, Academy of Sciences of the Czech Republic, Brno, Czech Republic)
 - BF-C2DL-MuSC : 