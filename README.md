
# Installation

## 1) Download morphable models

You need to obtain the Basel Face Model (BFM'09) and the Expression Model through the links below:
**Models**
* Basel Face Model (BFM'09): [click here](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) to obtain the Basel Face Model from the University of Basel
* Expression Model: Download the expression model (the Exp_Pca.bin) file from [this link](https://github.com/Juyong/3DFace)

Once you download both Basel Face Model (`01_MorphableModel.mat`) and the Expression Model (`Exp_Pca.bin`), copy theminto the `data/raw` directory. Specifically, these files should be in the following locations:

```
data/raw/01_MorphableModel.mat
data/raw/Exp_Pca.bin
```
## 2) Install in virtual environment
Create and activate python virtual environment

```
python3.8 -m venv env
source env/bin/activate
```

Install all the necessary repos and packages:
```
chmod +x install.sh
./install.sh
```

# Demo
The code below will run the Facial Basis method on a test video. The three lines correspond to the three necessary components:
* Facial landmark detection
* Computation of 3DMM expression coefficients
* Computation of the local expression coefficients
```
python face_alignment_opencv/process_video.py testdata/elaine.mp4  --save_result_video 0 --visualize_result 0
python process_video.py testdata/elaine.mp4 testdata/elaine.csv testdata/elaine.expressions testdata/elaine.poses
python compute_local_exp_coefficients.py testdata/elaine.expressions testdata/elaine.local_expressions
```

The local expression coefficients in the demo above are stored in `testdata/elaine.local_expressions`. This file is a text file with a matrix of size `T x 50`, where `T` is the number of frames in the video and 50 is the coefficients corresponding to the 50 Facial Basis Units (BUs).

