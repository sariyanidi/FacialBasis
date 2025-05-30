# list of files to verify
# These morphable models need to be obtained from the right sources and placed to the correct locations
files=(
  "data/raw/01_MorphableModel.mat"
  "data/raw/Exp_Pca.bin"
)

for f in "${files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Required file not found: $f" >&2
    exit 1
  fi
done


pip install --upgrade pip

pip install -r requirements.txt

git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python setup.py install
cd ..

git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/

python prepare_BFM.py

#!/usr/bin/env bash
set -e

# ensure target directories exist
mkdir -p ./checkpoints ./models/checkpoints

# download if missing
if [ ! -f ./checkpoints/backbone.pth ]; then
  wget https://sariyanidi.com/dbox/3DIlite/backbone.pth -P ./checkpoints/
fi

if [ ! -f ./checkpoints/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL_STORED.pth ]; then
  wget https://sariyanidi.com/dbox/3DIlite/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL_STORED.pth -P ./checkpoints/
fi

if [ ! -f ./checkpoints/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-052True139979_V2.pth ]; then
  wget https://sariyanidi.com/dbox/3DIlite/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-052True139979_V2.pth -P ./checkpoints/
fi

if [ ! -f ./models/checkpoints/resnet50-0676ba61.pth ]; then
  wget https://sariyanidi.com/dbox/3DIlite/resnet50-0676ba61.pth -P ./models/checkpoints/
fi


git clone https://github.com/sariyanidi/face_alignment_opencv.git
cd face_alignment_opencv
pip install --upgrade pip
pip install -r requirements_cpu.txt
cd ..

