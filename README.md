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

Demo
```
python face_alignment_opencv/process_video.py testdata/elaine.mp4  --save_result_video 0
python process_video.py testdata/elaine.mp4 testdata/elaine.csv testdata/elaine.expressions testdata/elaine.poses
python compute_local_exp_coefficients.py testdata/elaine.expressions testdata/elaine.local_expressions
```


