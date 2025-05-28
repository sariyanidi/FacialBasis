#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:55:21 2023

@author: sariyanide
"""
import os
import sys
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('vid_path')
parser.add_argument('lmks_path', type=str)
parser.add_argument('expressions_path', type=str)
parser.add_argument('poses_path', type=str)
parser.add_argument('alphas_path', type=str, nargs='?', default=None)
parser.add_argument('betas_path', type=str, nargs='?', default=None)
parser.add_argument('illums_path', type=str, nargs='?', default=None)
parser.add_argument('--fov', default=30.0, type=float)
parser.add_argument('--GPUid', default=0, type=int)
parser.add_argument('--first_3DID', default=1, type=int)
parser.add_argument('--device', default='cuda:0', type=str)

args = parser.parse_args()

dirs = [os.path.dirname(f) for f in [args.expressions_path, args.alphas_path, args.betas_path, args.poses_path, args.illums_path] if f is not None ]

for d in dirs:
    if not os.path.exists(d):
        print(f'Directory for target file(s) does not exist: {d}')
        print('Terminating without processing')
        sys.exit(1)


print('Loading pytorch and models ...')
import sys
sys.path.append('./models')

import video_fitter
import camera
import cv2

cap = cv2.VideoCapture(args.vid_path)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

camera = camera.Camera(args.fov, args.fov, float(width)/2.0, float(height)/2.0)

vf = video_fitter.VideoFitter(camera, device=args.device)
vf.load_models()
print('Done.')    

params3DIlite = vf.process_w3DID(args.vid_path, args.lmks_path)
vf.save_txt_files(params3DIlite, args.alphas_path, args.betas_path, args.expressions_path, args.poses_path, args.illums_path)
