#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:53:55 2023

@author: sariyanide
"""
import os
import cv2
from utils import utils
import torch
import random
import numpy as np
import pandas as pd
import morphable_model
from torchvision.transforms import Grayscale
from camera import Camera
from medium_model import MediumModel
from separated_model import SeparatedModelV3




class VideoFitter():
    
    def __init__(self, cam, outdir_root=None, device='cuda', Tmax=None):
        self.cam = cam
        self.device = device
        self.outdir_root = outdir_root
        self.rasterize_fov = 15.0
        self.rasterize_size = 224.0
        self.rasterize_cam = Camera(fov_x=self.rasterize_fov, fov_y=self.rasterize_fov,
                                    cx=self.rasterize_size/2.0, cy=self.rasterize_size/2.0)
        self.use_3DID_exp_model = True
        self.Nframes = 7
        self.Ntot_reconstructions = 30
        self.Tmax = Tmax
        self.use_exp_for_neutral = True
        
        self.models_loaded = False
        self.which_bfm = 'BFMmm-23660'
        self.which_pts = 'sampled'
        
        self.mm = morphable_model.MorphableModel(key=self.which_bfm, device=self.device)
        
        
        if outdir_root is not None:
            if not os.path.exists(self.outdir_root):
                os.mkdir(self.outdir_root)
            
            self.outdir_3DID = f'{self.outdir_root}/{self.get_key(is_final=False)}'
            self.outdir_final = f'{self.outdir_root}/{self.get_key(is_final=True)}'
            
            if not os.path.exists(self.outdir_3DID):
                os.mkdir(self.outdir_3DID)
            
            if not os.path.exists(self.outdir_final):
                os.mkdir(self.outdir_final)
            
    
    def get_key(self, is_final):
        
        key = f'{self.use_exp_for_neutral}{self.use_3DID_exp_model}'
        
        if is_final:
            key = f'f{self.cam.fov_x}-{self.which_pts}-{self.Nframes}-{self.Ntot_reconstructions}-{key}'
            
        return key
    
    
    def load_models(self):
        
        if self.models_loaded:
            return
        
        # dataset used to train models
        dbname = 'combined_celeb_ytfaces'
        checkpoint_dir = 'checkpoints'
        
        cfgid = 2
        Ntra = 139979
        lrate = 1e-5
        backbone = 'resnet50'
        tform_data = True

        init_path_id = f'{checkpoint_dir}/medium_model{self.rasterize_fov:.2f}{dbname}{backbone}{Ntra}{tform_data}{lrate}-{cfgid}-BFMmm-23660UNL_STORED.pth'

        
        checkpoint_id = torch.load(init_path_id)
        if not self.use_3DID_exp_model:
            self.model_id =  MediumModel(rasterize_fov=self.rasterize_fov, 
                                                       rasterize_size=self.rasterize_size, 
                                                    label_means=checkpoint_id['label_means'].to(self.device), 
                                                    label_stds=checkpoint_id['label_stds'].to(self.device),
                                                    which_bfm=self.which_bfm, which_backbone=backbone)
    
            self.model_id.load_state_dict(checkpoint_id['model_state'])
            self.model_id.to(self.device)
            self.model_id.eval()
        else:
            init_path_perframe = init_path_id
    
            spath = f'{checkpoint_dir}/sep_modelv3SP{self.rasterize_fov:.2f}{dbname}{backbone}{lrate}{cfgid}{tform_data}{Ntra}_V2.pth'
    
            checkpoint = torch.load(spath)
            self.model = SeparatedModelV3(rasterize_fov=self.rasterize_fov, rasterize_size=self.rasterize_size,
                                                    label_means=checkpoint_id['label_means'].to(self.device), 
                                                    label_stds=checkpoint_id['label_stds'].to(self.device),
                                                    init_path_id=init_path_id,
                                                    init_path_perframe=init_path_perframe,
                                                    which_backbone=backbone)
    
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()
        
        self.models_loaded = True
    
    
    def get_landmarks(self, lmkspath):
        
        if lmkspath.find('.csv') >= 0:
            csv = pd.read_csv(lmkspath)
            L = csv.values[:,1:]
        else:
            L = np.loadtxt(lmkspath)
        
        return L
    
    
    def process_w3DID(self, vpath, lmkspath, params3DIlite_path=None):
        
        self.load_models()
        
        L = self.get_landmarks(lmkspath)
        bn = '.'.join(os.path.basename(vpath).split('.')[:-1])
        
        # all_params_path = f'{self.outdir_3DID}/{bn}_3DID.npy'
        # print(all_params_path)
        
        """
        if os.path.exists(all_params_path):
            return False
            return np.load(all_params_path, allow_pickle=True).item()
        """
        
        cap = cv2.VideoCapture(vpath)
        
        transform_gray = Grayscale(num_output_channels=3)
    
        alphas = []
        exps = []
        angles = []
        betas = []
        taus = []
        invMs = []
        frame_idx = 0
        
        while(True):    
            print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")
            frame_idx += 1
            ret, frame = cap.read()
        
            if not ret:
                break 
            
            lmks51 = L[frame_idx-1,:].reshape(-1,2).astype(np.float32)
            
            if lmks51.shape[0] == 68:
                lmks51 = lmks51[17:,:]
            
            if lmks51[0,0] == 0:
                alphas.append(alphas[-1] if len(alphas) > 0 else np.nan*np.ones((1, self.mm.Kid)))
                betas.append(betas[-1] if len(betas) > 0 else np.nan*np.ones((1, self.mm.Ktex)))
                angles.append(angles[-1] if len(angles) > 0 else np.nan*np.ones((1,3)))
                taus.append(taus[-1] if len(taus) > 0 else np.nan*np.ones((1, 3)))
                exps.append(exps[-1] if len(exps) > 0 else np.nan*np.ones((1, self.mm.Kexp)))
                invMs.append(invMs[-1] if len(invMs) > 0 else np.nan*np.ones((2, 3)))
                continue
            
            cim = frame.astype(np.float32)/255.0
            M = utils.estimate_norm(lmks51, cim.shape[0], 1.5, [25,25])
            cim = utils.resize_n_crop_cv(cim, M, int(self.rasterize_size))
            invM = utils.estimate_inv_norm(lmks51, frame.shape[1], 1.5, [25,25])
            
            """
            lmks51_hom = np.concatenate((lmks51, np.ones((lmks51.shape[0], 1))), axis=1)
            lmks_new = (M @ lmks51_hom.T).T
            icim = utils.resize_n_crop_inv_cv(cim, invM, (frame.shape[1], frame.shape[0]))
            if True: #frame_idx == 9:
                print(M)
                print(invM)
                plt.figure(figsize=(50, 70))
                plt.imshow(frame)
                plt.plot(lmks51_hom[:,0], lmks51_hom[:,1])
                plt.show()
                plt.imshow(cim)
                plt.plot(lmks_new[:,0], lmks_new[:,1])
                plt.show()
                # print(frame)
                print(icim.shape)
                plt.figure(figsize=(50, 70))
                plt.imshow(icim)
                plt.title('hey')
                plt.show()
                # break
                
            """

            cim = np.transpose(cim, (2,0,1))
            cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
            cim = cim.to(self.device)
            
            if not self.use_3DID_exp_model:
                y = self.model_id(cim)
                params = self.model_id.parse_params(y[0])
            else:
                y, alpha_un, beta_un, _ = self.model.forward(cim)
                params = self.model.parse_params(y, alpha_un, beta_un)
    
            """
            if True:
                pts = self.mm.project_to_2d(self.rasterize_cam, params['angles'], params['tau'], params['alpha'], params['exp']).detach().cpu().squeeze().numpy()
                pts[:,1] = self.rasterize_size-pts[:,1]
                print(pts.shape)
                pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
                pts = (invM @ pts.T).T
                plt.figure(figsize=(1.4*50, 1.4*70))
                plt.imshow(frame)
                plt.plot(pts[:,0], pts[:,1], '.')
                plt.savefig('c.jpg')
                break            
            """

            
            """

            (mask, _, rim), pr = self.model.backbone_id.render_image(params)
            mask = mask.detach().cpu().numpy()[0,0,:,:]
            cim0 = cim.detach().cpu().numpy()[0,0,:,:]
            rim = rim.detach().cpu().numpy()[0,0,:,:]
            
            rim[mask==0] = (cim0[mask==0])
            rim[mask==1] = (cim0[mask==1]+2*rim[mask==1])/3.0
            """
                    
            """
            if frame_idx % 2  == 0:
                p = self.mm.compute_face_shape(params['alpha'], params['exp'])
                p = p.detach().cpu().squeeze().numpy()
                
                # plt.clf()
                plt.figure(frame_idx, figsize=(30*1.5,10*1.5))
                
                    
                plt.subplot(141)
                plt.imshow(cim0)
                
                plt.subplot(142)
                plt.imshow(rim)
                
                plt.subplot(143)
                # plt.plot(p0[:,0], p0[:,1], '.')
                plt.plot(p[:,0], p[:,1], '.')
                plt.ylim((-90, 90))
                
                
                plt.subplot(144)
                # plt.plot(p0[:,2], p0[:,1], '.')
                plt.plot(p[:,2], p[:,1], '.')
                plt.ylim((-90, 90))
                plt.show()
                """
            
            alphas.append(params['alpha'].detach().cpu().numpy().astype(float))
            betas.append(params['beta'].detach().cpu().numpy().astype(float))
            angles.append(params['angles'].detach().cpu().numpy().astype(float))
            taus.append(params['tau'].detach().cpu().numpy().astype(float))
            exps.append(params['exp'].detach().cpu().numpy().astype(float))
            invMs.append(invM)
        
        params3DIlite = {'alphas': alphas,
                      'betas': betas,
                      'angles': angles,
                      'taus': taus,
                      'exps': exps,
                      'invMs': invMs}
        
        if params3DIlite_path is not None:
            np.save(params3DIlite_path, params3DIlite)
        
        return params3DIlite
    
    def save_txt_files(self, params3DIlite, alphas_path, betas_path, expressions_path, poses_path, illums_path):
        
        alpha = np.mean(params3DIlite['alphas'], axis=0).T
        beta = np.mean(params3DIlite['betas'], axis=0).T
        
        if alphas_path is not None:
            np.savetxt(alphas_path, alpha)
        if betas_path is not None:
            np.savetxt(betas_path, beta)
        
        taus = np.concatenate(params3DIlite['taus'], axis=0)
        angles = np.concatenate(params3DIlite['angles'], axis=0)
        
        poses = np.concatenate((taus, angles), axis=1)
        
        # Default illums parameter (used only for video visualization)
        if illums_path is not None:
            illums = np.tile([48.06574, 9.913327, 798.2065, 0.005], (poses.shape[0], 1))
            np.savetxt(illums_path, illums)
        
        np.savetxt(expressions_path, np.concatenate(params3DIlite['exps'], axis=0))
        np.savetxt(poses_path, poses)
        
        
   
