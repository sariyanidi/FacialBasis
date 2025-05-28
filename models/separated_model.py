#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:23 2023

@author: v
"""

import os
import torch
import torch.nn as nn
from models import camera 
from utils import utils
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur
from kornia.morphology import erosion

from medium_model import MediumModel

import sys
sys.path.append('../insightface/recognition/')
from arcface_torch.backbones import get_model

# from resnets import resnet18, conv1x1

from models import resnets, morphable_model, mesh_renderer




def gradient_magnitude(gen_frames):
    alpha=2
    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    # gt_dx, gt_dy = gradient(gt_frames)
    #
    # grad_diff_x = torch.abs(gt_dx - gen_dx)
    # grad_diff_y = torch.abs(gt_dy - gen_dy)
    

    # condense into one tensor and avg
    return gen_dx ** alpha + gen_dy ** alpha
    






class SeparatedModelV0(nn.Module):
    
    resnet18_last_dim = 512
    resnet50_last_dim = 2048
    
    def __init__(self, 
                 rasterize_fov,
                 rasterize_size,
                  init_path_id,
                  init_path_perframe,
                  device='cuda',
                 label_stds=None,
                 label_means=None,
                 which_bfm='BFMmm-23660'
                 ):
        
        super().__init__()
        self.mm = morphable_model.MorphableModel()
        self.device = device
        
        self.label_stds = label_stds
        self.label_means = label_means
        
        if self.label_stds is not None and self.label_means is not None:
            self.alpha_stds = self.label_stds[:199].reshape(1,-1)
            self.exp_stds = self.label_stds[199:(199+79)].reshape(1,-1)
            self.beta_stds = self.label_stds[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_stds = self.label_stds[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_stds = self.label_stds[-6:-3].reshape(1,-1)
            self.tau_stds = self.label_stds[-3:].reshape(1,-1)

            self.alpha_means = self.label_means[:199].reshape(1,-1)
            self.exp_means = self.label_means[199:(199+79)].reshape(1,-1)
            self.beta_means = self.label_means[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_means = self.label_means[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_means = self.label_means[-6:-3].reshape(1,-1)
            self.tau_means = self.label_means[-3:].reshape(1,-1)

            
        self.use_last_fc = False
        self.rasterize_fov = rasterize_fov
        self.rasterize_size = rasterize_size
        
        self.cam = camera.Camera(fov_x=rasterize_fov, fov_y=rasterize_fov, 
                                 cx=self.rasterize_size/2.0, cy=self.rasterize_size/2.0)
        
        # Renderer requires GPU
        self.renderer = mesh_renderer.MeshRenderer(rasterize_fov, rasterize_size=rasterize_size, use_opengl=False).to(self.device)


        state_dict_id = torch.load(init_path_id, map_location='cpu')['model_state']
        state_dict_perframe = torch.load(init_path_perframe, map_location='cpu')['model_state']
        
        
        which_backbone = 'resnet50'
        if init_path_id.find('resnet18') > -1:
            which_backbone = 'resnet18'
        self.backbone_id = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone, which_bfm=which_bfm)
        self.backbone_id.load_state_dict(state_dict_id)
        
        self.backbone_perframe = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone, which_bfm=which_bfm)
        self.backbone_perframe.load_state_dict(state_dict_perframe)
        
        for param in self.backbone_id.rigid_layers.parameters():
            param.requires_grad = False

        # Disable illumination layers
        for param in self.backbone_id.illum_layers.parameters():
            param.requires_grad = False

        # Disable expression layer from identity network
        for param in self.backbone_id.MM_layers.parameters():
            param.requires_grad = False
            
        # Disable expression layer from identity network
        for param in self.backbone_id.backbone.parameters():
            param.requires_grad = False
        
        self.backbone_id.eval()
            
            

        # Disable last layers of per-frame backbone
        for param in self.backbone_perframe.illum_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.MM_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.rigid_layers.parameters():
            param.requires_grad = False
            
            
        self.Kids = self.mm.Kid+self.mm.Ktex
        N1 = 1024
        N2 = 768
        N3 = 512
        
        self.subnetwork = nn.Sequential(
            nn.Linear(self.backbone_perframe.resnet_lastdim, N1),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(N2, N3),
            nn.ReLU()
            )
        
        self.output = nn.Linear(N3, 79+6)
        
        
        

    def forward(self, input_im, tforms=None, render=False):
        self.backbone_id.eval()
        x_id = self.backbone_id(input_im)[0]
        x_perframe = self.backbone_perframe.backbone(input_im)
        x_perframe = torch.flatten(x_perframe,1)
        
        alpha = x_id[:,:199]
        beta = x_id[:,199+79:199*2+79]
        
        intermediate = []
        # intermediate.append(alpha) # 3DMM id parameters (shape)
        # intermediate.append(beta) # 3DMM id parameters (texture)
        intermediate.append(x_perframe) # backbone for exp & rigid
        x = torch.cat(intermediate, dim=1)
        
        x = self.subnetwork(x)
        x = self.output(x)
        
        return x, alpha, beta, None
    
        

    def test(self, input_im, alpha, beta):
        x_perframe = self.backbone_perframe.backbone(input_im)
        x_perframe = torch.flatten(x_perframe,1)
        
        intermediate = []
        intermediate.append(alpha) # 3DMM id parameters (shape)
        intermediate.append(beta) # 3DMM id parameters (texture)
        intermediate.append(x_perframe) # backbone for exp & rigid
        x = torch.cat(intermediate, dim=1)
        
        x = self.subnetwork(x)
        x = self.output(x)
        
        return x
    
            
    
    
    def parse_params(self, y, alpha, beta):
        params = self.parse_params_unnormalized(y, alpha, beta)
        if self.label_stds is not None and self.label_means is not None:
            params['alpha'] = (params['alpha']*self.alpha_stds)+self.alpha_means
            params['exp'] = (params['exp']*self.exp_stds)+self.exp_means
            params['beta'] = (params['beta']*self.beta_stds)+self.beta_means
            params['angles'] = (params['angles']*self.angle_stds)+self.angle_means
            params['tau'] = (params['tau']*self.tau_stds)+self.tau_means

        return params
        
    
    
    
    def parse_params_unnormalized(self, y, alpha, beta):
        params =  {'alpha': alpha,
                  'exp': y[:, :79],
                  'beta': beta,
                  'angles': y[:,79:79+3],
                  'tau': y[:,79+3:79+6]}
        return params
        
    
    def render_image(self, params):
        canonical_mesh = self.mm.compute_face_shape(params['alpha'], params['exp'])
        # canonical_mesh[:,:, 1] *= -1
        # R = self.mm.compute_rotation_matrix(params['angles'])
        R = self.mm.compute_rotation_matrix_from_eulerrod(params['angles'])
        
        # view-transformed mesh
        mesh = self.mm.view_transform(canonical_mesh, R, params['tau'])
        texture = self.mm.compute_texture(params['beta'])
        # mesh[:,:, 1] *= -1
        
        # print(mesh.shape)
        p = self.cam.map_to_2d(mesh)
        # print(p.shape)
        # print(mesh.shape)

        return self.renderer(mesh, self.mm.tri, feat=texture), p
        
        # d = y[1].squeeze().detach().cpu().numpy()
        # df = d.flatten()
        # w = np.where(df>0)[0]
        # df = df[w]
        
        # d = (d-df.min())/(df.max()-df.min())
        # d[np.where(d<0)] = 0
        # d[np.where(d>1)] = 1
        
        # im = y[-1].squeeze()[0,:,:].cpu().numpy()
        # plt.imshow(im)
        # plt.plot(lmks[:,0], 224-lmks[:,1], 'x')
        
        
    
    
    
    
    
    
    
    


class SeparatedModel(nn.Module):
    
    resnet18_last_dim = 512
    resnet50_last_dim = 2048
    
    def __init__(self, 
                 rasterize_fov,
                 rasterize_size,
                  init_path_id,
                  init_path_perframe,
                  device='cuda',
                 label_stds=None,
                 label_means=None,
                 which_bfm='BFMmm-23660'
                 ):
        
        super().__init__()
        self.mm = morphable_model.MorphableModel()
        self.device = device
        
        self.label_stds = label_stds
        self.label_means = label_means
        
        if self.label_stds is not None and self.label_means is not None:
            self.alpha_stds = self.label_stds[:199].reshape(1,-1)
            self.exp_stds = self.label_stds[199:(199+79)].reshape(1,-1)
            self.beta_stds = self.label_stds[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_stds = self.label_stds[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_stds = self.label_stds[-6:-3].reshape(1,-1)
            self.tau_stds = self.label_stds[-3:].reshape(1,-1)

            self.alpha_means = self.label_means[:199].reshape(1,-1)
            self.exp_means = self.label_means[199:(199+79)].reshape(1,-1)
            self.beta_means = self.label_means[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_means = self.label_means[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_means = self.label_means[-6:-3].reshape(1,-1)
            self.tau_means = self.label_means[-3:].reshape(1,-1)

            
        self.use_last_fc = False
        self.rasterize_fov = rasterize_fov
        self.rasterize_size = rasterize_size
        
        self.cam = camera.Camera(fov_x=rasterize_fov, fov_y=rasterize_fov, 
                                 cx=self.rasterize_size/2.0, cy=self.rasterize_size/2.0)
        
        # Renderer requires GPU
        self.renderer = mesh_renderer.MeshRenderer(rasterize_fov, rasterize_size=rasterize_size, use_opengl=False).to(self.device)


        state_dict_id = torch.load(init_path_id, map_location='cpu')['model_state']
        state_dict_perframe = torch.load(init_path_perframe, map_location='cpu')['model_state']
        
        
        which_backbone = 'resnet50'
        if init_path_id.find('resnet18') > -1:
            which_backbone = 'resnet18'
        self.backbone_id = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone, which_bfm=which_bfm)
        self.backbone_id.load_state_dict(state_dict_id)
        
        self.backbone_perframe = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone, which_bfm=which_bfm)
        self.backbone_perframe.load_state_dict(state_dict_perframe)
        
        for param in self.backbone_id.rigid_layers.parameters():
            param.requires_grad = False

        # Disable illumination layers
        for param in self.backbone_id.illum_layers.parameters():
            param.requires_grad = False

        # Disable expression layer from identity network
        for param in self.backbone_id.MM_layers.parameters():
            param.requires_grad = False
            
        # Disable expression layer from identity network
        for param in self.backbone_id.backbone.parameters():
            param.requires_grad = False
        
        self.backbone_id.eval()
        
        # Disable last layers of per-frame backbone
        for param in self.backbone_perframe.illum_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.MM_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.rigid_layers.parameters():
            param.requires_grad = False
            
            
        self.Kids = self.mm.Kid+self.mm.Ktex
        N1 = 1024
        N2 = 768
        N3 = 512
        
        self.subnetwork = nn.Sequential(
            nn.Linear(self.backbone_perframe.resnet_lastdim+self.Kids, N1),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(N2, N3),
            nn.ReLU()
            )
        
        self.output = nn.Linear(N3, 79+6)
        
        
        

    def forward(self, input_im, tforms=None, render=False):
        self.backbone_id.eval()
        x_id = self.backbone_id(input_im)[0]
        x_perframe = self.backbone_perframe.backbone(input_im)
        x_perframe = torch.flatten(x_perframe,1)
        
        alpha = x_id[:,:199]
        beta = x_id[:,199+79:199*2+79]
        
        intermediate = []
        intermediate.append(alpha) # 3DMM id parameters (shape)
        intermediate.append(beta) # 3DMM id parameters (texture)
        intermediate.append(x_perframe) # backbone for exp & rigid
        x = torch.cat(intermediate, dim=1)
        
        x = self.subnetwork(x)
        x = self.output(x)
        
        return x, alpha, beta, None
        

    def test(self, input_im, alpha, beta):
        x_perframe = self.backbone_perframe.backbone(input_im)
        x_perframe = torch.flatten(x_perframe,1)
        
        intermediate = []
        intermediate.append(alpha) # 3DMM id parameters (shape)
        intermediate.append(beta) # 3DMM id parameters (texture)
        intermediate.append(x_perframe) # backbone for exp & rigid
        x = torch.cat(intermediate, dim=1)
        
        x = self.subnetwork(x)
        x = self.output(x)
        
        return x
    
    
    def parse_params(self, y, alpha, beta):
        params = self.parse_params_unnormalized(y, alpha, beta)
        if self.label_stds is not None and self.label_means is not None:
            params['alpha'] = (params['alpha']*self.alpha_stds)+self.alpha_means
            params['exp'] = (params['exp']*self.exp_stds)+self.exp_means
            params['beta'] = (params['beta']*self.beta_stds)+self.beta_means
            params['angles'] = (params['angles']*self.angle_stds)+self.angle_means
            params['tau'] = (params['tau']*self.tau_stds)+self.tau_means

        return params
        
    
    
    
    def parse_params_unnormalized(self, y, alpha, beta):
        params =  {'alpha': alpha,
                  'exp': y[:, :79],
                  'beta': beta,
                  'angles': y[:,79:79+3],
                  'tau': y[:,79+3:79+6]}
        return params
        
    
    def render_image(self, params):
        canonical_mesh = self.mm.compute_face_shape(params['alpha'], params['exp'])
        # canonical_mesh[:,:, 1] *= -1
        # R = self.mm.compute_rotation_matrix(params['angles'])
        R = self.mm.compute_rotation_matrix_from_eulerrod(params['angles'])
        
        # view-transformed mesh
        mesh = self.mm.view_transform(canonical_mesh, R, params['tau'])
        texture = self.mm.compute_texture(params['beta'])
        # mesh[:,:, 1] *= -1
        
        # print(mesh.shape)
        p = self.cam.map_to_2d(mesh)
        # print(p.shape)
        # print(mesh.shape)

        return self.renderer(mesh, self.mm.tri, feat=texture), p
        
        # d = y[1].squeeze().detach().cpu().numpy()
        # df = d.flatten()
        # w = np.where(df>0)[0]
        # df = df[w]
        
        # d = (d-df.min())/(df.max()-df.min())
        # d[np.where(d<0)] = 0
        # d[np.where(d>1)] = 1
        
        # im = y[-1].squeeze()[0,:,:].cpu().numpy()
        # plt.imshow(im)
        # plt.plot(lmks[:,0], 224-lmks[:,1], 'x')
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    

class SeparatedModelV3(nn.Module):
    
    resnet18_last_dim = 512
    resnet50_last_dim = 2048
    
    def __init__(self, 
                 rasterize_fov,
                 rasterize_size,
                  init_path_id,
                  init_path_perframe,
                  device='cuda',
                 label_stds=None,
                 label_means=None,
                 which_bfm='BFMmm-23660',
                 which_backbone='resnet18'
                 ):
        
        super().__init__()
        self.mm = morphable_model.MorphableModel(key=which_bfm)
        self.device = device
        
        self.label_stds = label_stds
        self.label_means = label_means
        
        if self.label_stds is not None and self.label_means is not None:
            self.alpha_stds = self.label_stds[:199].reshape(1,-1)
            self.exp_stds = self.label_stds[199:(199+79)].reshape(1,-1)
            self.beta_stds = self.label_stds[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_stds = self.label_stds[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_stds = self.label_stds[-6:-3].reshape(1,-1)
            self.tau_stds = self.label_stds[-3:].reshape(1,-1)

            self.alpha_means = self.label_means[:199].reshape(1,-1)
            self.exp_means = self.label_means[199:(199+79)].reshape(1,-1)
            self.beta_means = self.label_means[(199+79):(199+79+199)].reshape(1,-1)
            self.gamma_means = self.label_means[(199*2+79):(199*2+79+27)].reshape(1,-1)
            self.angle_means = self.label_means[-6:-3].reshape(1,-1)
            self.tau_means = self.label_means[-3:].reshape(1,-1)
            
        self.gaussian_smoothing = GaussianBlur(7, sigma=(2, 2.0))

        self.erosion_kernel = torch.ones(10, 10).to(self.device)

        self.use_last_fc = False
        self.rasterize_fov = rasterize_fov
        self.rasterize_size = rasterize_size
        
        self.cam = camera.Camera(fov_x=rasterize_fov, fov_y=rasterize_fov, 
                                 cx=self.rasterize_size/2.0, cy=self.rasterize_size/2.0)
        
        # Renderer requires GPU
        self.renderer = mesh_renderer.MeshRenderer(rasterize_fov, rasterize_size=rasterize_size, use_opengl=False).to(self.device)
        
        state_dict_id = torch.load(init_path_id, map_location='cpu')['model_state']
        state_dict_perframe = torch.load(init_path_perframe, map_location='cpu')['model_state']
        
        
        # which_backbone = 'resnet50'
        # if init_path_id.find('resnet18') > -1:
            # which_backbone = 'resnet18'
        print(which_backbone)
        self.backbone_id = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone,
                                       which_bfm=which_bfm)
        self.backbone_id.load_state_dict(state_dict_id)
        
        self.backbone_perframe = MediumModel(self.rasterize_fov, self.rasterize_size,
                                       label_stds=self.label_stds, label_means=self.label_means,
                                       which_backbone=which_backbone,
                                       which_bfm=which_bfm)
        
        self.backbone_perframe.load_state_dict(state_dict_perframe)
        
        
        for param in self.backbone_id.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone_id.rigid_layers.parameters():
            param.requires_grad = False

        # Disable illumination layers
        for param in self.backbone_id.illum_layers.parameters():
            param.requires_grad = False

        # Disable expression layer from identity network
        for param in self.backbone_id.MM_layers.parameters():
            param.requires_grad = False
        
        # Disable last layers of per-frame backbone
        for param in self.backbone_perframe.illum_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.MM_layers.parameters():
            param.requires_grad = False
        for param in self.backbone_perframe.rigid_layers.parameters():
            param.requires_grad = False
            
            
        self.Kids = self.mm.Kid+self.mm.Ktex

        N1 = 1024
        N2 = 768
        N3 = 512
        # self.subnetwork1 = nn.Sequential(
        #     nn.Linear(self.backbone_perframe.resnet_lastdim, N1),
        #     nn.Dropout(p=0.3),
            # nn.ReLU(),
            # nn.Linear(N1, N2),
            # nn.Dropout(p=0.3),
            # nn.ReLU(),
            # nn.Linear(N2, N3),
            # nn.ReLU()
            # )
        
        self.output = nn.Linear(self.backbone_perframe.resnet_lastdim, 79+6)
        

    def forward(self, input_im, tforms=None, render=False):
        x_id = self.backbone_id(input_im)[0]
        # x_perframe = self.backbone_perframe.backbone(input_im)
        # x_perframe = torch.flatten(x_perframe,1)
        
        params0 = self.backbone_id.parse_params(x_id)
        alpha_un = x_id[:,:199]
        beta_un = x_id[:,199+79:199*2+79]
        params0['exp'] *= 0
        
        # x = self.output1(x)
        
        # params = self.parse_params(x, alpha_un, beta_un)
        (mask, depth, rim), _ = self.render_image(params0)
        
        input_im[:,0:1,:,:] *= mask
        input_im[:,2:3,:,:] *= mask
        
        input_im[:,1:2,:,:] = rim
        
        y = self.backbone_perframe.backbone(input_im)
        
        y = torch.flatten(y,1)



        # y = self.subnetwork1(y)
        y = self.output(y)
        # # print(rim.shape)
        # # print(mask.shape)
        # minput_im = input_im[:,0:1,:,:]*mask
        
        # outim = torch.cat([M1, M2], dim=1)
        
        
        return y, alpha_un, beta_un, input_im
    
    
    def test(self, input_im, alpha, beta):
        x_id = self.backbone_id(input_im)[0]
        # x_perframe = self.backbone_perframe.backbone(input_im)
        # x_perframe = torch.flatten(x_perframe,1)
        
        params0 = self.backbone_id.parse_params(x_id)
        # alpha_un = x_id[:,:199]
        # beta_un = x_id[:,199+79:199*2+79]
        params0['exp'] *= 0
        params0['alpha'] = alpha
        params0['beta'] = beta
        
        
        # x = self.output1(x)
        
        # params = self.parse_params(x, alpha_un, beta_un)
        (mask, depth, rim), _ = self.render_image(params0)
        
        input_im[:,0:1,:,:] *= mask
        input_im[:,2:3,:,:] *= mask
        
        input_im[:,1:2,:,:] = rim
        
        y = self.backbone_perframe.backbone(input_im)
        
        y = torch.flatten(y,1)



        # y = self.subnetwork1(y)
        y = self.output(y)
        # # print(rim.shape)
        # # print(mask.shape)
        # minput_im = input_im[:,0:1,:,:]*mask
        
        # outim = torch.cat([M1, M2], dim=1)
        
        
        return y
    
    
    def parse_params(self, y, alpha, beta):
        params = self.parse_params_unnormalized(y, alpha, beta)
        if self.label_stds is not None and self.label_means is not None:
            params['alpha'] = (params['alpha']*self.alpha_stds)+self.alpha_means
            params['exp'] = (params['exp']*self.exp_stds)+self.exp_means
            params['beta'] = (params['beta']*self.beta_stds)+self.beta_means
            params['angles'] = (params['angles']*self.angle_stds)+self.angle_means
            params['tau'] = (params['tau']*self.tau_stds)+self.tau_means

        return params
        
    
    
    
    def parse_params_unnormalized(self, y, alpha, beta):
        params =  {'alpha': alpha,
                  'exp': y[:, :79],
                  'beta': beta,
                  'angles': y[:,79:79+3],
                  'tau': y[:,79+3:79+6]}
        return params
        
    
    def render_image(self, params):
        canonical_mesh = self.mm.compute_face_shape(params['alpha'], params['exp'])
        # canonical_mesh[:,:, 1] *= -1
        # R = self.mm.compute_rotation_matrix(params['angles'])
        R = self.mm.compute_rotation_matrix_from_eulerrod(params['angles'])
        
        # view-transformed mesh
        mesh = self.mm.view_transform(canonical_mesh, R, params['tau'])
        texture = self.mm.compute_texture(params['beta'])
        # mesh[:,:, 1] *= -1
        
        # print(mesh.shape)
        p = self.cam.map_to_2d(mesh)
        # print(p.shape)
        # print(mesh.shape)

        return self.renderer(mesh, self.mm.tri, feat=texture), p
        
        # d = y[1].squeeze().detach().cpu().numpy()
        # df = d.flatten()
        # w = np.where(df>0)[0]
        # df = df[w]
        
        # d = (d-df.min())/(df.max()-df.min())
        # d[np.where(d<0)] = 0
        # d[np.where(d>1)] = 1
        
        # im = y[-1].squeeze()[0,:,:].cpu().numpy()
        # plt.imshow(im)
        # plt.plot(lmks[:,0], 224-lmks[:,1], 'x')









