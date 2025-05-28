#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:26:18 2023

@author: v
"""
import cv2
import torch
from skimage import transform as trans
from kornia.geometry import warp_affine


import numpy as np


def get_rot_matrix(u):
    # if u.shape[0] == 1:
        # u = u.T
    u = u.reshape(-1,1)
    # print(u.shape)

    theta = np.linalg.norm(u)
    unorm = u / theta
    unorm_skew = skew(unorm)

    I = np.eye(3)
    R = I + np.sin(theta) * unorm_skew + (1 - np.cos(theta)) * (unorm_skew @ unorm_skew)

    IR = I - R

    R1 = np.zeros((3, 3, 3))
    R1[:, :, 0] = u[0] * skew(u)
    R1[:, :, 1] = u[1] * skew(u)
    R1[:, :, 2] = u[2] * skew(u)

    R2 = np.zeros((3, 3, 3))
    R2[:, :, 0] = (IR @ I[:, 0:1]) @ u.T
    R2[:, :, 1] = (IR @ I[:, 1:2]) @ u.T
    R2[:, :, 2] = (IR @ I[:, 2:3]) @ u.T
    
    
    R3 = np.zeros((3,3,3));
    R3[:,:,0] = R1[:,:,0]+R2[:,:,0]-(R2[:,:,0]).T
    R3[:,:,1] = R1[:,:,1]+R2[:,:,1]-(R2[:,:,1]).T
    R3[:,:,2] = R1[:,:,2]+R2[:,:,2]-(R2[:,:,2]).T
    
    
    dR_du1 = (theta ** -2) * (R3[:, :, 0] @ R).reshape(3,3,1)
    dR_du2 = (theta ** -2) * (R3[:, :, 1] @ R).reshape(3,3,1)
    dR_du3 = (theta ** -2) * (R3[:, :, 2] @ R).reshape(3,3,1)

    u1, u2, u3 = u

    dR_du = np.concatenate((dR_du1, dR_du2, dR_du3), axis=2)

    return R, dR_du #, d2R_d11, d2R_d12, d2R_d13, d2R_d22, d2R_d23, d2R_d33



def get_rot_matrix_torch(u, device):
    # if u.shape[0] == 1:
        # u = u.T
    u = u.reshape(-1,1)
    # print(u.shape)

    theta = torch.norm(u)
    unorm = u / theta
    unorm_skew = skew_torch(unorm).to(device)

    I = torch.eye(3).to(device)
    R = I + torch.sin(theta) * unorm_skew + (1 - torch.cos(theta)) * unorm_skew@unorm_skew

    IR = I - R

    R1 = torch.zeros((3, 3, 3)).to(device)
    R1[:, :, 0] = u[0] * skew_torch(u).to(device)
    R1[:, :, 1] = u[1] * skew_torch(u).to(device)
    R1[:, :, 2] = u[2] * skew_torch(u).to(device)

    R2 = torch.zeros((3, 3, 3)).to(device)
    R2[:, :, 0] = (IR @ I[:, 0:1]) @ u.T
    R2[:, :, 1] = (IR @ I[:, 1:2]) @ u.T
    R2[:, :, 2] = (IR @ I[:, 2:3]) @ u.T
    
    
    R3 = torch.zeros((3,3,3)).to(device)
    R3[:,:,0] = R1[:,:,0]+R2[:,:,0]-(R2[:,:,0]).T
    R3[:,:,1] = R1[:,:,1]+R2[:,:,1]-(R2[:,:,1]).T
    R3[:,:,2] = R1[:,:,2]+R2[:,:,2]-(R2[:,:,2]).T
    
    
    dR_du1 = (theta ** -2) * (R3[:, :, 0] @ R).reshape(3,3,1)
    dR_du2 = (theta ** -2) * (R3[:, :, 1] @ R).reshape(3,3,1)
    dR_du3 = (theta ** -2) * (R3[:, :, 2] @ R).reshape(3,3,1)

    u1, u2, u3 = u

    dR_du = torch.cat([dR_du1, dR_du2, dR_du3], axis=2)

    return R, dR_du #, d2R_d11, d2R_d12, d2R_d13, d2R_d22, d2R_d23, d2R_d33



def Uij(i, j):
    U = np.zeros((3, 3))
    U[i, j] = 1
    return U


def skew_torch(u):
    Ux = torch.tensor([[0, -u[2,0], u[1,0]], [u[2,0], 0, -u[0,0]], [-u[1,0], u[0,0], 0]])
    return Ux



def skew(u):
    Ux = np.array([[0, -u[2,0], u[1,0]], [u[2,0], 0, -u[0,0]], [-u[1,0], u[0,0], 0]])
    return Ux



# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31-17, 37-17, 40-17, 43-17, 46-17, 49-17, 55-17]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# # utils for face recognition model
# def estimate_norm(lm_51p, H):
#     # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
#     """
#     Return:
#         trans_m            --numpy.array  (2, 3)
#     Parameters:
#         lm                 --numpy.array  (68, 2), y direction is opposite to v direction
#         H                  --int/float , image height
#     """
#     lm = extract_5p(lm_51p)
#     lm[:, -1] = H - 1 - lm[:, -1]
#     tform = trans.SimilarityTransform()
#     src = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [41.5493, 92.3655], [70.7299, 92.2041]],
#     dtype=np.float32)
#     tform.estimate(lm, src)
#     M = tform.params
#     if np.linalg.det(M) == 0:
#         M = np.eye(3)

#     return M[0:2, :]


# utils for face recognition model
def estimate_norm(lm_51p, H, scale=1.5, off=[0,0]):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (51, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_51p)
    # lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = scale*np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)+np.array(off).reshape(1,2)
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]



# utils for face recognition model
def estimate_inv_norm(lm_51p, H, scale=1.5, off=[0,0]):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (51, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_51p)
    # lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = scale*np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)+np.array(off).reshape(1,2)
    tform.estimate(src, lm)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)
        
    return M[0:2, :]


def estimate_norm_torch(lm_51p, H):
    lm_51p_ = lm_51p.detach().cpu().numpy()
    M = []
    for i in range(lm_51p_.shape[0]):
        M.append(estimate_norm(lm_51p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_51p.device)
    return M




def resize_n_crop_inv_cv(image, M, dsize):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return cv2.warpAffine(image, M, dsize=dsize, flags=cv2.INTER_LANCZOS4)



def resize_n_crop_cv(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return cv2.warpAffine(image, M, dsize=(dsize, dsize), flags=cv2.INTER_LANCZOS4)



def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

