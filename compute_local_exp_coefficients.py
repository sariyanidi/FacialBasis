#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import sys

import numpy as np
from common import rel_ids, create_expression_sequence, chosen_feats

from sklearn.impute import KNNImputer
from sklearn.exceptions import ConvergenceWarning

exp_coeffs_file = sys.argv[1] 
local_exp_coeffs_file = sys.argv[2]
morphable_model =  'BFMmm-23660'
basis_version = 'local_basis_FacialBasis1.0' # '0.0.1.4'

sdir = f'data/{morphable_model}/'
localized_basis_file = f'basis_models/{basis_version}.npy'

basis_set = np.load(localized_basis_file, allow_pickle=True).item()

li = np.loadtxt(f'{sdir}/li.dat').astype(int)
imputer = KNNImputer(n_neighbors=2, weights="uniform")
facial_feats = list(rel_ids.keys())

epsilons = np.loadtxt(exp_coeffs_file)
for i in range(epsilons.shape[1]):
    epsilons[:,i:i+1] = imputer.fit_transform(epsilons[:,i:i+1])

T = epsilons.shape[0]

Es = {}

cf = chosen_feats

if 'chosen_feats' in basis_set:
    cf = basis_set['chosen_feats']

for feat in cf:
    rel_id = rel_ids[feat]
    EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    Es[feat] = np.concatenate((EX, EY, EZ), axis=0)
    

ConvergenceWarning('ignore')
imputer = KNNImputer(n_neighbors=2, weights="uniform")
C = []
for feat in cf:
    rel_id = rel_ids[feat]
    dp = create_expression_sequence(epsilons, Es[feat])
    
    if not basis_set['use_abs']:
        dp = np.diff(dp, axis=0)
    
    coeffs = basis_set[feat].transform(dp)
    
    if not basis_set['use_abs']:
        coeffs = np.cumsum(coeffs, axis=0)
        coeffs = np.concatenate((coeffs[0:1,:], coeffs), axis=0)
    
    coeffs = coeffs.T
    
    C.append(coeffs)
    
C = np.concatenate(C)

np.savetxt(local_exp_coeffs_file, C.T)
