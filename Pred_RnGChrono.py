#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:15:28 2019


Predictions with models of type Ratings and Genres with Chronological Data


@author: nicholas
"""


########  IMPORTS  ########


import sys
import json
import torch
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

# Personnal imports
import AutoEncoders 
import Utils
import Settings 
from Arguments import args 
# To use Orion
if args.orion:
    from orion.client import report_results

                                 
                    

########  INIT  ########


# Print agrs that will be used
print(sys.argv)

# Add 1 to nb_movies_in_total because index of movies starts at 1
nb_movies = Settings.nb_movies_in_total + 1

# Cuda availability check
if args.DEVICE == "cuda" and not torch.cuda.is_available():
    raise ValueError("DEVICE specify a GPU computation but CUDA is not available")
  
# Seed 
if args.seed:
    manualSeed = 1
    # Python
  #  random.seed(manualSeed)
    # Numpy
    np.random.seed(manualSeed)
    # Torch
    torch.manual_seed(manualSeed)
    # Torch with GPU
    if args.DEVICE == "cuda":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True





######## DATA ########

######## LOAD DATA 
# R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
print('******* Loading SAMPLES from *******', args.dataPATH + args.dataTrain)
train_data = json.load(open(args.dataPATH + args.dataTrain))
valid_data = json.load(open(args.dataPATH + args.dataValid))
# Use only samples where there is a genres mention
valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
if args.DEBUG: 
    train_data = train_data[:128]
    valid_data = valid_data[:128]

# G (genres) - Format [ [UserID, [movies uID of genres mentionned]] ]    
print('******* Loading GENRES from *******', args.genresDict)
dict_genresInter_idx_UiD = json.load(open(args.dataPATH + args.genresDict))

# Getting the popularity vector 
print('** Including popularity')
popularity = np.load(args.dataPATH + 'popularity_vector.npy')
popularity = torch.from_numpy(popularity).float()


######## CREATING DATASET ListRatingDataset 
print('******* Creating torch datasets *******')
train_dataset = Utils.RnGChronoDataset(train_data, dict_genresInter_idx_UiD, \
                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                       args.no_data_merge, args.noiseTrain, args.top_cut)
valid_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                       args.no_data_merge, args.noiseEval, args.top_cut)
# FOR CHRONO (hence no_data_merge is True). With genres mentions or not
valid_chrono_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
                                         nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                         True, args.noiseEval, args.top_cut)   
# FOR CHRONO (hence no_data_merge is True) + use only samples where there is a genres mention
valid_g_chrono_dataset = Utils.RnGChronoDataset(valid_g_data, dict_genresInter_idx_UiD, \
                                         nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                         True, args.noiseEval, args.top_cut)           


######## CREATE DATALOADER
print('******* Creating dataloaders *******\n\n')    
kwargs = {}
if(args.DEVICE == "cuda"):
    kwargs = {'num_workers': 0, 'pin_memory': False}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)    
# For PredChrono
valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
valid_g_chrono_loader = torch.utils.data.DataLoader(valid_g_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    





########  MODEL  ########
    
    
# Organize args layers 
if args.layer2 == 0:
    layers = [nb_movies, args.layer1]
else: 
    layers = [nb_movies, args.layer1, args.layer2]


# Load Model 1
    print('******* Loading Model 1 *******')      
#    model_base = AutoEncoders.AsymmetricAutoEncoder(layers, nl_type=args.activations, \
#                                                    is_constrained=False, dp_drop_prob=0.0, \
#                                                    last_layer_activations=False, \
#                                                    lla = args.last_layer_activation).to(args.DEVICE)
#    model1 = AutoEncoders.GenresWrapperChrono(model_base, args.g_type).to(args.DEVICE)
#    checkpoint = torch.load(args.M1_path, map_location=args.DEVICE)
#    model1.load_state_dict(checkpoint['state_dict'])

checkpoint1 = torch.load(args.M1_path, map_location=args.DEVICE)
model_base1 = AutoEncoders.AsymmetricAutoEncoder(checkpoint1['layers'], \
                                                 nl_type=checkpoint1['activations'], \
                                                 is_constrained=False, dp_drop_prob=0.0, \
                                                 last_layer_activations=False, \
                                                 lla = checkpoint1['lla']).to(args.DEVICE)
model1 = AutoEncoders.GenresWrapperChrono(model_base1, checkpoint1['g_type']).to(args.DEVICE)
model1.load_state_dict(checkpoint1['state_dict'])

if checkpoint1['criterion'] == 'BCEWLL':
    criterion1 = torch.nn.BCEWithLogitsLoss(reduction='none')
elif checkpoint1['criterion'] == 'BCE':
    criterion1 = torch.nn.BCELoss(reduction='none')





# CHRONO EVALUATION
# If one model (with and without genres)
if args.M2_path == 'none':
    
    # MAke predictions
    print("\n\nPrediction Chronological...")
    l1, l0, e1, e0, a1, a0, mr1, mr0, r1, r0, d1, d0 = \
         Utils.EvalPredictionRnGChrono(valid_g_chrono_loader, model1, criterion1, \
                                       True, args.completionPredChrono, args.topx)
                                 #without_genres is True because only one model
    
    
    # Print results
    print("\n  ====> RESULTS <==== \n")
    print("Global avrg pred error with {:.4f} and without {:.4f}".format(l1, l0)) 
    print("\n  ==> BY Nb of mentions, on to be mentionned Liked <== \n")
    
    avrg_e1, avrg_e0 = Utils.ChronoPlot(e1, e0, 'Avrg pred error')
    print("ReDial liked avrg pred error with {:.4f} and without {:.4f}".format(avrg_e1, avrg_e0))
    
    avrg_a1, avrg_a0 = Utils.ChronoPlot(a1, a0, 'Avrg_rank')
    print("ReDial liked avrg ranks with {:.2f} and without {:.2f}".format(avrg_a1, avrg_a0))
    
    avrg_mr1, avrg_mr0 = Utils.ChronoPlot(mr1, mr0, 'MMRR')
    print("ReDial MMRR with {:.2f} and without {:.2f}".format(avrg_mr1, avrg_mr0))
    
    avrg_r1, avrg_r0 = Utils.ChronoPlot(r1, r0, 'MRR')
    print("ReDial MRR with {:.2f} and without {:.2f}".format(avrg_r1, avrg_r0))
    
    avrg_d1, avrg_d0 = Utils.ChronoPlot(d1, d0, 'NDCG')
    print("ReDial NDCG with {:.2f} and without {:.2f}".format(avrg_d1, avrg_d0))


# If two models
else:   
    
    # Load Model 2
    print('\n******* Loading Model 2 *******')      
    checkpoint2 = torch.load(args.M2_path, map_location=args.DEVICE)
    model_base2 = AutoEncoders.AsymmetricAutoEncoder(checkpoint2['layers'], \
                                                     nl_type=checkpoint2['activations'], \
                                                     is_constrained=False, dp_drop_prob=0.0, \
                                                     last_layer_activations=False, \
                                                     lla = checkpoint2['lla']).to(args.DEVICE)
    model2 = AutoEncoders.GenresWrapperChrono(model_base2, checkpoint2['g_type']).to(args.DEVICE)
    model2.load_state_dict(checkpoint2['state_dict'])
    
    if checkpoint2['criterion'] == 'BCEWLL':
        criterion2 = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif checkpoint2['criterion'] == 'BCE':
        criterion2 = torch.nn.BCELoss(reduction='none')
       
     
    # Make predictions    
    print("\n\nPrediction Chronological Model1...")
    l1, _, e1, _, a1, _, mr1, _, r1, _, d1, _ = \
         Utils.EvalPredictionRnGChrono(valid_chrono_loader, model1, criterion1, \
                                       False, args.completionPredChrono, args.topx)
                             # without_genres False because don't do the no genres pred
    print("Prediction Chronological Model2...")                             
    l2, _, e2, _, a2, _, mr2, _, r2, _, d2, _ = \
         Utils.EvalPredictionRnGChrono(valid_chrono_loader, model2, criterion2, \
                                       False, args.completionPredChrono, args.topx)
                             # without_genres False because don't do the no genres pred


    # Print results
    print("\n  ====> RESULTS <==== \n")
    print("Global avrg pred error with {:.4f} and without {:.4f}".format(l1, l2))
    print("\n  ==> BY Nb of mentions, on to be mentionned Liked <== \n")
    
    avrg_e1, avrg_e2 = Utils.ChronoPlot(e1, e2, 'Avrg pred error', args.M1_label, args.M2_label)
    print("ReDial liked avrg pred error with {:.4f} and without {:.4f}".format(avrg_e1, avrg_e2))
    
    avrg_a1, avrg_a2 = Utils.ChronoPlot(a1, a2, 'Avrg_rank', args.M1_label, args.M2_label)
    print("ReDial liked avrg ranks with {:.2f} and without {:.2f}".format(avrg_a1, avrg_a2))
    
    avrg_mr1, avrg_mr2 = Utils.ChronoPlot(mr1, mr2, 'MMRR', args.M1_label, args.M2_label)
    print("ReDial MMRR with {:.2f} and without {:.2f}".format(avrg_mr1, avrg_mr2))
    
    avrg_r1, avrg_r2 = Utils.ChronoPlot(r1, r2, 'MRR', args.M1_label, args.M2_label)
    print("ReDial MRR with {:.2f} and without {:.2f}".format(avrg_r1, avrg_r2))
    
    avrg_d1, avrg_d2 = Utils.ChronoPlot(d1, d2, 'NDCG', args.M1_label, args.M2_label)
    print("ReDial NDCG with {:.2f} and without {:.2f}".format(avrg_d1, avrg_d2))






























































