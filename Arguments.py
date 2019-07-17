#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:46:45 2019


List of argumnents usable with parser 


@author: nicholas
"""

import argparse


parser = argparse.ArgumentParser(description='Train an AutoEncoder Recommender and Pred')



parser.add_argument('--id', type=str, metavar='', required=True, help='ID of experience. Will be used when saving file.')


# Data
parser.add_argument('--dataPATH', type=str, metavar='', default='./Data/', \
                    help='Path to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='ReDialRnGChronoTRAIN.json', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='ReDialRnGChronoVALID.json', \
                    help='File name of Dataset to for validation')
parser.add_argument('--exclude_genres', default=False, action='store_true', \
                    help='If arg added, genres not used in input (Dataset part empty for genres)')
parser.add_argument('--no_data_merge', default=False, action='store_true', \
                    help='If arg added, mentionned and to be mentionned data are NOT added. \
                    Used in Dataset. ALWAYS True when for PredChrono.')


# Training
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')
parser.add_argument('--loss_fct', type=str, metavar='', default='BCEWLL', \
                    choices=['BCEWLL', 'BCE'], help='Loss function')
parser.add_argument('--noiseTrain', default=False, action='store_true', \
                    help='If arg added, mimics ReDial inputs by allowing only from 1 to 7 (random)\
                    ratings as inputs.')
parser.add_argument('--noiseEval', default=False, action='store_true', \
                    help='If arg added, mimics ReDial inputs by allowing only from 1 to 7 (random)\
                    ratings as inputs.')
parser.add_argument('--disliked', default=False, action='store_true', \
                    help='If arg added, ratings and masks are added, meaning we now have \
                    for inputs: 0 = not seen, 1 = not liked and 2 = liked.')
parser.add_argument('--weights', type=float, metavar='', default=1, \
                    help='Weights multiplying the errors on ratings of 0 (underrepresented) \
                    during training.  1 -> no weights')
parser.add_argument('--patience', type=int, metavar='', default=1, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')

parser.add_argument('--completionTrain', type=float, metavar='', default=100, \
                    help='% of data used during 1 training epoch ~ "early stopping"')
parser.add_argument('--completionPred', type=float, metavar='', default=100, \
                    help='% of data used for final prediction')
parser.add_argument('--completionPredEpoch', type=float, metavar='', default=100, \
                    help='% of data used for prediction during training (each epoch)')
parser.add_argument('--EARLY', default=False, action='store_true', \
                    help="If arg added, Train at 10%, Pred at 1% and PredChrono at 1%")
# ...for Pred file
parser.add_argument('--completionPredChrono', type=float, metavar='', default=100, \
                    help='% of data used for prediction')
parser.add_argument('--pred_not_liked', default=False, action='store_true', \
                    help='If arg added, PredChrono will be on not liked movies only')


# Model
parser.add_argument('--g_type', type=str, metavar='', default='genres', \
                    choices=['none', 'fixed', 'one', 'genres', 'unit'], \
                    help="Parameter(s) learned for genres inputs. None: no genres, Fixed: no learning, \
                    One: one global parameter, Genres: one parameter by genres, Unit:one parameter per movie,...")
parser.add_argument('--layer1', type=int, metavar='', default=344, \
                    help='Integers corresponding to the first hidden layer size')
parser.add_argument('--layer2', type=int, metavar='', default=27, \
                    help='Integers corresponding to the second hidden layer size. 0 if none.')
parser.add_argument('--activations', type=str, metavar='', default='sigmoid', \
                    choices=['relu', 'sigmoid'],\
                    help='Activations in hidden layers of the model')
parser.add_argument('--last_layer_activation', type=str, metavar='', default='none', \
                    choices=['none', 'sigmoid', 'softmax'],\
                    help='Last layer activation of the model')
parser.add_argument('--preModel', type=str, metavar='', default='none', \
                    help='Path to a pre-trained model to start with. Should \
                    include a GenresWrapper of same type')
# ...for Pred file
parser.add_argument('--M1_path', type=str, metavar='', default='none', \
                    help='Path to a Model 1')
parser.add_argument('--M1_label', type=str, metavar='', default='none', \
                    help='Label for Model 1')
parser.add_argument('--M2_path', type=str, metavar='', default='none', \
                    help='Path to a Model 2')
parser.add_argument('--M2_label', type=str, metavar='', default='none', \
                    help='Label to a Model 1')


# Genres 
parser.add_argument('--genresDict', type=str, metavar='', default='dict_genresInter_idx_UiD.json', \
                    help='File name of Dict of genres')
parser.add_argument('--top_cut', type=int, metavar='', default=100, \
                    help='number of movies in genres vector (for torch Dataset)')


# Metrics
parser.add_argument('--topx', type=int, metavar='', default=100, \
                    help='for NDCG mesure, size of top ranks considered')


# Others
parser.add_argument('--seed', default=False, action='store_true', \
                    help="If arg added, random always give the same")

parser.add_argument('--orion', default=False, action='store_true', \
                    help="If arg added, run Orion - Hyper Parameter search")

parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")

parser.add_argument('--DEBUG', default=False, action='store_true', \
                    help="If arg added, reduced dataset and epoch to 1 for rapid debug purposes")



args = parser.parse_args()






# ASSERTIONS


if args.loss_fct == 'BCE':
    assert args.last_layer_activation != 'none','Need last layer activation with BCE'
if args.loss_fct == 'BCEWLL':
    assert args.last_layer_activation == 'none',"Last layer activation must be 'none' with BCEWLL"

# Pourcentage
assert 0 <= args.completionTrain <=100,'completionTrain should be in [0,100]'
assert 0 <= args.completionPred <=100,'completionPred should be in [0,100]'
assert 0 <= args.completionPredEpoch <=100,'completionPredEpoch should be in [0,100]'





# CONVERSION
# (bunch of hyper-parameters group under a name for efficiency when running)

if args.EARLY:
    args.completionTrain = 10 
    args.completionPred = 1
    args.completionPredEpoch = 1
    
    