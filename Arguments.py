#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:46:45 2019


%%%%%$%$?$?$?$?$?%@?%$@?*&$&(@#*&@(*?$*&(!?*&



@author: nicholas
"""

import argparse


parser = argparse.ArgumentParser(description='Train an AutoEncoder Recommender')



parser.add_argument('--id', type=str, metavar='', required=True, help='ID of experience. Will be used when saving file.')


# Data
parser.add_argument('--dataPATH', type=str, metavar='', default='./Data/', \
                    help='Path to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='ReDialRnGChronoTRAIN.json', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='ReDialRnGChronoVALID.json', \
                    help='File name of Dataset to for validation')


# Training
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')
parser.add_argument('--patience', type=int, metavar='', default=1, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--EARLY', type=bool, metavar='', default=False, \
                    help="Reduced dataset for early stopping")
parser.add_argument('--criterion', type=str, metavar='', default='BCEWLL', \
                    choices=['BCEWLL', 'BCE'], help='Loss function')



# Model
parser.add_argument('--layer1', type=int, metavar='', default=344, \
                    help='Integers corresponding to the first hidden layer size')
parser.add_argument('--layer2', type=int, metavar='', default=27, \
                    help='Integers corresponding to the second hidden layer size. 0 if none.')
parser.add_argument('--last_layer_activation', type=str, metavar='', default='none', \
                    choices=['none', 'sigmoid', 'softmax'],\
                    help='Last layer activation of the model')


# Genres 
parser.add_argument('--genresDict', type=str, metavar='', default='dict_genresInter_idx_UiD.json', \
                    help='File name of Dict of genres')
parser.add_argument('--top_cut', type=int, metavar='', default=100, \
                    help='number of movies in genres vector (for torch Dataset)')


# Metrcis
parser.add_argument('--topx', type=int, metavar='', default=100, \
                    help='for NDCG mesure, size of top ranks considered')


# Others
parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")

parser.add_argument('--DEBUG', type=bool, metavar='', default=False, \
                    help="Reduced dataset and epoch to 1 for rapid debug purposes")



args = parser.parse_args()





if args.criterion == 'BCE':
    assert args.last_layer_activation != 'none','Need last layer activation with BCE'
if args.criterion == 'BCEWLL':
    assert args.last_layer_activation == 'none',"Last layer activation must be 'none' with BCEWLL"
