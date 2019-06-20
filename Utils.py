#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:53:48 2018


Classes and function for ReDial project.


@author: nicholas
"""

import numpy as np
from torch.utils import data
import torch
import nltk
import matplotlib.pyplot as plt
from statistics import mean


import Settings




"""
DATASET - Sub classes of Pytorch DATASET to prepare for Dataloader

"""


class ListRatingsDataset(data.Dataset):
    """
    This is a class working with [ [UserID, [movies uID], [ratings 0-1]] ].
    Each element of the main list corresponds to one user's ratings for each movie.
    
    Returns a sample (input) and a targeted value (target).
    Also returns a MASK, which is a list of bool of len = nb_movies with 1.0 for movies rated by user, 0.0 elsewhere.
    
    If noise='none', all inputs are returned
    If noise='uniform', returns input with p ratings only, where p follows uniform(1, nb of ratings)
    If noise='one", returns input with one rating missing (for evaluation)
    """
    
    def __init__(self, R_list, nb_movies, noise='none'):
        self.R_list = R_list
        self.nb_movies = nb_movies
        self.noise = noise
        
    def __len__(self):
        "Total number of samples. Here one sample is one user's ratings"
        return len(self.R_list)

    def __getitem__(self, index):
        "Generate one sample of data."
        # Get list of movies and ratings for user number (=index) 
        UserID, l_movies, l_ratings = self.R_list[index]
        # Init
        inputs = torch.zeros(self.nb_movies)
        targets = torch.zeros(self.nb_movies)   
        masks = torch.zeros(self.nb_movies)
        # Targets and Masks
        for i in range(len(l_movies)):
            targets[l_movies[i]] = l_ratings[i]
            masks[l_movies[i]] = 1.0     
        # Manage noise
        if self.noise == 'asReDial' or self.noise == 'uniform' or self.noise == 'one':
            if self.noise == 'asReDial':
                max_input = min(7, len(l_movies))   
                p = torch.randint(0, max_input, (1,)).type(torch.uint8)                
            if self.noise == 'uniform':
                p = torch.randint(0, len(l_movies)+1, (1,)).type(torch.uint8)
            if self.noise == 'one':
                p = torch.randint(len(l_movies)-1, len(l_movies), (1,)).type(torch.uint8)
            ind_to_take = torch.randperm(len(l_movies))[:p]
            l_movies = torch.IntTensor(l_movies)
            l_ratings = torch.IntTensor(l_ratings)
            l_movies = l_movies[ind_to_take]
            l_ratings = l_ratings[ind_to_take]
        # Inputs
        for i in range(len(l_movies)):
            inputs[l_movies[i]] = l_ratings[i]
        return masks, inputs, targets 



class RatingsGenresDataset(data.Dataset):
    """
    INPUT: 
        R_list is movie list. Format [ [UserID, [movies uID], [ratings 0-1]] ].  Each element of the main list 
        corresponds to one user's ratings for each movie.
        
        G_list is genres list. Format [ [UserID, [movies UiD of genres mentionnned in ConvID = UserID]] ]. 
    
        If noise='none', all inputs are returned
        If noise='uniform', returns input with p ratings only, where p follows uniform(1, nb of ratings)
        If noise='one", returns input with one rating missing (for evaluation)
    
    RETUNRS:
        A sample (INPUT) and a targeted value (TARGET) corresponding to movies ratings.
        Also returns a MASK, which is a list of bool of len = nb_movies with 1.0 for movies rated by user, 0.0 elsewhere.
        Finally, returns a GENRES vector with one for all movies having ALL genres mentionned by User.

    """
    
    def __init__(self, R_list, G_list, nb_movies, noise='none'):
        self.R_list = R_list
        self.G_list = G_list
        self.nb_movies = nb_movies
        self.noise = noise
        
    def __len__(self):
        "Total number of samples. Here one sample is one user's ratings"
        return len(self.R_list)

    def __getitem__(self, index):
        "Generate one sample of data."
        # Get list of movies and ratings for user number (=index) 
        R_UserID, l_movies, l_ratings = self.R_list[index]
        G_UserID, l_genres = self.G_list[index]
        # Test if same UserID
        if R_UserID != G_UserID:
            raise ValueError("Not the same UserID", R_UserID, G_UserID)
        # Init
        ratings = torch.zeros(self.nb_movies)
        targets = torch.zeros(self.nb_movies)   
        masks = torch.zeros(self.nb_movies)
        genres = torch.zeros(self.nb_movies)
        # Targets and Masks
        for i in range(len(l_movies)):
            targets[l_movies[i]] = l_ratings[i]
            masks[l_movies[i]] = 1.0     
        # Manage noise
        if self.noise == 'uniform' or self.noise == 'one':
            if self.noise == 'uniform':
                p = torch.randint(0, len(l_movies)+1, (1,)).type(torch.uint8)
            if self.noise == 'one':
                p = torch.randint(len(l_movies)-1, len(l_movies), (1,)).type(torch.uint8)
            ind_to_take = torch.randperm(len(l_movies))[:p]
            l_movies = torch.IntTensor(l_movies)
            l_ratings = torch.IntTensor(l_ratings)
            l_movies = l_movies[ind_to_take]
            l_ratings = l_ratings[ind_to_take]
        # Inputs
        for i in range(len(l_movies)):
            ratings[l_movies[i]] = l_ratings[i]
        # Genres
        for m in l_genres:
            genres[m] = 1.0

        return masks, (ratings, genres), targets



class RatingsGenresNormalizedDataset(data.Dataset):
    """
    ****** SAME AS RatingsGenresNormalizedDataset BUT WITH GENRES VECTORS OF LENGHT ONE ******
    
    INPUT: 
        R_list is movie list. Format [ [UserID, [movies uID], [ratings 0-1]] ].  Each element of the main list 
        corresponds to one user's ratings for each movie.
        
        G_list is genres list. Format [ [UserID, [movies UiD of genres mentionnned in ConvID = UserID]] ]. 
    
        If noise='none', all inputs are returned
        If noise='uniform', returns input with p ratings only, where p follows uniform(1, nb of ratings)
        If noise='one", returns input with one rating missing (for evaluation)
    
    RETUNRS:
        A sample (INPUT) and a targeted value (TARGET) corresponding to movies ratings.
        Also returns a MASK, which is a list of bool of len = nb_movies with 1.0 for movies rated by user, 0.0 elsewhere.
        Finally, returns a GENRES vector with one for all movies having ALL genres mentionned by User.

    """
    
    def __init__(self, R_list, G_list, nb_movies, popularity, noise='none'):
        self.R_list = R_list
        self.G_list = G_list
        self.nb_movies = nb_movies
        self.popularity = popularity
        self.noise = noise
        
    def __len__(self):
        "Total number of samples. Here one sample is one user's ratings"
        return len(self.R_list)

    def __getitem__(self, index):
        "Generate one sample of data."
        # Get list of movies and ratings for user number (=index) 
        R_UserID, l_movies, l_ratings = self.R_list[index]
        G_UserID, l_genres = self.G_list[index]
        # Test if same UserID
        if R_UserID != G_UserID:
            raise ValueError("Not the same UserID", R_UserID, G_UserID)
        # Init
        ratings = torch.zeros(self.nb_movies)
        targets = torch.zeros(self.nb_movies)   
        masks = torch.zeros(self.nb_movies)
        genres = torch.zeros(self.nb_movies)
        # Targets and Masks
        for i in range(len(l_movies)):
            targets[l_movies[i]] = l_ratings[i]
            masks[l_movies[i]] = 1.0     
        # Manage noise
        if self.noise == 'uniform' or self.noise == 'one':
            if self.noise == 'uniform':
                p = torch.randint(0, len(l_movies)+1, (1,)).type(torch.uint8)
            if self.noise == 'one':
                p = torch.randint(len(l_movies)-1, len(l_movies), (1,)).type(torch.uint8)
            ind_to_take = torch.randperm(len(l_movies))[:p]
            l_movies = torch.IntTensor(l_movies)
            l_ratings = torch.IntTensor(l_ratings)
            l_movies = l_movies[ind_to_take]
            l_ratings = l_ratings[ind_to_take]
        # Inputs
        for i in range(len(l_movies)):
            ratings[l_movies[i]] = l_ratings[i]
        # Genres
        for m in l_genres:
            genres[m] = 1.0
        # Include popularity in genres
        genres = genres * self.popularity
        # Take top 100 movies
        genres_cut = torch.zeros(self.nb_movies)
        genres_cut[genres.topk(100)[1]] = genres.topk(100)[0]
        genres = genres_cut
        
        # Normalize vector
        genres = torch.nn.functional.normalize(genres, dim=0)
        
        return masks, (ratings, genres), targets





class RnGChronoDataset(data.Dataset):
    """    
    
    ****** Now inputs and targets are seperated ******
    
    
    INPUT: 
        RnGlist format is:
            ["ConvID", [(UiD, Rating) mentionned], ["genres"], [(UiD, Rating) to be mentionned]]
        top_cut is the number of movies in genres vector
    
    RETUNRS:
        masks, (inputs, genres) and targets. 
        Genres is vector with value for top 100 movies of intersection of genres mentionned 
        by User, normalized.

    """
    
    def __init__(self, RnGlist, dict_genresInter_idx_UiD, nb_movies, popularity, top_cut=100):
        self.RnGlist = RnGlist
        self.dict_genresInter_idx_UiD = dict_genresInter_idx_UiD
        self.nb_movies = nb_movies
        self.popularity = popularity
        self.top_cut = top_cut
        
    def __len__(self):
        "Total number of samples. Here one sample corresponds to a new mention in Conversation"
        return len(self.RnGlist)

    def __getitem__(self, index):
        "Generate one sample of data."
        # Get list of movies and ratings for user number (=index) 
        ConvID, l_inputs, l_genres, l_targets = self.RnGlist[index]
        
        # Init
        inputs = torch.zeros(self.nb_movies)
        targets = torch.zeros(self.nb_movies)   
        masks_inputs = torch.zeros(self.nb_movies)
        masks_targets = torch.zeros(self.nb_movies)
        genres = torch.zeros(self.nb_movies)
        
        # Inputs
        for uid, rating in l_inputs:
            inputs[uid] = rating
            masks_inputs[uid] = 1
        
        # Targets 
        for uid, rating in l_targets:
            targets[uid] = rating
            masks_targets[uid] = 1
        
        # Genres
        # Turn list of genres into string
        str_genres = str(l_genres)
        # Try - if no movies of that genres (key error)
        try:
            genres_idx, l_genres_uid = self.dict_genresInter_idx_UiD[str_genres] 
        except:
   #         print('No movie with genres:', str_genres)
            genres_idx = 1
        # If there is a genres...   (no else needed, since already at 0)
        if genres_idx != 1:
            for uid in l_genres_uid:
                genres[uid] = 1.0
                
            """normalization and popularity"""
            # Include popularity in genres
            genres = genres * self.popularity
            # Take top 100 movies
            genres_cut = torch.zeros(self.nb_movies)
            genres_cut[genres.topk(self.top_cut)[1]] = genres.topk(self.top_cut)[0]
            genres = genres_cut  
            # Normalize vector
            genres = torch.nn.functional.normalize(genres, dim=0)
        
        
        return (masks_inputs, masks_targets), (inputs, (genres_idx, genres)), targets






"""

TRAINING AND EVALUATION 

"""



def TrainReconstruction(train_loader, model, criterion, optimizer, DEVICE, EARLY=False):
    model.train()
    train_loss = 0
    nb_batch = len(train_loader) 
    
    
    
    """ """
    pred_mean_values = []
    
    """ """
   

     
    for batch_idx, (masks, inputs, targets) in enumerate(train_loader):
        
        # Early stopping at 10% of data
        if EARLY:
            if batch_idx > (nb_batch/10): 
                print('EARLY stopping')
                break
        
        # Print update
        if batch_idx % 100 == 0: 
            print('Batch {} out of {}.  Loss:{}'.format(batch_idx, nb_batch,\
                  train_loss/(batch_idx+1)))  
        
        targets = targets.to(DEVICE)
        
        # Test if list format == Genres Case so inputs has genres also 
        # and can't put a list .to(DEVICE)
        if type(inputs) == list:
            inputs[0] = inputs[0].to(DEVICE)
            # Chrono Case: 
            if type(inputs[1]) == list:
                # ...cover genres idx
                inputs[1][0] = inputs[1][0].to(DEVICE)
                inputs[1][1] = inputs[1][1].to(DEVICE)
                # cover masks seperation
                masks[0] = masks[0].to(DEVICE)
                masks[1] = masks[1].to(DEVICE)
                # ...add inputs and targets
                sum_i_t = (inputs[0] + targets).clone()
                inputs[0] = sum_i_t
                targets = sum_i_t
                masks = masks[0] + masks[1]
            # Genres Case, but not chrono
            else:
                inputs[1] = inputs[1].to(DEVICE)
                masks = masks.to(DEVICE)
        # Basic case
        else: 
            inputs = inputs.to(DEVICE)
            # Not Genres case, so can add weights to BCE training
            # 94% are target 1, so *94 targets at 0
            weights = (targets -1.0) * -94.0 + targets
            criterion.weight = weights.float()
        
        # Add weights on targets rated 0 because outnumbered 94 times by targets 1
        weights = (masks == 1) * (targets == 0) * 93 + torch.ones_like(targets, dtype=torch.uint8)
        criterion.weight = weights.float()

        """ UNcomment TO TRAIN WITHOUT GENRES"""
        # Genres removed from inputs 
#        inputs[1][0] = torch.ones(inputs[0].size(0), dtype=torch.uint8)
#        inputs[1][1] = torch.zeros(inputs[0].size(0), 48272)

        # re-initialize the gradient computation
        optimizer.zero_grad()   
        
        pred = model(inputs)

            
        
        
        
        """ To look into pred values evolution during training"""
        if model.model_pre.lla == 'none':
            pred_mean_values.append((torch.nn.Sigmoid()(pred)[:,Settings.l_ReDUiD]).mean())
        else:
            pred_mean_values.append((pred[:,Settings.l_ReDUiD]).mean())
        """ """
        
        
        
     
        """ Try reconstruction including genres """
#        targets = targets + model.g * inputs[1][1]
#        masks = masks * (inputs[1][1] != 0).float()
     
        
        # Using only predictions of movies that were rated
     #   pred = pred * masks
        nb_ratings = masks.sum()
     #   del(masks)
        loss = (criterion(pred, targets) * masks).sum()
        # Put weights back at None for other evaluations
        criterion.weight = None

        loss = loss / nb_ratings
        if loss < 0: 
            print("negative loss, with pred min at:", pred.min())
        loss.backward()
        optimizer.step()
        train_loss += loss
        
    train_loss /= nb_batch
        
    return (pred_mean_values, train_loss) 



def EvalReconstruction(valid_loader, model, criterion, DEVICE, EARLY=False):
    model.eval()
    eval_loss = 0
    nb_batch = len(valid_loader)
    
    with torch.no_grad():
        for batch_idx, (masks, inputs, targets) in enumerate(valid_loader):
            
            # Early stopping at 10% of data
            if EARLY:
                if batch_idx > (nb_batch/10): 
                    print('EARLY stopping')
                    break
            
            # Print update
            if batch_idx % 100 == 0: 
                print('Batch {} out of {}.  Loss:{}'.format(batch_idx, nb_batch,\
                      eval_loss/(batch_idx+1)))  
        
            targets = targets.to(DEVICE)
        
            # Test if list format == Genres Case so inputs has genres also 
            # and can't put a list .to(DEVICE)
            if type(inputs) == list:
                inputs[0] = inputs[0].to(DEVICE)
                # Chrono Case: 
                if type(inputs[1]) == list:
                    # ...cover genres idx
                    inputs[1][0] = inputs[1][0].to(DEVICE)
                    inputs[1][1] = inputs[1][1].to(DEVICE)
                    # cover masks seperation
                    masks[0] = masks[0].to(DEVICE)
                    masks[1] = masks[1].to(DEVICE)
                    # ...add inputs and targets
                    sum_i_t = (inputs[0] + targets).clone()
                    inputs[0] = sum_i_t
                    targets = sum_i_t
                    masks = masks[0] + masks[1]
                # Genres Case, but not chrono
                else:
                    inputs[1] = inputs[1].to(DEVICE)
                    masks = masks.to(DEVICE)
            # Basic case
            else: 
                inputs = inputs.to(DEVICE)
                
                
            pred = model(inputs)  
            
            # Using only movies that were rated
        #   pred = pred * masks
            nb_ratings = masks.sum()
       #     del(masks)
            loss = (criterion(pred, targets) * masks).sum()
            loss = loss / nb_ratings
            eval_loss += loss
    
    eval_loss /= nb_batch 
    
    return eval_loss



def EvalPrediction(loader, model, criterion, DEVICE, EARLY=False):
    """
    Takes a list of ratings from a user (len = nb_movies) and a list of movies uID
    For each movie m in l_movies, predicts m's rating according to rest of movies in l_movies
    Returns a list of errors for each prediction (len = len(l_movies))
    
    ******* LOADER must be of BATCH SIZE == 1 **********
    """
    model.eval()
    l_loss = []
    l_rank_liked = []
    l_rank_disliked = []
    nb_batch = len(loader)
    
    with torch.no_grad():
        # For each user
        for batch_idx, (masks, inputs, _) in enumerate(loader):
            
            # Early stopping at 1% of data
            if EARLY:
                if batch_idx > (nb_batch/100): 
                    print('EARLY stopping')
                    break
                
            # Print Update
            if batch_idx % 100 == 0:
                print('Batch {} out of {}.'.format(batch_idx, nb_batch))     
            
            # For each movie
            for m, mask in enumerate(masks[0]):           # [0] because loader returns list of list (batches usually > 1)
                if mask == 1:
                    inputs = inputs.to(DEVICE)
                    # Get the rating for movie m
                    r = inputs[0][m].clone().detach()         # To insure deepcopy and not reference and no backprop
                    # "Hide" the rating of movie m
                    inputs[0][m] = 0
                    # TODO ############################ THIS IS WHERE WE SHOULD UPDATE INPUTS WITH GENRES (or other KB)
                    # Get the predictions
                    pred = model(inputs)
                 #   pred = torch.ones_like(inputs).to(DEVICE)
                    # Put the ratings in original condition  
                    # TODO: ######## LATER REMOVE UPDATES FROM KB
                    inputs[0][m] = r
                    # Evaluate error
                    # error = (r - pred[0][m])**2
                    error = criterion(pred[0][m], r)
                    l_loss.append(error.item())
                    # Ranking of this prediction among all predictions
                    
                    
                    """ Adding Sigmoid to pred if BCELogits used """
#                    if model.model_pre.lla == 'none':
#                        pred = torch.nn.Sigmoid()(pred)
                    """ """
                    
     #               ranks = (torch.sort(pred[0][Settings.l_ReDUiD], descending=True)[0] == pred[0][m]).nonzero()
                    ranks = (torch.sort(pred[0], descending=True)[0] == pred[0][m]).nonzero()
                 #   print("Value of rating (r) is:", r)
                    if r == 1.0:
                 #       print("Added to the liked movies ranking")
                        l_rank_liked.append(ranks[0,0].item())
                    elif r == 0.0:
                 #       print("Added to the DISliked movies ranking")
                        l_rank_disliked.append(ranks[0,0].item())

                    
    l_loss = np.array(l_loss)
    l_rank_liked = np.array(l_rank_liked)
    l_rank_disliked= np.array(l_rank_disliked)
    mean_error = np.mean(l_loss)
    mean_rank_liked = np.mean(l_rank_liked)
    mean_rank_disliked = np.mean(l_rank_disliked)
    
    return mean_error, mean_rank_liked, mean_rank_disliked


def EvalPredictionGenres(loader, model, criterion, DEVICE):
    """
    Takes a list of TUPLES (ratings AND GENRES) from a user (len = nb_movies) and a list of movies uID
    For each movie m in l_movies, predicts m's rating according to rest of movies in l_movies
    Returns a list of errors for each prediction (len = len(l_movies))
    
    ******* LOADER must be of BATCH SIZE == 1 **********
    """
    model.eval()
    l_loss = []
    l_rank_liked = []
    l_rank_disliked = []
    
    with torch.no_grad():
        # For each user
        for batch_idx, (masks, inputs, _) in enumerate(loader):
            # For each movie
            for m, mask in enumerate(masks[0]):           # [0] because loader returns list of list (batches usually > 1)
                if mask == 1:
                    inputs[0] = inputs[0].to(DEVICE)
                    inputs[1] = inputs[1].to(DEVICE)
                    # Get the rating for movie m
                    r = inputs[0][0][m].clone().detach()         # To insure deepcopy and not reference
                    # "Hide" the rating of movie m
                    inputs[0][0][m] = 0
                    # TODO ############################ THIS IS WHERE WE SHOULD UPDATE INPUTS WITH GENRES (or other KB)
                    # Get the predictions
                    pred = model(inputs)
                 #   pred = torch.ones_like(inputs).to(DEVICE)
                    # Put the ratings in original condition  
                    # TODO: ######## LATER REMOVE UPDATES FROM KB
                    inputs[0][0][m] = r
                    # Evaluate error
                    # error = (r - pred[0][m])**2
                    error = criterion(pred[0][m], r)
                    l_loss.append(error.item())
                    # Ranking of this prediction among all predictions. 
                    # ranks is a 2D array of size (nb of pred with same value as m, position of value)
                    
                    
                    """ Adding Sigmoid to pred if BCELogits used """
                    if model.model_pre.lla == 'none':
                        pred = torch.nn.Sigmoid()(pred)
                    """ """
                    
                    
                    ranks = (torch.sort(pred[0][Settings.l_ReDUiD], descending=True)[0] == pred[0][m]).nonzero() + 1
                 #   print("Value of rating (r) is:", r)
                    if r == 1.0:
                 #       print("Added to the liked movies ranking")
                        l_rank_liked.append(ranks[0,0].item())
                    elif r == 0.0:
                 #       print("Added to the DISliked movies ranking")
                        l_rank_disliked.append(ranks[0,0].item())
                 #   print("number of predictions with same value", ranks.size()[0])
                    
    l_loss = np.array(l_loss)
    l_rank_liked = np.array(l_rank_liked)
    l_rank_disliked= np.array(l_rank_disliked)
    mean_error = np.mean(l_loss)
    mean_rank_liked = np.mean(l_rank_liked)
    mean_rank_disliked = np.mean(l_rank_disliked)

    return mean_error, mean_rank_liked, mean_rank_disliked




def EvalPredictionGenresRaw(loader, model, criterion, DEVICE, EARLY=False):
    """
    Same as EvalPredictionGenres, but values returned are complete (not their mean)
    """
    model.eval()
    l_loss = []
    l_rank_liked = []
    l_rank_disliked = []
    nb_batch = len(loader)
    
    with torch.no_grad():
        # For each user
        for batch_idx, (masks, inputs, targets) in enumerate(loader):
            
            # Early stopping at 10% of data
            if EARLY:
                if batch_idx > (nb_batch/10): 
                    print('EARLY stopping')
                    break
                
                
            # Print Update
            if batch_idx % 1000 == 0:
                print('Batch {} out of {}.'.format(batch_idx, nb_batch))     
            
            
            # Prepare data for cuda
            # Manage Chrono Case
            if type(masks) == list: 
                masks = masks[0] + masks[1]
            inputs[0] = inputs[0].to(DEVICE)
            # Chrono Case
            if type(inputs[1]) == list:
                inputs[1][0] = inputs[1][0].to(DEVICE)
                inputs[1][1] = inputs[1][1].to(DEVICE)
                targets = targets.to(DEVICE)
                inputs[0] = inputs[0] + targets
            else:
                inputs[1] = inputs[1].to(DEVICE)
                
            # For each movie
            for m, mask in enumerate(masks[0]):           # [0] because loader returns list of list (batches usually > 1)
                if mask == 1:                             # If a rated movie    
                        
                    # Get the rating for movie m
                    r = inputs[0][0][m].clone().detach()         # To insure deepcopy and not reference
                    # "Hide" the rating of movie m
                    inputs[0][0][m] = 0
                    # Get the predictions
                    pred = model(inputs)
                    # Put the ratings in original condition  
                    inputs[0][0][m] = r
                    # Evaluate error
                    error = criterion(pred[0][m], r)
                    l_loss.append(error.item())
                    # Ranking of this prediction among all predictions. 
                    # ranks is a 2D array of size (nb of pred with same value as m, position of value)
                    
                    
                    """ Adding Sigmoid to pred if BCELogits used """
                    if model.model_pre.lla == 'none':
                        pred = torch.nn.Sigmoid()(pred)
                    """ """
                    
                    
                    ranks = (torch.sort(pred[0][Settings.l_ReDUiD], descending=True)[0] == pred[0][m]).nonzero() + 1
                 #   print("Value of rating (r) is:", r)
                    if r == 1.0:
                 #       print("Added to the liked movies ranking")
                        l_rank_liked.append(ranks[0,0].item())
                    elif r == 0.0:
                 #       print("Added to the DISliked movies ranking")
                        l_rank_disliked.append(ranks[0,0].item())
                 #   print("number of predictions with same value", ranks.size()[0])
      
          #  if batch_idx > 10: break

                    
    l_loss = np.array(l_loss)
    l_rank_liked = np.array(l_rank_liked)
    l_rank_disliked= np.array(l_rank_disliked)

    return l_loss,l_rank_liked, l_rank_disliked




def EvalPredictionRnGChrono(valid_loader, model, criterion, DEVICE, topx=100, EARLY=False):
    """
    Prediction on targets = to be mentionned movies...
    
    ** Only works with RnGChronoDataset **
    
    """
    model.eval()
    nb_batch = len(valid_loader)
    
    eval_loss_with_genres = 0
    eval_loss_without_genres = 0
    
    results_error_with_genres = {}
    results_error_without_genres = {}
    results_Avrg_Ranks_with_genres = {}
    results_Avrg_Ranks_without_genres = {}
    results_MRR_with_genres = {}
    results_MRR_without_genres = {}
    results_RR_with_genres = {}
    results_RR_without_genres = {}
    results_DCG_with_genres = {}
    results_DCG_without_genres = {}
                
                
    with torch.no_grad():
        for batch_idx, (masks, inputs, targets) in enumerate(valid_loader):
            
            # Early stopping at 10% of data
            if EARLY:
                if batch_idx > (nb_batch/10): 
                    print('EARLY stopping')
                    break
            
            # Print Update
            if batch_idx % 10 == 0:
                print('Batch {} out of {}.  Loss:{}'\
                      .format(batch_idx, nb_batch, eval_loss_with_genres/(batch_idx+1)))
            
            # Prepare data for cuda
            inputs[0] = inputs[0].to(DEVICE)
            inputs[1][0] = inputs[1][0].to(DEVICE)
            inputs[1][1] = inputs[1][1].to(DEVICE)
            masks[0] = masks[0].to(DEVICE)
            masks[1] = masks[1].to(DEVICE)
            targets = targets.to(DEVICE)     
            
    # WITH GENRES
            # Make a pred
            pred = model(inputs)  
            
            # LOSS - Using only movies to be mentionned that were rated
       #     pred_masked = pred * masks[1]
            nb_ratings = masks[1].sum()
            loss = (criterion(pred, targets) * masks[1]).sum()
            loss = loss / nb_ratings
            eval_loss_with_genres += loss
    
    
            """ Adding Sigmoid to pred if BCELogits used """
            if model.model_pre.lla == 'none':
                pred = torch.nn.Sigmoid()(pred)
            """ """
    
    
            # NRR & NDCG
            # Need to evaluate each samples seperately, since diff number of targets
            # For ReDial Chrono Dataset
            for i, sample in enumerate(pred[:,Settings.l_ReDUiD]):
                
                # Insure their is at least one target movie rated 1
                # (if all rated 0, sample not considered)
                if targets[i].sum() == 0: continue
                
                # Get error on pred ratings
                error = ((criterion(pred[i], targets[i]) * masks[1][i]).sum() / masks[1][i].sum()).item()
                # ... get Ranks for targets (not masks[1] because only care about liked movies)
                rk, avrg_rk, mrr, rr, ndcg = Ranks(sample, \
                                                  pred[i][targets[i].nonzero().flatten().tolist()],\
                                                  topx)                
                # Get the number of inputs mentionned before genres
                qt_mentionned_before = masks[0][i].sum(dtype=torch.uint8).item()
                
                # Add Ranks results to appropriate dict
                if qt_mentionned_before in results_RR_with_genres.keys():
                    results_error_with_genres[qt_mentionned_before].append(error)
                    results_Avrg_Ranks_with_genres[qt_mentionned_before].append(avrg_rk)
                    results_MRR_with_genres[qt_mentionned_before].append(mrr)
                    results_RR_with_genres[qt_mentionned_before].append(rr)
                    results_DCG_with_genres[qt_mentionned_before].append(ndcg)
                else:
                    results_error_with_genres[qt_mentionned_before] = [error]
                    results_Avrg_Ranks_with_genres[qt_mentionned_before] = [avrg_rk]
                    results_MRR_with_genres[qt_mentionned_before] = [mrr]
                    results_RR_with_genres[qt_mentionned_before] = [rr]
                    results_DCG_with_genres[qt_mentionned_before] = [ndcg]

            
            
    # WITHOUT GENRES
            # Make a pred with genres removed from inputs 
            inputs[1][0] = torch.ones(inputs[0].size(0), dtype=torch.uint8)
            inputs[1][1] = torch.zeros(inputs[0].size(0), 48272)
            pred = model(inputs)  
            
            # LOSS - Using only movies to be montionned that were rated
       #     pred_masked = pred * masks[1]
            nb_ratings = masks[1].sum()
            loss = (criterion(pred, targets) * masks[1]).sum()
            loss = loss / nb_ratings
            eval_loss_without_genres += loss
    
    
            """ Adding Sigmoid to pred if BCELogits used """
            if model.model_pre.lla == 'none':
                pred = torch.nn.Sigmoid()(pred)
            """ """
    
    
            # NRR & NDCG
            # Need to evaluate each samples seperately, since diff number of targets
            # For ReDial Chrono Dataset
            for i, sample in enumerate(pred[:,Settings.l_ReDUiD]):
                
                # Insure their is at least one target movie rated 1
                # (if all rated 0, sample not considered)
                if targets[i].sum() == 0: continue
            
                # Get error on pred ratings
                error = ((criterion(pred[i], targets[i]) * masks[1][i]).sum() / masks[1][i].sum()).item()         
                # ... get Ranks for targets (not masks[1] because only care about liked movies)
                rk, avrg_rk, mrr, rr, ndcg = Ranks(sample, \
                                                  pred[i][targets[i].nonzero().flatten().tolist()],\
                                                  topx)
                # Get the number of inputs mentionned before genres
                qt_mentionned_before = masks[0][i].sum(dtype=torch.uint8).item()
                
                # Add Ranks results to appropriate dict
                if qt_mentionned_before in results_RR_without_genres.keys():
                    results_error_without_genres[qt_mentionned_before].append(error)
                    results_Avrg_Ranks_without_genres[qt_mentionned_before].append(avrg_rk)
                    results_MRR_without_genres[qt_mentionned_before].append(mrr)
                    results_RR_without_genres[qt_mentionned_before].append(rr)
                    results_DCG_without_genres[qt_mentionned_before].append(ndcg)
                else:
                    results_error_without_genres[qt_mentionned_before] = [error]
                    results_Avrg_Ranks_without_genres[qt_mentionned_before] = [avrg_rk]
                    results_MRR_without_genres[qt_mentionned_before] = [mrr]
                    results_RR_without_genres[qt_mentionned_before] = [rr]
                    results_DCG_without_genres[qt_mentionned_before] = [ndcg]
    
         #  if batch_idx > 10: break
    
        eval_loss_with_genres /= nb_batch 
        eval_loss_without_genres /= nb_batch


    return eval_loss_with_genres, eval_loss_without_genres, \
           results_error_with_genres, results_error_without_genres, \
           results_Avrg_Ranks_with_genres, results_Avrg_Ranks_without_genres, \
           results_MRR_with_genres, results_MRR_without_genres,\
           results_RR_with_genres, results_RR_without_genres,\
           results_DCG_with_genres, results_DCG_without_genres







"""

CLASSES

"""


class Conversation:
    """
    Class to work with the original Conversation Data from ReDial
    """
    
    def __init__(self, json):
        
        self.json = json
        self.id = json["conversationId"]
        self.movie_mentions = [k for k in json["movieMentions"]]
        self.movie_seek_liked = [m for m in json["initiatorQuestions"] if \
                                 json["initiatorQuestions"][m]["liked"] == 1 and \
                                 json["initiatorQuestions"][m]["liked"] == 1]
        self.movie_seek_notliked = [m for m in json["initiatorQuestions"] if \
                                   json["initiatorQuestions"][m]["liked"] == 0 and \
                                   json["initiatorQuestions"][m]["liked"] == 0]
        self.seek_wId = self.json["initiatorWorkerId"]
        self.recom_wId = self.json["respondentWorkerId"]
    
    
            
    def getSeekRecomText(self):
        self.seek_text = []
        self.recom_text = []
        
        for msg in self.json["messages"]:
            if msg["senderWorkerId"] == self.seek_wId:
                self.seek_text.append(msg["text"])
            else:
                self.recom_text.append(msg["text"])
                               
                
    def getSeekerGenres(self, genres_to_find):
        # Get unique genres mentionned in all ut of seekers 
        self.genres_seek = getGenresListOfTextToOneList(self.seek_text, genres_to_find)


def getGenresFromOneText(text, genres_to_find):
    """
    Take a string
    Returns list of genres (strings) mentionned in text
    
    EXAMPLE: 
        In: "Hey everybody, meet warren, he's a kid a bit drama"
        Out: ['drama', 'kid']
    """
    genres_in_text = []
    # Get list of unique words mentionned in text
    words = nltk.word_tokenize(text.lower())
    for g in genres_to_find:
        for w in words:
            if g == w: 
                genres_in_text.append(g)
    return genres_in_text



def getGenresListOfTextToOneList(l_text, genres_to_find):
    """
    Take a list of strings
    Returns list of unique genres (strings) mentionned in all texts
    
    EXAMPLE:
        In: ["Hey everybody, meet warren, he's a kid a bit drama", 
             "Sentence with no genre",
             "Genres repeating, like drama",
             "Horror movies are fun"]
        Out: ['drama', 'kid', 'horror']
    """
        
    l_genres = []
    for text in l_text:
        genres_in_text = getGenresFromOneText(text, genres_to_find)
        # Concat only if genres retreived  
        if genres_in_text != []:
            l_genres += genres_in_text
    # Return without duplicates
    return list(set(l_genres))






"""

OTHERS

"""


def Splitting(l_items, ratio_1, ratio_2, ratio_3):
    """
    Splitting a list of items randowly, into sublists, according to ratios.
    Returns the 3 sublist (2 could be empty)
    """
    # Make sure ratios make sense
    if ratio_1 + ratio_2 + ratio_3 != 1:
        raise Exception("Total of ratios need to be 1, got {}".format(ratio_1 + ratio_2 + ratio_3))
    size_1 = round(ratio_1 * len(l_items))
    size_2 = round(ratio_2 * len(l_items))
    np.random.shuffle(l_items)
    sub_1 = l_items[:size_1]
    sub_2 = l_items[size_1:size_1+size_2]
    sub_3 = l_items[size_1+size_2:]

    return sub_1, sub_2, sub_3 



def SplittingDataset(full_dataset, ratio_train, ratio_valid, ratio_test):
    """
    Splitting a torch dataset into Train, Valid and Test sets randomly.
    Returns the 3 torch datasets
    """
    train_size = round(ratio_train * len(full_dataset))
    valid_size = round(ratio_valid * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    # Split train & valid from test 
    train_n_valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size + valid_size, test_size])
    # Split train and valid
    train_dataset, valid_dataset = torch.utils.data.random_split(train_n_valid_dataset, [train_size, valid_size])
    return train_dataset, valid_dataset, test_dataset
    


# DCG (Discounted Cumulative Gain)   
 
# Needed to compare rankings when the numbre of item compared are not the same
# and/or when relevance is not binary

def DCG(v, top):
    """
    V is vector of ranks, lowest is better
    top is the max rank considered 
    Relevance is 1 if items in rank vector, 0 else
    """
    
    discounted_gain = 0
    
    for i in np.round(v):
        if i <= top:
            discounted_gain += 1/np.log2(i+1)

    return round(discounted_gain, 2)


def nDCG(v, top, nb_values=0):
    """
    DCG normalized with what would be the best evaluation.
    nb_values is the max number of good values there is. If not specified, 
    assumed to be same as top.
    """
    assert nb_values <= top, "nb_values can't be larger than top value"
    if nb_values == 0: nb_values = top
    dcg = DCG(v, top)
    idcg = DCG(np.arange(nb_values)+1, top)
    
    return round(dcg/idcg, 2)
    

    
# RR (Reciprocal Rank)
    
# Gives a value in [0,1] for the first relevant item in list.
# 1st = 1 and than lower until cloe to 0.
# Only consern with FIRST relevant item in the list.
    
def RR(v):
    return 1/np.min(v)

    

def Ranks(all_values, values_to_rank, topx = 0):
    """
    Takes 2 numpy array and return, for all values in values_to_rank,
    the ranks, average ranks, MRR and nDCG for ranks smaller than topx
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # Initiate ranks
    ranks = np.zeros(len(values_to_rank))
    
    for i,v in enumerate(values_to_rank):
        ranks[i] = len(all_values[all_values >= v])
        
    ndcg = nDCG(ranks, topx, len(values_to_rank))
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), ndcg
    


    
def ChronoPlot(d1, d0, title=''):
    """
    Plot graph of 2 dict, doing mean of values
    """
    d1x = []
    d1y = []
    d1mean = []    # global mean
    d0x = []
    d0y = []
    d0mean = []
    
    for k, v in sorted(d1.items()):
        d1x.append(k)
        d1y.append(mean(v))
        d1mean += v
        
    for k, v in sorted(d0.items()):
        d0x.append(k)
        d0y.append(mean(v))
        d0mean += v

    plt.plot(d1x, d1y, label='with genres')
    plt.plot(d0x, d0y, label='without')
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.show()
    
    return mean(d1mean), mean(d0mean)
    
    
    
    
    
def EpochPlot(tup, title=''):
    """
    Plot graph of 2 dict, doing mean of values
    """
        
    y1 = [w1 for (w1, w0) in tup]
    y0 = [w0 for (w1, w0) in tup]

    print(y1)
    print(y0)
    
    plt.plot(y1, label='with genres')
    plt.plot(y0, label='without')
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






