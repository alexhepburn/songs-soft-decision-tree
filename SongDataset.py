import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

"""
For use with the raw chromagram taken from the One Million Songs Dataset
"""
class SongDataset(Dataset):

    def __init__(self, transform=None):
        self.transform=transform
        feature_names = ['danceability', 'energy', 'key', 'loudness', \
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']  
        chroma = {}
        chromastd = {}
        for i in range(0,12):
            feature_names.append('chroma' + str(i))
            feature_names.append('chroma' + str(i) + 'std')
            chroma[i] = []
            chromastd[i] = []
        df = pd.read_hdf('./df_magd_with_chroma_partition.h5')
        class_dict = {}
        num_dict = {}
        i = 0
        for name in df['genre'].unique():
            class_dict[name] = i
            num_dict[i] = name
            i += 1
        df['labels'] = ([class_dict[genre] for genre in df['genre']])
        self.chromas = df['chromas']
        chromas=df['chromas']
        for c in chromas:
            for i in range(0, 12):
                chroma[i].append(c[0][i])
                chromastd[i].append(c[1][i])
        for i in range(0, 12):
            df['chroma' + str(i)] = chroma[i]
            df['chroma' + str(i) + 'std'] = chromastd[i]
        self.X_train = df[feature_names].as_matrix().astype(np.float32)
        genre = pd.Series(df['labels']).astype('category')
        self.Y_train = np.asarray(genre.cat.codes, dtype=np.int64)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_train[index, :]), (self.Y_train[index])

    def __len__(self):
        return (len(self.chromas.index))

"""
To use with the features taken from the One Million Songs Dataset including the mean and std
for each column in the chromagram.
"""
class SongDataset2(Dataset):

    def __init__(self, transform=None):
        self.transform=transform
        feature_names = ['danceability', 'energy', 'key', 'loudness', \
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']  
        chroma = {}
        chromastd = {}
        for i in range(0,12):
            feature_names.append('chroma' + str(i))
            feature_names.append('chroma' + str(i) + 'std')
            chroma[i] = []
            chromastd[i] = []
        df = pd.read_hdf('./df_magd_with_chroma_partition.h5')
        class_dict = {}
        num_dict = {}
        i = 0
        for name in df['genre'].unique():
            class_dict[name] = i
            num_dict[i] = name
            i += 1
        df['labels'] = ([class_dict[genre] for genre in df['genre']])
        self.chromas = df['chromas']
        chromas=df['chromas']
        for c in chromas:
            for i in range(0, 12):
                chroma[i].append(c[0][i])
                chromastd[i].append(c[1][i])
        for i in range(0, 12):
            df['chroma' + str(i)] = chroma[i]
            df['chroma' + str(i) + 'std'] = chromastd[i]
        self.X_train = df[feature_names].as_matrix().astype(np.float32)
        genre = pd.Series(df['labels']).astype('category')
        self.Y_train = np.asarray(genre.cat.codes, dtype=np.int64)

    def __getitem__(self, index):
        return torch.from_numpy(self.X_train[index, :]), (self.Y_train[index])

    def __len__(self):
        return (len(self.chromas.index))
