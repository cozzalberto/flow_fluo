import abc
import io
import os
import re
from typing import AnyStr, ByteString, ClassVar, Dict, Generic, List, Literal, Tuple, TypeVar, Union
from typing_extensions import override
import dataclasses
import functools
from pathlib import Path
import warnings
import zipfile
import json
import tifffile
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as v2

from ddm4ip.data.utils import AddLocMapTransform
from ddm4ip.utils import distributed
from .base import Batch, DatasetType, Datasplit

IMG_EXTENSIONS = [
    '.tif', '.tiff'
]


def is_image_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, Path) and not filename.is_file():
        return False
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

class TiffDataset(torch.utils.data.Dataset[Batch], DatasetType):
    def __init__(
        self,
        path: str,
        degradation,
        split: Datasplit,
        dset_cfg,
        shuffle_clean: bool = True,
        generator=None,
    ):
        self.split = split
        self.corruption = degradation
        self.M = dset_cfg.get("M", 1)
        
        # 1. Caricamento dell'intero file TIFF in memoria
        # tifffile caricherà un array numpy di forma (500, 90, 90) o (500, Canali, 90, 90)
        frames_np = tifffile.imread(path)
        
        # Se l'immagine non ha il canale colore/grigio esplicito (es. è 500x90x90),
        # aggiungiamo la dimensione del canale: (500, 1, 90, 90)
        if frames_np.ndim == 3:
            frames_np = np.expand_dims(frames_np, axis=1)
            
        self.frames = torch.from_numpy(frames_np).to(torch.float32)
       
        self.frames = self.frames / self.M

        # 3. Inizializzazione attributi del protocollo DatasetType
        self.label_dim = 0 # Assumiamo che non ci siano etichette classiche associate
        self.clean_img_size = (self.frames.shape[1], self.frames.shape[2], self.frames.shape[3])
        self.clean_conditioning_channels = 0
        
        # Calcoliamo la dimensione dell'immagine corrotta con un sample
        sample_img = self.frames[0]
        sample_corrupt = self.add_corruption(sample_img, conditioning=None)
        self.corrupt_img_size = (sample_corrupt.shape[-3], sample_corrupt.shape[-2], sample_corrupt.shape[-1])
        self.corrupt_conditioning_channels = 0

        # 4. Gestione degli indici per lo shuffle (come in ZipFileDataset)
        if shuffle_clean:
            self.clean_ids = torch.randperm(len(self), generator=generator)
        else:
            self.clean_ids = torch.arange(len(self))

    def add_corruption(self, img, conditioning):
        if self.corruption is None:
            return img
        # Il modello di corruzione si aspetta spesso un batch dimension, 
        # quindi usiamo unsqueeze(0) e poi squeeze(0)
        corrupt_img = self.corruption(
            img.unsqueeze(0), conditioning=conditioning,
        ).squeeze(0).clamp(0, 1)
        return corrupt_img

    @property
    def noise_level(self) -> torch.Tensor:
        if self.corruption is None:
            return torch.tensor(0.0)
        try:
            return self.corruption.noise_model.sigma
        except AttributeError:
            return torch.tensor(0.0)

    def __len__(self):
        # La lunghezza del dataset è il numero di frame (500)
        return self.frames.shape[0]

    def __getitem__(self, index) -> Batch:
        # Recupera l'indice del frame pulito (con eventuale shuffle)
        clean_idx = int(self.clean_ids[index % len(self)])
        clean_img = self.frames[clean_idx]
        
        # L'immagine corrotta (se accoppiata al frame esatto, usa lo stesso index)
        noisy_idx = index % len(self)
        noisy_img = self.frames[noisy_idx]
        
        # Applica il modello di degradazione per ottenere l'input corrotto
        noisy_img = self.add_corruption(noisy_img, conditioning=None)

        # Restituisci l'oggetto Batch standard richiesto dal framework
        return Batch(
            clean=clean_img,
            corrupt=noisy_img,
            clean_label=None,
            corrupt_label=None,
            noise_level=self.noise_level,
            clean_conditioning=None,
            corrupt_conditioning=None,
        )