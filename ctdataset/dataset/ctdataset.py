import os
from torch.utils.data import Dataset
import random
from nmesh import NMesh
from ctdataset.functional import *
import numpy as np
import logging
from gnutools.fs import name, parent
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class CTDataset(Dataset):
    def __init__(self,
                 ply_folder,
                 xpath=None,
                 exploration_ratio="1.0",
                 bbox_w = 24,
                 dim=64,
                 strech_box=False):
        self.strech_box = strech_box
        self.cache = {}
        if ply_folder is not None:
            ids = [(
                f"{ply_folder}/{dir}/x.ply",
                f"{ply_folder}/{dir}/y.ply"
            ) for dir in os.listdir(ply_folder) if os.path.isdir(f"{ply_folder}/{dir}")]
            kwargs = {
                "dim": dim,
                "bbox_w": bbox_w,
                "strech_box": strech_box
            }
            self.load_function = self.load_ply
            with ProcessPoolExecutor() as e:
                fs = [e.submit(self.load_ply_file, x_path, y_path, **kwargs) for (x_path, y_path) in ids]
                for f in tqdm(as_completed(fs), total=len(fs), desc="Caching"):
                    ID, x_vertices, y_vertices, ops = f._result
                    self.cache[ID] = (x_vertices, y_vertices, ops)
        else:
            ids = [(xpath, xpath)]
            self.load_function = self.load_ply

        self.ids = ids
        self.bbox_w = bbox_w
        self.dim = dim
        self.size = int(len(ids)*eval(exploration_ratio)) if type(eval(exploration_ratio))==float else \
            int(eval(exploration_ratio)(len(ids)))

    @staticmethod
    def load_ply_file(x_path, y_path, dim, bbox_w, strech_box):
        ID = name(parent(x_path))
        ops = []
        if not os.path.exists(y_path):
            y_path = x_path
        x, y = NMesh(x_path), NMesh(y_path)
        x_vertices, y_vertices = x.vertices, y.vertices
        T = np.min(np.array([np.min(x_vertices, axis=0), np.min(y_vertices, axis=0)]), axis=0)
        x_vertices, y_vertices = x_vertices-T, y_vertices-T,

        scale2 = bbox_w if not strech_box else np.max(x_vertices, axis=0)
        x_vertices, y_vertices = x_vertices/scale2, y_vertices/scale2

        logging.warning(f"Streching {np.max(x_vertices, axis=0)}") if strech_box else None

        x_vertices, y_vertices = np.array(x_vertices*(dim-1), dtype=int), np.array(y_vertices*(dim-1), dtype=int)
        ops.append(("scale", 1/(dim-1)))
        ops.append(("scale", scale2))
        ops.append(("translation", T))
        x_vertices, y_vertices = np.unique(x_vertices, axis=0),np.unique(y_vertices, axis=0)
        return ID, x_vertices, y_vertices, ops

    def load_ply(self, index):
        x_path, y_path = self.ids[index]
        ID = name(parent(x_path))
        try:
            (x_vertices, y_vertices, ops) = self.cache[ID]
            return self.cp2matrix(x_vertices, dim=self.dim), \
                   self.cp2matrix(y_vertices, dim=self.dim), \
                   ops, \
                   name(parent(x_path)),\
                   x_path
        except AssertionError as e:
            print(x_path)
            raise AssertionError

    def __getitem__(self, index):
        return self.load_function(index)

    def __len__(self):
        return self.size

    def shuffle(self):
        ids = list(range(len(self.ids)))
        random.shuffle(ids)
        self.ids = list(np.array(self.ids)[ids])

    @staticmethod
    def cp2matrix(cp, dim=64):
        return cp2matrix(cp, dim)

    @staticmethod
    def matrix2cp(M):
        return matrix2cp(M)

