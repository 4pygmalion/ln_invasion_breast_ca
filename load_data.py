import os
import re
import tensorflow as tf
from collections import defaultdict


import numpy as np
from matplotlib import pyplot as plt

def get_patches(patch_dir:str) -> list:
    """패치가 저장된 디렉토리를 받아서 이미지 원본소스 별: 패치경로를 반환"""

    patch_paths = list()
    for patch_name in os.listdir(patch_dir):
        patch_paths.append( os.path.join(patch_dir, patch_name))

    return patch_paths


def data_generate(bag_names:list, labels:list, bag_dirs:list) -> tuple:

    while True:
        for bag_name, label, bag_dir in zip(bag_names, labels, bag_dirs):
            patch_paths = get_patches(bag_dir)

            for patch in patch_paths:
                yield tf.convert_to_tensor(np.load(patch), dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)
            # patch_imgs = np.stack([np.load(patch) for patch in patch_paths], axis=0) 

            # yield tf.convert_to_tensor(patch_imgs, dtype=tf.float64), tf.convert_to_tensor(label, dtype=tf.int64)