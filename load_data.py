import os
import re
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_patches(patch_dir:str) -> dict:
    """패치가 저장된 디렉토리를 받아서 이미지 원본소스 별: 패치경로를 반환"""

    patch_path = defaultdict(list)

    original_image_expression = re.compile("BC\_\d+\_\d+_\d+")
    for patch_name in os.listdir(patch_dir):
        original_image_name = original_image_expression.search(patch_name).group(0)
        patch_path[original_image_name].append(os.path.join(patch_dir, patch_name))

    return patch_path




def get_multiple_xs(index:int, clinical_array:np.ndarray, all_image_paths:dict) -> tuple:
    id, image_path, label = clinical_array[index]
    
    
    
    return  clinical_array[index], all_image_paths[id]