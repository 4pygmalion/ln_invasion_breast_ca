import os
import numpy as np
import tensorflow as tf
from PIL import Image

def get_patches_path(patch_dir:str) -> list:
    """패치가 저장된 디렉토리를 받아서 이미지 원본소스 별: 패치경로를 반환"""

    patch_paths = list()
    for patch_name in os.listdir(patch_dir):
        patch_paths.append( os.path.join(patch_dir, patch_name))

    return patch_paths


def data_generate(bag_names:list, labels:list, bag_dirs:list) -> tuple:
    for bag_name, label, bag_dir in zip(bag_names, labels, bag_dirs):
        patch_paths = get_patches_path(bag_dir)

        bag_image = list()
        for patch_path in patch_paths:
            patch = np.asarray(Image.open(patch_path))
            bag_image.append(patch)
            
        bag_img_tensor = tf.convert_to_tensor(np.stack(bag_image, axis=0))
        bag_label_tensor = tf.convert_to_tensor(np.array(label).reshape(1, 1), dtype=tf.float32)

        yield bag_img_tensor, bag_label_tensor