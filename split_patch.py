import os
import argparse
import numpy as np
from PIL import Image

import cv2


def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", help="Input image folder", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output image folder", required=True
    )
    parser.add_argument(
        "-s", "--stride", help="Stride", required=False, default=128, type=int
    )
    parser.add_argument(
        "-w", "--patch_width", help="path width", required=False, default=512, type=int
    )
    parser.add_argument(
        "-c", "--fill_cutoff", help="Fill cutoff", default=0.8, type=float
    )
    return parser.parse_args()


def get_image_abs_path(directory: str) -> list:
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def is_background_patch(image_array: np.ndarray, fill_cutoff:float) -> bool:
    """이미지의 대부분이 백그라운드인지 확인하는 메서드"""
    h, w, c = image_array.shape

    n_pixels = h * w
    threshold, bin_img = cv2.threshold(
        image_array, 224, 255, cv2.THRESH_BINARY_INV
    )
    n_filled_pixels = len(np.where(bin_img.sum(axis=-1) != 0)[0])

    if n_filled_pixels / n_pixels <= fill_cutoff:
        return True

    return False


def extract_patch(image_path: str, stride: int, patch_width: int, fill_cutoff:float):
    patches = dict()
    image = np.array(Image.open(image_path))
    x_max, y_max, n_channel = image.shape

    x_stride_max = x_max - patch_width
    y_stride_max = y_max - patch_width
    for x_start in range(0, x_stride_max, stride):
        for y_start in range(0, y_stride_max, stride):
            patch_image = image[
                x_start : x_start + patch_width, 
                y_start : y_start + patch_width
            ]

            if patch_image.shape != (patch_width, patch_width, 3):
                continue

            if is_background_patch(patch_image, fill_cutoff):
                continue

            patches[f"{x_start}_{y_start}"] = patch_image

    return patches


if __name__ == "__main__":
    ARGS = get_args()
    input_dir = os.path.abspath(ARGS.input_folder)
    output_dir = os.path.abspath(ARGS.output)

    if not os.path.exists(ARGS.input_folder):
        print(f"{ARGS.input_folder} not found")
        exit()

    image_paths = get_image_abs_path(input_dir)
    for image_path in image_paths:
        print(f"In processing: {image_path}", end="\r")

        f_name = os.path.basename(image_path).rstrip(".png")
        image_dir = os.path.join(output_dir, f_name)
        os.makedirs(image_dir, exist_ok=True)

        patches = extract_patch(
            image_path, stride=ARGS.stride, patch_width=ARGS.patch_width, fill_cutoff=ARGS.fill_cutoff
        )
        for region_coordiante, patch_image in patches.items():
            img = Image.fromarray(patch_image)
            img.save(os.path.join(image_dir, region_coordiante) + ".png")
