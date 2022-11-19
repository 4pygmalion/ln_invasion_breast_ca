import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from model import build_model
from load_data import get_patches_path, data_generate
from loss import bag_loss, bag_accuracy

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_patch_folder",
        help="Input patch image folder",
        required="data/train_imgs_patch",
    )
    parser.add_argument(
        "-w",
        "--patch_width",
        help="path width",
        required=False,
        default=256,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = get_args()
    PATCH_WIDTH = ARGS.patch_width
    PATCH_SHAPE = (PATCH_WIDTH, PATCH_WIDTH, 3)
    BATCH_SIZE = 1

    clinical_data = pd.read_csv("data/train.csv")
    bag_names = list(clinical_data["ID"])
    labels = list(clinical_data["N_category"])
    patch_bags = get_patches_path(ARGS.input_patch_folder)

    (
        train_bag_names,
        val_bag_names,
        train_y,
        val_y,
        train_bags,
        val_bags,
    ) = train_test_split(
        bag_names[: len(patch_bags)], labels[: len(patch_bags)], patch_bags
    )

    train_dataset = tf.data.Dataset.from_generator(
        generator=data_generate,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, PATCH_WIDTH, PATCH_WIDTH, 3]),
            tf.TensorShape([1, 1]),
        ),
        args=(train_bag_names, train_y, train_bags),
    )

    val_dataset = tf.data.Dataset.from_generator(
        generator=data_generate,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, PATCH_WIDTH, PATCH_WIDTH, 3]),
            tf.TensorShape([1, 1]),
        ),
        args=(val_bag_names, val_y, val_bags),
    )

    model = build_model(PATCH_SHAPE)
    model.summary()

    os.makedirs("check_points", exist_ok=True)
    model_name = (
        "check_points/"
        + "acc({accuracy:.4f})"
        + "epoch({epoch})"
        + "val_loss({val_loss:.4f}).hd5"
    )
    check_point = tf.keras.callbacks.ModelCheckpoint(
        model_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # (=decay)
        verbose=True,
    )
    callbacks = [check_point, early_stopping, reduce_lr]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset.repeat(),
        validation_data=val_dataset.repeat(),
        callbacks=callbacks,
        epochs=100,
        steps_per_epoch=int(len(train_bag_names) / BATCH_SIZE),
        validation_steps=int(len(val_bag_names) / BATCH_SIZE),
    )
