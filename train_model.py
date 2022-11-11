import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from model import build_model
from load_data import get_patches, data_generate
from loss import bag_loss, bag_accuracy

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

PATCH_SHAPE = (224, 224, 3)

if __name__ == "__main__":
    clinical_data = pd.read_csv("data/train.csv")
    bag_names = list(clinical_data["ID"])
    labels = list(clinical_data["N_category"])
    patch_bags = get_patches("data/train_imgs_patch")

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
            tf.TensorShape([None, 224, 224, 3]),
            tf.TensorShape(
                [
                    1,
                ]
            ),
        ),
        args=(train_bag_names, train_y, train_bags),
    )

    val_dataset = tf.data.Dataset.from_generator(
        generator=data_generate,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, 224, 224, 3]),
            tf.TensorShape(
                [
                    1,
                ]
            ),
        ),
        args=(val_bag_names, val_y, val_bags),
    )

    bag_img_tensor, bag_label_tensor = next(iter(train_dataset))

    model = build_model(PATCH_SHAPE)
    model.summary()

    model_name = "Saved_model/" + "_Batch_size_" + "epoch_" + "best.hd5"
    checkpoint_fixed_name = tf.keras.callbacks.ModelCheckpoint(
        model_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        period=1,
    )

    EarlyStop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    callbacks = [checkpoint_fixed_name, EarlyStop]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bag_loss,
        metrics=[bag_accuracy],
    )

    model.fit(
        train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=10
    )
