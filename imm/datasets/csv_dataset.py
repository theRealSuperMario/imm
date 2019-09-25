from __future__ import division

import os
import numpy as np
import tensorflow as tf

from imm.datasets.impair_dataset import ImagePairDataset
import pandas as pd
import random


def add_choices(labels, return_by_cid=False, character_id_key="character_id"):
    labels = dict(labels)
    cid_labels = np.asarray(labels[character_id_key])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                print("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels[character_id_key])):
        cid = labels[character_id_key][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    if return_by_cid:
        return labels, cid_indices
    return labels


def load_dataset(data_root, data_csv, id_col_name="id", fname_col_name="im1"):
    data_frame = pd.read_csv(data_csv)
    data_frame[fname_col_name] = data_frame[fname_col_name].apply(
        lambda x: os.path.join(data_root, x)
    )
    labels = dict(data_frame)
    labels = {k: list(v) for k, v in labels.items()}
    labels = add_choices(labels, character_id_key=id_col_name)
    return labels


class CSVDataset(ImagePairDataset):
    def __init__(
        self,
        data_dir,
        data_csv,
        id_col_name,
        fname_col_name,
        subset,
        dataset=None,
        order_stream=False,
        image_size=[128, 128],
        max_samples=None,
        jittering=None,
        augmentations=["flip", "swap"],
        name="CSVDataset",
    ):
        super(CSVDataset, self).__init__(
            data_dir,
            subset,
            image_size=image_size,
            augmentations=augmentations,
            jittering=jittering,
            name=name,
        )
        assert dataset is not None
        self.labels = load_dataset(data_dir, data_csv, id_col_name, fname_col_name)
        self.fname_col_name = fname_col_name
        self.id_col_name = id_col_name
        self._length = len(list(self.labels.items())[0])
        if max_samples is None:
            self._max_samples = self._length  #  TODO: not sure if this is a good idea
        else:
            self._max_samples = max_samples
        self._order_stream = order_stream

    def __len__(self):
        return self._length

    def _get_random_pair(self):
        i = np.random.randint(len(self))
        choices = self.labels["choices"][i]
        j = random.choice(choices)
        view0 = self.labels[self.fname_col_name][i]
        view1 = self.labels[self.fname_col_name][j]
        pair = {"image": view0, "future_image": view1}
        return pair

    def _get_ordered_stream(self):
        for i in range(len(self)):
            view0 = self.labels[self.fname_col_name][i]
            view1 = view0  # TODO: this is a dirty hack because I do not know it better
            yield {"image": view0, "future_image": view1}

    def _get_sample_shape(self):
        return {k: None for k in self._get_sample_dtype().keys()}

    def _get_sample_dtype(self):
        return {"image": tf.string, "future_image": tf.string}

    def sample_image_pair(self):
        f_sample = self._get_random_pair
        if self._order_stream:
            g = self._get_ordered_stream()
            f_sample = lambda: next(g)
        max_samples = float("inf")
        if self._max_samples is not None:
            max_samples = self._max_samples
        i_samp = 0
        while i_samp < max_samples:
            yield f_sample()
            if self._max_samples is not None:
                i_samp += 1

    def num_samples(self):
        return self._length


if __name__ == "__main__":
    dset = CSVDataset(
        "/home/sandro/Projekte/gitlab_projects/2019_ma_hd/ma_code/nips19/baselines/jakab18/imm/data/datasets/exercise_dataset/",
        "/home/sandro/Projekte/gitlab_projects/2019_ma_hd/ma_code/nips19/baselines/jakab18/imm/data/datasets/exercise_dataset/csvs/instance_level_train_split.csv",
        id_col_name="id",
        fname_col_name="im1",
        subset="train",
    )
    image_pair = next(dset.sample_image_pair())
