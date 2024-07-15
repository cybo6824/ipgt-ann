from collections import defaultdict
import pickle
import itertools
import numpy as np
import torch

from data.data_sampling import GroupSampling


class DataLoader:
    def __init__(
        self,
        batch_size=16,
        series_lengths=[35],
        train_size=1000,
        eval_size=100,
        pig_counts=[10],
        sample_rates=[1.0],
        shuffle=True,
        length_limit=-1,
        cv_fold_total=3,
        nested_fold_total=None,
        time_shift=False,
        pre_load=False,
        time_weight=0.0,
        use_feed=False,
        day_trim=-1,
        max_days=180,
        cluster_label=False,
    ):
        self.batch_size = batch_size
        self.train_size = train_size
        self.eval_size = eval_size
        self.time_shift = time_shift
        self.time_weight = time_weight
        self.use_feed = use_feed
        self.day_trim = day_trim
        self.max_days = max_days
        self.pre_load = pre_load
        self.cluster_label = cluster_label
        self.day_seconds = 86400

        self.cv_folds = list(range(cv_fold_total))
        if nested_fold_total == None:
            self.nested_folds = None
        else:
            self.nested_folds = list(range(nested_fold_total))
        self.shuffle = shuffle

        if self.use_feed:
            self.features_1d = 5
            self.features_2d = 8
        else:
            self.features_1d = 3
            self.features_2d = 4

        print("Pre-loading sequences...")
        with open("data/data_dump.dat", "rb") as fh:
            data_cache = pickle.load(fh)["pig_series"]
        with open("data/data_folds.dat", "rb") as fh:
            fold_sets = pickle.load(fh)[cv_fold_total]
        group_data = self.generate_sequences(
            data_cache,
            fold_sets,
            pig_counts,
            sample_rates,
            length_limit,
            max(train_size, eval_size),
            series_lengths,
        )
        self.dataset = self.generate_dataset(group_data)

        if nested_fold_total != None:
            self.nested_indices = self.get_nested_indices(
                cv_fold_total, nested_fold_total
            )
        self._np_rng = np.random.Generator(np.random.PCG64(666))

    def generate_sequences(
        self,
        data_cache,
        fold_sets,
        pig_counts,
        sample_rates,
        length_limit,
        series_count,
        series_lengths,
    ):
        seq_ids = list(
            itertools.product(
                pig_counts, sample_rates, series_lengths, list(range(series_count))
            )
        )
        group_data = list()
        fold_indices = list()
        weight_data = list()
        feed_quan_data = list()
        feed_dur_data = list()

        group_index = 0
        for fold_n in self.cv_folds:
            fold_indices_n = list()
            fold_set = fold_sets[fold_n]
            index_offset = 0
            for pig_count, sample_rate, series_length, group_id in seq_ids:
                while True:
                    current_id = group_id + index_offset
                    group_mat = GroupSampling(
                        data_cache,
                        group_id=current_id,
                        pig_count=pig_count,
                        sample_rate=sample_rate,
                        fold_set=fold_set,
                        series_length=series_length,
                    )
                    if length_limit == -1 or group_mat.shape[0] <= length_limit:
                        break
                    index_offset += 1
                weight_data.append(group_mat[:, 0])
                feed_quan_data.append(group_mat[:, 3])
                feed_dur_data.append(group_mat[:, 4])
                group_dict = {
                    "fold_n": fold_n,
                    "pig_count": pig_count,
                    "sample_rate": sample_rate,
                    "group_id": current_id,
                    "group_mat": group_mat,
                    "series_length": series_length,
                }
                group_data.append(group_dict)
                fold_indices_n.append(group_index)
                group_index += 1
            fold_indices_n = np.asarray(fold_indices_n, dtype=int)
            fold_indices.append(fold_indices_n)
        self.fold_indices = fold_indices

        weight_data = np.concatenate(weight_data)
        feed_quan_data = np.concatenate(feed_quan_data)
        feed_dur_data = np.concatenate(feed_dur_data)

        self.norm = dict()
        self.norm["weight_mean"] = np.mean(weight_data)
        self.norm["weight_std"] = np.std(weight_data)
        self.norm["feed_quan_mean"] = np.mean(feed_quan_data)
        self.norm["feed_quan_std"] = np.std(feed_quan_data)
        self.norm["feed_dur_mean"] = np.mean(feed_dur_data)
        self.norm["feed_dur_std"] = np.std(feed_dur_data)
        return group_data

    def get_nested_indices(self, cv_total, nested_total):
        rng = np.random.Generator(np.random.PCG64(555))
        cv_outer = list()
        for fold_n in range(cv_total):
            group_indices = list()
            for i in range(cv_total):
                if i != fold_n:
                    group_indices.append(self.fold_indices[i])
            group_indices = np.concatenate(group_indices)
            rng.shuffle(group_indices)
            chunk_size = group_indices.shape[0] // nested_total
            cv_inner = list()
            for n_fold_n in range(nested_total - 1):
                cv_inner.append(
                    group_indices[n_fold_n * chunk_size : (n_fold_n + 1) * chunk_size]
                )
            cv_inner.append(group_indices[(nested_total - 1) * chunk_size :])
            cv_outer.append(cv_inner)
        return cv_outer

    def pair_format(self, weight_data, time_data, feed_data=None):
        seq_size = weight_data.shape[0]
        weight_mat = np.expand_dims(weight_data, axis=1)
        weight_mat = np.broadcast_to(weight_mat, (seq_size, seq_size))
        time_mat = np.expand_dims(time_data, axis=1)
        time_mat = np.broadcast_to(time_mat, (seq_size, seq_size))
        time_mat = (time_mat.T - time_mat) / self.max_days
        time_mat_sin = np.sin(time_mat)
        time_mat_cos = np.cos(time_mat)
        time_mat_weight = 1.0 + (np.abs(time_mat) * self.time_weight)

        if self.use_feed:
            feed_quan_mat = np.expand_dims(feed_data[0], axis=1)
            feed_quan_mat = np.broadcast_to(feed_quan_mat, (seq_size, seq_size))
            feed_dur_mat = np.expand_dims(feed_data[1], axis=1)
            feed_dur_mat = np.broadcast_to(feed_dur_mat, (seq_size, seq_size))
            pair_mat = np.stack(
                (
                    weight_mat,
                    weight_mat.T,
                    feed_quan_mat,
                    feed_quan_mat.T,
                    feed_dur_mat,
                    feed_dur_mat.T,
                    time_mat_sin,
                    time_mat_cos,
                ),
                axis=2,
            )
        else:
            pair_mat = np.stack(
                (weight_mat, weight_mat.T, time_mat_sin, time_mat_cos), axis=2
            )
        return pair_mat, time_mat_weight

    def get_target_matrix(self, animal_indices):
        animal_ids = np.unique(animal_indices)
        target_matrix = np.full(
            (animal_indices.shape[0], animal_indices.shape[0]), -1.0, dtype=float
        )
        index_locations = dict()
        for animal_id in animal_ids:
            index_locations[animal_id] = animal_indices == animal_id
        for i in range(animal_indices.shape[0]):
            target_matrix[i, index_locations[animal_indices[i]]] = 1.0
        return target_matrix

    def get_target_labels(self, animal_indices):
        animal_ids = np.unique(animal_indices)
        target_labels = list()
        for animal_id in animal_ids:
            target_indices = np.nonzero(animal_indices == animal_id)[0]
            target_labels.append(torch.from_numpy(target_indices))
        return target_labels

    def get_target_indices(self, animal_indices):
        animal_ids = np.unique(animal_indices)
        indices = defaultdict(list)
        labels = list()
        for animal_id in animal_ids:
            labels.append(np.nonzero(animal_indices == animal_id)[0])
        for i in range(animal_ids.shape[0]):
            for j in range(i, animal_ids.shape[0]):
                x_indices, y_indices = np.meshgrid(labels[i], labels[j])
                x_indices = x_indices.flatten()
                y_indices = y_indices.flatten()
                if i == j:
                    indices["same_x"].append(x_indices)
                    indices["same_y"].append(y_indices)
                else:
                    indices["diff_x"].append(x_indices)
                    indices["diff_y"].append(y_indices)
        indices["same_x"] = torch.from_numpy(np.concatenate(indices["same_x"]))
        indices["same_y"] = torch.from_numpy(np.concatenate(indices["same_y"]))
        indices["diff_x"] = torch.from_numpy(np.concatenate(indices["diff_x"]))
        indices["diff_y"] = torch.from_numpy(np.concatenate(indices["diff_y"]))
        return indices

    def generate_dataset(self, group_data_all):
        dataset = list()
        for group_data in group_data_all:
            data_out = dict()
            data_out["features"] = dict()
            group_mat = group_data["group_mat"]

            if self.day_trim > 0 and self.day_trim < group_data["series_length"]:
                animal_count = np.unique(group_mat[:, 2]).shape[0]
                group_mat = group_mat[
                    group_mat[:, 1] < self.day_trim * self.day_seconds
                ]
                if np.unique(group_mat[:, 2]).shape[0] != animal_count:
                    dataset.append(None)
                    continue
                group_data["group_mat"] = group_mat

            weight_data = (group_mat[:, 0] - self.norm["weight_mean"]) / self.norm[
                "weight_std"
            ]
            time_data = group_mat[:, 1] / self.day_seconds

            if self.time_shift:
                low_limit = 180.0 + time_data[0]
                high_limit = 180.0 - time_data[-1]
                time_shift = self._np_rng.uniform(low=-high_limit, high=low_limit)
                time_data -= time_shift

            time_data_n = time_data / self.max_days
            time_data_sin = np.sin(time_data_n)
            time_data_cos = np.cos(time_data_n)

            if self.use_feed:
                feed_quan_data = (
                    group_mat[:, 3] - self.norm["feed_quan_mean"]
                ) / self.norm["feed_quan_std"]
                feed_dur_data = (
                    group_mat[:, 4] - self.norm["feed_dur_mean"]
                ) / self.norm["feed_dur_std"]
                x_1d = np.stack(
                    (
                        weight_data,
                        feed_quan_data,
                        feed_dur_data,
                        time_data_sin,
                        time_data_cos,
                    ),
                    axis=1,
                )
            else:
                x_1d = np.stack((weight_data, time_data_sin, time_data_cos), axis=1)
            data_out["x_1d"] = torch.from_numpy(x_1d).float()

            if self.cluster_label:
                data_out["y"] = self.get_target_indices(group_mat[:, 2])

            if self.pre_load:
                if self.use_feed:
                    x_2d, time_weights = self.pair_format(
                        weight_data,
                        time_data,
                        feed_data=(feed_quan_data, feed_dur_data),
                    )
                else:
                    x_2d, time_weights = self.pair_format(weight_data, time_data)
                data_out["x_2d"] = torch.from_numpy(x_2d).float()

                if not self.cluster_label:
                    y = self.get_target_matrix(group_mat[:, 2])
                    data_out["y"] = torch.from_numpy(y).float()

                if self.time_weight > 0.0:
                    data_out["y_w"] = torch.from_numpy(time_weights).float()
            else:
                data_out["features"]["weight"] = weight_data
                data_out["features"]["time"] = time_data

                if self.use_feed:
                    data_out["features"]["feed_quan"] = feed_quan_data
                    data_out["features"]["feed_dur"] = feed_dur_data

            data_out["data"] = group_data
            dataset.append(data_out)
        return dataset

    def get_feature_size(self):
        return self.features_1d, self.features_2d

    def set_dataset(self, dataset, cv_fold, nested_fold=None):
        self.batch_index = -1
        if dataset == "eval":
            if nested_fold == None:
                group_indices = self.fold_indices[cv_fold][: self.eval_size]
            else:
                group_indices = self.nested_indices[cv_fold][nested_fold][
                    : self.eval_size
                ]
        elif dataset == "train":
            if nested_fold == None:
                group_indices = list()
                for fold_n in self.cv_folds:
                    if fold_n != cv_fold:
                        group_indices.append(
                            self.fold_indices[fold_n][: self.train_size]
                        )
                group_indices = np.concatenate(group_indices)
            else:
                group_indices = list()
                for fold_n in self.nested_folds:
                    if fold_n != nested_fold:
                        group_indices.append(
                            self.nested_indices[cv_fold][fold_n][: self.train_size]
                        )
                group_indices = np.concatenate(group_indices)
        else:
            raise ValueError("set_dataset: dataset needs to equal 'train' or 'eval'")
        self.total_batches = np.ceil(group_indices.shape[0] / self.batch_size)
        if self.shuffle and dataset == "train":
            self._np_rng.shuffle(group_indices)
        self.current_indices = group_indices

    def __iter__(self):
        return self

    def __next__(self):
        item = None
        while item == None:
            self.batch_index += 1
            if self.batch_index >= self.total_batches:
                raise StopIteration
            item = self.create_batch()
        return item

    def create_batch(self):
        batch_start_index = self.batch_index * self.batch_size
        batch_end_index = (self.batch_index + 1) * self.batch_size
        batch_indices = list()
        for i in range(batch_start_index, batch_end_index):
            if i >= len(self.current_indices):
                break
            if self.dataset[self.current_indices[i]] != None:
                batch_indices.append(self.current_indices[i])
        if len(batch_indices) == 0:
            return None
        batch_lengths = [self.dataset[x]["x_1d"].size(0) for x in batch_indices]
        batch_size = len(batch_indices)
        max_length = max(batch_lengths)

        batch = dict()
        batch["x_1d"] = torch.zeros(max_length, batch_size, self.features_1d)
        batch["x_2d"] = torch.zeros(
            batch_size, max_length, max_length, self.features_2d
        )
        if self.cluster_label:
            batch["y"] = list()
        else:
            batch["y"] = torch.zeros(batch_size, max_length, max_length)
        batch["y_w"] = torch.zeros(batch_size, max_length, max_length)
        batch["pad_mask"] = torch.zeros(batch_size, max_length)
        batch["data"] = list()
        for i, group_index in enumerate(batch_indices):
            group_data = self.dataset[group_index]
            group_features = group_data["features"]
            group_mat = group_data["data"]["group_mat"]
            current_length = group_data["x_1d"].size(0)

            if self.pre_load:
                x_2d = group_data["x_2d"]
                y = group_data["y"]

                if self.time_weight > 0.0:
                    y_w = group_data["y_w"]
            else:
                if self.use_feed:
                    x_2d, time_weights = self.pair_format(
                        group_features["weight"],
                        group_features["time"],
                        feed_data=(
                            group_features["feed_quan"],
                            group_features["feed_dur"],
                        ),
                    )
                else:
                    x_2d, time_weights = self.pair_format(
                        group_features["weight"], group_features["time"]
                    )
                x_2d = torch.from_numpy(x_2d).float()

                if self.cluster_label:
                    y = group_data["y"]
                else:
                    y = self.get_target_matrix(group_mat[:, 2])
                    y = torch.from_numpy(y).float()

                if self.time_weight > 0.0:
                    y_w = torch.from_numpy(time_weights).float()

            if self.time_weight > 0.0:
                batch["y_w"][i, :current_length, :current_length] = y_w
            else:
                batch["y_w"][i, :current_length, :current_length] = 1.0

            batch["x_1d"][:current_length, i, :] = group_data["x_1d"]
            batch["x_2d"][i, :current_length, :current_length] = x_2d
            if self.cluster_label:
                batch["y"].append(y)
            else:
                batch["y"][i, :current_length, :current_length] = y
            batch["pad_mask"][i, current_length:] = float("-inf")
            batch["data"].append(group_data["data"])
        return batch
