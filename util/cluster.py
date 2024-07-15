from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as skm
from scipy.optimize import linear_sum_assignment

import util.plot as pd_plt


class PidentCluster:
    def __init__(self):
        self.interp_freq = 3600

    def cluster_score(self, dist_mat, group_mat, pig_count, plot_mode=False):
        clustering = self.h_cluster(dist_mat, n_clusters=pig_count, linkage="complete")
        true_series, pred_series = self.get_pred_series(
            group_mat, clustering.labels_, pig_count
        )
        assign_score, assign_indices = self.assignment_match(
            true_series, pred_series, pig_count
        )
        assign_indices["clusters"] = clustering.labels_

        if plot_mode:
            true_to_pred = dict(zip(assign_indices["true"], assign_indices["pred"]))
            # pd_plt.plot_true_pred(group_mat, clustering.labels_, pig_count)
            # pd_plt.plot_comparison_combo(true_to_pred, true_series, pred_series)
            # pd_plt.plot_series_basic(group_mat)
            # pd_plt.plot_cluster_map(dist_mat)
            # pd_plt.plot_bw_comparison(true_to_pred, true_series, pred_series, assign_indices["all_scores"])
            pd_plt.plot_full_combo(
                dist_mat,
                clustering.labels_,
                group_mat[:, 2],
                true_to_pred,
                true_series,
                pred_series,
            )
        return assign_score, assign_indices

    def h_cluster(self, dist_mat, n_clusters=50, linkage="complete"):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage=linkage
        )
        clustering.fit(dist_mat)
        return clustering

    def get_pred_series(self, group_mat, pred_labels, pig_count=50):
        animal_ids = np.unique(group_mat[:, 2])

        true_series = list()
        pred_series = list()
        for i in range(pig_count):
            pred_series_i = group_mat[pred_labels == i, :2]
            if pred_series_i.ndim == 1:
                pred_series_i = np.expand_dims(pred_series_i, 0)
            pred_series_i = pred_series_i[np.argsort(pred_series_i[:, 1])]
            pred_series.append(pred_series_i)

            true_series_i = group_mat[group_mat[:, 2] == animal_ids[i]]
            if true_series_i.ndim == 1:
                true_series_i = np.expand_dims(true_series_i, 0)
            true_series_i = true_series_i[np.argsort(true_series_i[:, 1])]
            true_series.append(true_series_i)
        return true_series, pred_series

    def assignment_match(self, true_series, pred_series, pig_count):
        diff_mat = np.zeros((pig_count, pig_count), dtype=float)
        for j in range(pig_count):
            for i in range(pig_count):
                if pred_series[j].shape[0] == 1 and true_series[i].shape[0] != 1:
                    if (
                        pred_series[j][0, 1] >= true_series[i][0, 1]
                        and pred_series[j][0, 1] <= true_series[i][-1, 1]
                    ):
                        true_interp = np.interp(
                            pred_series[j][0, 1],
                            true_series[i][:, 1],
                            true_series[i][:, 0],
                        )
                        distance = pred_series[i][0, 1] - true_interp
                        diff_mat[i, j] = np.sqrt(distance * distance)
                    else:
                        diff_mat[i, j] = np.Inf
                elif true_series[i].shape[0] == 1 and pred_series[j].shape[0] != 1:
                    if (
                        true_series[i][0, 1] >= pred_series[j][0, 1]
                        and true_series[i][0, 1] <= pred_series[j][-1, 1]
                    ):
                        pred_interp = np.interp(
                            true_series[i][0, 1],
                            pred_series[j][:, 1],
                            pred_series[j][:, 0],
                        )
                        distance = pred_series[i][0, 1] - pred_interp
                        diff_mat[i, j] = np.sqrt(distance * distance)
                    else:
                        diff_mat[i, j] = np.Inf
                elif true_series[i].shape[0] == 1 and pred_series[j].shape[0] == 1:
                    if true_series[i][0, 1] == pred_series[j][0, 1]:
                        distance = pred_series[j][0, 0] - true_series[i][0, 0]
                        diff_mat[i, j] = np.sqrt(distance * distance)
                    else:
                        diff_mat[i, j] = np.Inf
                else:
                    overlap_start = max(pred_series[j][0, 1], true_series[i][0, 1])
                    overlap_end = min(pred_series[j][-1, 1], true_series[i][-1, 1])
                    interp_times = np.arange(
                        overlap_start, overlap_end, self.interp_freq
                    )
                    if interp_times.shape[0] == 0:
                        diff_mat[i, j] = np.Inf
                    else:
                        pred_interp = np.interp(
                            interp_times, pred_series[j][:, 1], pred_series[j][:, 0]
                        )
                        true_interp = np.interp(
                            interp_times, true_series[i][:, 1], true_series[i][:, 0]
                        )
                        distances = pred_interp - true_interp
                        diff_mat[i, j] = np.sqrt(np.mean(distances * distances))

        miss_count = 0
        try:
            true_ind, pred_ind = linear_sum_assignment(diff_mat)
        except ValueError:
            miss_count = 1

        if miss_count > 0:
            inf_counts = np.sum(diff_mat == np.Inf, axis=0)
            inf_indices = np.flip(np.argsort(inf_counts))
            while miss_count < len(true_series):
                diff_mat_r = np.delete(diff_mat, inf_indices[:miss_count], axis=1)
                try:
                    true_ind, pred_ind = linear_sum_assignment(diff_mat_r)
                    break
                except ValueError:
                    miss_count += 1

            true_miss = set(range(pig_count)) - set(true_ind)
            pred_miss = set(range(pig_count)) - set(pred_ind)
            for true_miss_i, pred_miss_i in zip(true_miss, pred_miss):
                true_ind = np.insert(true_ind, true_miss_i, true_miss_i)
                pred_ind = np.insert(pred_ind, true_miss_i, pred_miss_i)

        true_to_pred = dict(zip(true_ind, pred_ind))
        final_score = dict()
        final_score["rmse"] = np.mean(diff_mat[true_ind, pred_ind]) / 1000
        final_score.update(self.score(true_to_pred, true_series, pred_series))
        pred_indices = {
            "score": final_score,
            "pred": pred_ind,
            "true": true_ind,
            "miss_count": miss_count,
            "all_scores": diff_mat[true_ind, pred_ind],
        }
        if miss_count > 0:
            print("Warning: miss_count", miss_count)

        return final_score, pred_indices

    def score(self, true_to_pred, true_series, pred_series):
        scores = defaultdict(list)
        for true_i, pred_i in true_to_pred.items():
            true_time = true_series[true_i][:, 1]
            pred_time = pred_series[pred_i][:, 1]

            overlap_start = max(pred_time[0], true_time[0])
            overlap_end = min(pred_time[-1], true_time[-1])
            min_start = min(pred_time[0], true_time[0])
            max_end = max(pred_time[-1], true_time[-1])

            if max_end - min_start != 0:
                jaccard_score = (overlap_end - overlap_start) / (max_end - min_start)
            else:
                jaccard_score = 0.0
            scores["jaccard"].append(jaccard_score)

            interp_times = np.arange(overlap_start, overlap_end, self.interp_freq)
            if interp_times.shape[0] > 1:
                pred_interp = np.interp(
                    interp_times, pred_time, pred_series[pred_i][:, 0]
                )
                true_interp = np.interp(
                    interp_times, true_time, true_series[true_i][:, 0]
                )
                r2_score = max(skm.r2_score(true_interp, pred_interp), 0.0)
            else:
                r2_score = 0.0
            scores["r2"].append(r2_score)

            if r2_score + jaccard_score != 0.0:
                f1_score = 2.0 * (r2_score * jaccard_score) / (r2_score + jaccard_score)
                f05_score = (
                    (1.0 + 0.5 * 0.5)
                    * (r2_score * jaccard_score)
                    / (0.5 * 0.5 * r2_score + jaccard_score)
                )
                f2_score = (
                    (1.0 + 2.0 * 2.0)
                    * (r2_score * jaccard_score)
                    / (2.0 * 2.0 * r2_score + jaccard_score)
                )
            else:
                f1_score, f05_score, f2_score = 0.0, 0.0, 0.0
            scores["f1"].append(f1_score)
            scores["f05"].append(f05_score)
            scores["f2"].append(f2_score)
        return dict(zip(list(scores.keys()), [np.mean(x) for x in scores.values()]))
