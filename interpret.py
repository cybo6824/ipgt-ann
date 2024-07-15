import sys
from collections import defaultdict
import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import captum.attr as cattr
import scipy.stats as sps

from util.data_loader import DataLoader
from model import pident_models
from util.cluster import PidentCluster

DAY_SECONDS = 86400


def get_baselines(x_1d, x_2d, baseline, torch_device):
    if baseline == "zero":
        baseline_1d = torch.zeros(x_1d.shape, device=torch_device)
        baseline_2d = torch.zeros(x_2d.shape, device=torch_device)
    elif baseline == "zero-c":
        baseline_1d = torch.zeros(x_1d.shape, device=torch_device)
        baseline_1d[0, :, 2] = 1.0
        baseline_2d = torch.zeros(x_2d.shape, device=torch_device)
        baseline_2d[0, :, :, 3] = 1.0
    elif baseline == "zero-w":
        baseline_1d = torch.clone(x_1d)
        baseline_1d[0, :, 0] = 0.0
        baseline_2d = torch.clone(x_2d)
        baseline_2d[0, :, :, 0] = 0.0
        baseline_2d[0, :, :, 1] = 0.0
    elif baseline == "random" or baseline == "random-f" or baseline == "random-t":
        baseline_1d = torch.clone(x_1d)
        baseline_2d = torch.clone(x_2d)

        if baseline != "random-t":
            weight_rand = torch.randint(
                int(torch.min(x_1d[0, :, 0]).item() * 10000),
                int(torch.max(x_1d[0, :, 0]).item() * 10000),
                (x_1d.shape[1],),
            )
            weight_rand = weight_rand.float() / 10000
            baseline_1d[0, :, 0] = weight_rand
            weight_rand_2d = weight_rand.unsqueeze(dim=1).expand(
                (x_1d.shape[1], x_1d.shape[1])
            )
            baseline_2d[0, :, :, 0] = weight_rand_2d
            baseline_2d[0, :, :, 1] = torch.transpose(weight_rand_2d, 0, 1)

        if baseline == "random-f" or baseline == "random-t":
            time_rand = torch.randint(0, 35 * 10000, (x_1d.shape[1],)).float() / (
                10000 * 180
            )
            baseline_1d[0, :, 1] = torch.sin(time_rand)
            baseline_1d[0, :, 2] = torch.cos(time_rand)
            time_rand_2d = time_rand.unsqueeze(dim=1).expand(
                (x_1d.shape[1], x_1d.shape[1])
            )
            time_rand_2d = torch.transpose(time_rand_2d, 0, 1) - time_rand_2d
            baseline_2d[0, :, :, 2] = torch.sin(time_rand_2d)
            baseline_2d[0, :, :, 3] = torch.cos(time_rand_2d)

    elif baseline == "random-s":
        perm_indices = torch.randperm(x_1d.shape[1])
        baseline_1d = torch.clone(x_1d)
        baseline_1d[0, :, 0] = baseline_1d[0, perm_indices, 0]
        baseline_2d = torch.clone(x_2d)
        baseline_2d[0, :, :, 0] = baseline_2d[0, perm_indices, perm_indices, 0]
        baseline_2d[0, :, :, 1] = baseline_2d[0, perm_indices, perm_indices, 1]
    return baseline_1d, baseline_2d


def model_pred(
    model_id,
    fold_n,
    data_loader,
    model,
    torch_device,
    baseline="zero",
    method="ig",
    group_index=0,
):
    data_loader.set_dataset("eval", fold_n)

    if method == "ig":
        ca = cattr.IntegratedGradients(model)
    elif method == "dl":
        ca = cattr.DeepLift(model)
    elif method == "svs":
        ca = cattr.ShapleyValueSampling(model)
    elif method == "gs":
        ca = cattr.GradientShap(model)
    elif method == "oc":
        ca = cattr.Occlusion(model)

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if batch_index != group_index:
                continue

            x_1d = batch["x_1d"].to(torch_device)
            x_1d = x_1d.transpose(0, 1)
            x_2d = batch["x_2d"].to(torch_device)
            pad_mask = batch["pad_mask"].to(torch_device)
            batch_data = batch["data"]

            net_output = model(x_1d, x_2d, pad_mask).squeeze()
            net_output = (net_output + net_output.transpose(-1, -2)) / 2.0
            net_output = F.normalize(net_output, dim=(-1, -2))
            net_output -= torch.min(net_output)
            net_output = net_output.detach().cpu().numpy()

            with open(
                "metrics/interpret_attr_"
                + model_id
                + "_"
                + str(group_index)
                + "_pred.dat",
                "wb",
            ) as fh:
                pickle.dump(net_output, fh)

            attr_1d = torch.zeros(
                (x_1d.shape[1], x_1d.shape[1], x_1d.shape[1], x_1d.shape[2]),
                device=torch_device,
            )
            baseline_values = get_baselines(x_1d, x_2d, baseline, torch_device)

            if method == "ig" or method == "dl":
                delta = torch.zeros((x_1d.shape[1], x_1d.shape[1]), device=torch_device)
            else:
                delta = None

            for i in range(x_1d.shape[1]):
                for j in range(x_1d.shape[1]):
                    print(i, j)

                    if method == "ig" or method == "dl":
                        attribution, delta_i = ca.attribute(
                            (x_1d, x_2d),
                            target=(i, j),
                            additional_forward_args=pad_mask,
                            return_convergence_delta=True,
                            baselines=baseline_values,
                        )
                    elif method == "oc":
                        attribution = ca.attribute(
                            (x_1d, x_2d),
                            ((1, 3), (1, 1, 4)),
                            target=(0, 0),
                            baselines=baseline_values,
                        )
                    else:
                        attribution = ca.attribute(
                            (x_1d, x_2d),
                            target=(i, j),
                            additional_forward_args=pad_mask,
                            baselines=baseline_values,
                        )
                    attr_1d[i, j] = attribution[0][0].to(torch_device)
                    if method == "ig" or method == "dl":
                        delta[i, j] = delta_i[0].to(torch_device)

            attr_1d = attr_1d.cpu().numpy()
            if method == "ig" or method == "dl":
                delta = delta.cpu().numpy()

            attr_out = dict(
                {
                    "attr_1d": attr_1d,
                    "attr_2d": None,
                    "delta": delta,
                    "group_mat": batch_data[0]["group_mat"],
                    "method": method,
                    "baseline": baseline,
                }
            )
            file_code = (
                method + "_" + model_id + "_" + baseline + "_" + str(group_index)
            )
            with open("metrics/interpret_attr_" + file_code + ".dat", "wb") as fh:
                pickle.dump(attr_out, fh)
            sys.exit(0)


def load_attr(model_id, baseline, method, group_id, plot_type, absolute):
    file_code = method + "_" + model_id + "_" + baseline + "_" + str(group_id)
    with open("metrics/interpret_attr_" + file_code + ".dat", "rb") as fh:
        attr_data = pickle.load(fh)
    with open(
        "metrics/interpret_attr_" + model_id + "_" + str(group_id) + "_pred.dat", "rb"
    ) as fh:
        pred = pickle.load(fh)

    if plot_type == "pair":
        n_clusters = np.unique(attr_data["group_mat"][:, 2]).shape[0]
        _, assign_indices = PidentCluster().cluster_score(
            pred, attr_data["group_mat"], pig_count=n_clusters
        )
        PairAttrPlot(
            attr_data["group_mat"], attr_data["attr_1d"], assign_indices["clusters"]
        )
    elif plot_type == "corr":
        plot_attr_corr(
            attr_data["group_mat"],
            attr_data["attr_1d"],
            pred,
            absolute=absolute,
            use_pred=False,
        )
    elif plot_type == "pred-corr":
        plot_attr_corr(
            attr_data["group_mat"],
            attr_data["attr_1d"],
            pred,
            absolute=absolute,
            use_pred=True,
        )
    elif plot_type == "feature":
        plot_attr_feature(attr_data["group_mat"], attr_data["attr_1d"])
    elif plot_type == "dist":
        plot_attr_dist(
            attr_data["group_mat"], attr_data["attr_1d"], pred, absolute=absolute
        )


def plot_attr_iter(group_mat, attr_1d, attr_2d, delta):
    plt_x = group_mat[:, 1] / DAY_SECONDS
    plt_y = group_mat[:, 0] / 1000
    attr_1d = np.mean(attr_1d, axis=-1)

    for i in range(group_mat.shape[0]):
        attr_1d_i = np.copy(attr_1d[i])
        attr_1d_i -= np.amin(attr_1d)
        attr_1d_i /= np.amax(attr_1d)

        fig = plt.figure(figsize=(24, 12))
        plt.scatter(plt_x, plt_y, c=attr_1d_i, s=10)
        plt.scatter(plt_x[i], plt_y[i], c="black", marker="x")
        plt.xlabel("Time (days)")
        plt.ylabel("Weight (Kg)")
        plt.tight_layout()
        plt.show()


class PairAttrPlot:
    def __init__(self, group_mat, attr_1d, clusters, absolute=True):
        self.fig = plt.figure(figsize=(8, 5))
        # self.fig = plt.figure(figsize=(12,7))
        self.plt_x = group_mat[:, 1] / DAY_SECONDS
        self.plt_y = group_mat[:, 0] / 1000
        self.cm = mpl.cm.get_cmap("viridis")
        self.clustering = clusters
        self.prev_index = 0

        attr_1d = np.mean(attr_1d, axis=-1)
        if absolute:
            self.attr_1d = np.abs(attr_1d)
        else:
            self.attr_1d = attr_1d

        self.plot_pair(0, 10, init=True)

    def plot_pair(self, index_1, index_2, init=False):
        attr_1d_i = np.copy(self.attr_1d[index_1, index_2])
        attr_1d_norm = np.delete(attr_1d_i, [index_1, index_2])
        attr_1d_mean = np.mean(attr_1d_norm)
        attr_1d_i[index_1] = attr_1d_mean
        attr_1d_i[index_2] = attr_1d_mean
        attr_max = np.amax(attr_1d_norm)
        attr_min = np.amin(attr_1d_norm)

        plt.clf()
        cluster_1 = self.clustering == self.clustering[index_1]
        cluster_2 = self.clustering == self.clustering[index_2]

        plt.scatter(
            self.plt_x,
            self.plt_y,
            c=attr_1d_i,
            s=15,
            picker=True,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        plt.colorbar(label="Positional Importance", pad=0.01)
        plt.scatter(
            self.plt_x[cluster_1],
            self.plt_y[cluster_1],
            c=attr_1d_i[cluster_1],
            s=30,
            marker="D",
            picker=False,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        plt.scatter(
            self.plt_x[cluster_2],
            self.plt_y[cluster_2],
            c=attr_1d_i[cluster_2],
            s=30,
            marker="s",
            picker=False,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        line = plt.plot(
            [self.plt_x[index_1], self.plt_x[index_2]],
            [self.plt_y[index_1], self.plt_y[index_2]],
            c="black",
            linestyle="None",
            picker=False,
        )
        plt.scatter(
            [self.plt_x[index_1]],
            [self.plt_y[index_1]],
            c="red",
            s=35,
            marker="D",
            picker=False,
        )
        plt.scatter(
            [self.plt_x[index_2]],
            [self.plt_y[index_2]],
            c="red",
            s=35,
            marker="s",
            picker=False,
        )
        if self.clustering[index_1] == self.clustering[index_2]:
            plt.scatter(
                [self.plt_x[index_1]],
                [self.plt_y[index_1]],
                c="red",
                s=35,
                marker="s",
                picker=False,
            )
            plt.scatter(
                [self.plt_x[index_2]],
                [self.plt_y[index_2]],
                c="red",
                s=35,
                marker="D",
                picker=False,
            )
        line[0].axes.annotate(
            "",
            xytext=(self.plt_x[index_1], self.plt_y[index_1]),
            xy=(self.plt_x[index_2], self.plt_y[index_2]),
            arrowprops=dict(arrowstyle="-|>", color="black"),
            size=14,
        )

        plt.xlabel("Time (days)")
        plt.ylabel("Weight (Kg)")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if init:
            self.fig.canvas.mpl_connect("pick_event", self.pick_event)
            plt.subplots_adjust(left=0.07, right=1.02, top=0.98, bottom=0.1)
            # plt.tight_layout()
            plt.show()
        plt.draw()

    def pick_event(self, event):
        new_index = event.ind[0]
        prev_index = self.prev_index
        self.prev_index = new_index
        self.plot_pair(prev_index, new_index)


def plot_attr_dist(group_mat, attr_1d, pred, diff=False, absolute=False):
    if absolute:
        attr_1d = np.abs(attr_1d)
    attr_1d = np.mean(attr_1d, axis=-1)
    seq_length = group_mat.shape[0]

    assign_score, assign_indices = PidentCluster().cluster_score(
        pred, group_mat, pig_count=10
    )
    clustering_full = assign_indices["clusters"]
    pred_to_true = dict(zip(assign_indices["pred"], assign_indices["true"]))

    dists = defaultdict(list)
    for i in range(seq_length):
        for j in range(seq_length):
            attr_pair = np.copy(attr_1d[i, j])
            clustering = np.copy(clustering_full)
            attr_pair = np.delete(attr_pair, [i, j])
            clustering = np.delete(clustering, [i, j])

            if clustering_full[i] == clustering_full[j]:
                cluster_indices = clustering == clustering_full[i]
                back_indices = np.logical_not(cluster_indices)
                dists["same"].append(attr_pair[cluster_indices])
                dists["pair_same"].append(attr_1d[i, j, [i, j]])
                dists["back (same)"].append(attr_pair[back_indices])

                if diff:
                    dist_diff = np.transpose(
                        np.meshgrid(attr_pair[cluster_indices], attr_pair[back_indices])
                    ).reshape(-1, 2)
                    dist_diff = dist_diff[:, 1] - dist_diff[:, 0]
                    dists["same-back"].append(dist_diff)
            else:
                attr_i = clustering == clustering_full[i]
                attr_j = clustering == clustering_full[j]
                cluster_indices = np.logical_or(
                    clustering == clustering_full[i], clustering == clustering_full[j]
                )
                back_indices = np.logical_not(cluster_indices)
                dists["diff"].append(attr_pair[cluster_indices])
                dists["pair_diff"].append(attr_1d[i, j, [i, j]])
                dists["back (diff)"].append(attr_pair[back_indices])

                if diff:
                    dist_diff = np.transpose(
                        np.meshgrid(attr_pair[attr_i], attr_pair[attr_j])
                    ).reshape(-1, 2)
                    dist_diff = dist_diff[:, 1] - dist_diff[:, 0]
                    dists["diff1-diff2"].append(dist_diff)

                    dist_diff = np.transpose(
                        np.meshgrid(attr_pair[attr_i], attr_pair[back_indices])
                    ).reshape(-1, 2)
                    dist_diff = dist_diff[:, 1] - dist_diff[:, 0]
                    dists["diff1-back"].append(dist_diff)

                    dist_diff = np.transpose(
                        np.meshgrid(attr_pair[attr_j], attr_pair[back_indices])
                    ).reshape(-1, 2)
                    dist_diff = dist_diff[:, 1] - dist_diff[:, 0]
                    dists["diff2-back"].append(dist_diff)
            dists["back"].append(attr_pair[back_indices])

            if diff:
                dist_diff = np.transpose(
                    np.meshgrid(attr_pair[back_indices], attr_pair[back_indices])
                ).reshape(-1, 2)
                dist_diff = dist_diff[:, 1] - dist_diff[:, 0]
                dists["back-back"].append(dist_diff)

            attr_pair = np.copy(attr_1d[i, j])
            clustering = np.copy(clustering_full)
            attr_pair = np.delete(attr_pair, [i, j])
            clustering = np.delete(clustering, [i, j])
            cluster_i = pred_to_true[clustering_full[i]]
            cluster_j = pred_to_true[clustering_full[j]]
            if cluster_i == cluster_j:
                cluster_indices = clustering == cluster_i
                back_indices = np.logical_not(cluster_indices)
                dists["t_same"].append(attr_pair[cluster_indices])
                dists["t_pair_same"].append(attr_1d[i, j, [i, j]])
                dists["back (t_same)"].append(attr_pair[back_indices])
            else:
                cluster_indices = np.logical_or(
                    clustering == cluster_i, clustering == cluster_j
                )
                back_indices = np.logical_not(cluster_indices)
                dists["t_diff"].append(attr_pair[cluster_indices])
                dists["t_pair_diff"].append(attr_1d[i, j, [i, j]])
                dists["back (t_diff)"].append(attr_pair[back_indices])
            dists["t_back"].append(attr_pair[back_indices])

    for key, values in dists.items():
        dists[key] = np.concatenate(values)
        print(key, "std:", np.std(dists[key]))

    print()
    print("diff-back:", np.std(dists["diff"]) - np.std(dists["back"]))
    print("same-diff:", np.std(dists["same"]) - np.std(dists["diff"]))
    print()
    print(sps.bartlett(dists["same"], dists["diff"], dists["back"]))
    print(sps.levene(dists["same"], dists["diff"], dists["back"]))
    print(sps.fligner(dists["same"], dists["diff"], dists["back"]))

    fig = plt.figure(figsize=(5, 6))
    plt.boxplot(
        [dists["same"], dists["back (same)"]],
        labels=["Same cluster points", "Remaining points"],
        showfliers=False,
        showmeans=True,
        widths=(0.4, 0.4),
    )

    if diff:
        plt.boxplot(
            [
                dists["same-back"],
                dists["diff1-diff2"],
                dists["diff1-back"],
                dists["diff2-back"],
                dists["back-back"],
            ],
            labels=[
                "same-back",
                "diff1-diff2",
                "diff1-back",
                "diff2-back",
                "back-back",
            ],
            showfliers=False,
        )

    if absolute:
        plt.ylabel("Absolute attribution")
    else:
        plt.ylabel("Attribution")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(5, 6))
    plt.boxplot(
        [dists["diff"], dists["back (diff)"]],
        labels=["Points belonging to both clusters", "Remaining points"],
        showfliers=False,
        showmeans=True,
        widths=(0.4, 0.4),
    )

    if absolute:
        plt.ylabel("Absolute attribution")
    else:
        plt.ylabel("Attribution")
    plt.tight_layout()
    plt.show()


def plot_attr_corr(group_mat, attr_1d, pred, absolute=False, use_pred=False):
    if use_pred:
        c_values = pred
        cbar_label = "Predicted value"
    else:
        c_values = np.mean(attr_1d, axis=(-1, -2))
        if absolute:
            attr_1d = np.abs(attr_1d)
            cbar_label = (
                "Absolute mean of 1D feature gradients (positional contribution)"
            )
        else:
            cbar_label = "Mean of 1D feature gradients (positional contribution)"
        c_values = np.mean(attr_1d, axis=(-1, -2))

    time_data = group_mat[:, 1] / DAY_SECONDS
    weight_data = group_mat[:, 0] / 1000
    seq_size = group_mat.shape[0]

    weight_mat = np.expand_dims(weight_data, axis=1)
    weight_mat = np.broadcast_to(weight_mat, (seq_size, seq_size))
    weight_mat = weight_mat.T - weight_mat
    time_mat = np.expand_dims(time_data, axis=1)
    time_mat = np.broadcast_to(time_mat, (seq_size, seq_size))
    time_mat = time_mat.T - time_mat

    values = np.stack((weight_mat, time_mat, c_values), axis=0)
    values = values.reshape(values.shape[0], -1)

    fig = plt.figure(figsize=(24, 12))
    plt.scatter(values[0], values[1], c=values[2], s=15)
    plt.colorbar(label=cbar_label)
    plt.xlabel("Weight difference (Kg)")
    plt.ylabel("Time difference (days)")
    plt.tight_layout()
    plt.show()


def plot_attr_mean(group_mat, attr_1d):
    attr_1d = np.mean(attr_1d, axis=(0, 1))
    plt_x = group_mat[:, 1] / DAY_SECONDS
    plt_y = group_mat[:, 0] / 1000
    fig = plt.figure(figsize=(24, 12))
    plt.scatter(plt_x, plt_y, c=attr_1d, s=15)
    cbar = plt.colorbar(
        label="Sum of 1D feature gradients (positional contribution)", ticks=None
    )
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.tight_layout()
    plt.show()
    plt.draw()


def plot_attr(group_mat, attr_1d, attr_2d, delta, top=5):
    fig = plt.figure(figsize=(24, 12))
    plt_x = group_mat[:, 1] / DAY_SECONDS
    plt_y = group_mat[:, 0] / 1000
    cm = mpl.cm.get_cmap("Reds")

    attr_1d = np.sum(attr_1d, axis=-1)
    attr_2d = np.sum(attr_2d, axis=-1)
    attr_1d = np.mean(np.stack((attr_1d, attr_2d)), axis=0)

    def plot_attr_on_pick(event):
        new_index = event.ind
        attr_1d_ind = np.copy(attr_1d[new_index]).squeeze()
        attr_1d_ind -= np.amin(attr_1d_ind)
        attr_1d_ind /= np.amax(attr_1d_ind)
        attr_1d_top_ix = np.argsort(attr_1d_ind)[-top:]

        plt.clf()
        plt.scatter(plt_x, plt_y, c="gray", s=15, picker=True, pickradius=0.3)
        plt.scatter(
            plt_x[new_index], plt_y[new_index], c="black", marker="x", picker=False
        )
        for i in attr_1d_top_ix:
            plt.plot(
                [plt_x[new_index].item(), plt_x[i].item()],
                [plt_y[new_index].item(), plt_y[i].item()],
                c=cm(attr_1d_ind[i]),
                picker=False,
            )
        plt.xlabel("Time (days)")
        plt.ylabel("Weight (Kg)")
        plt.draw()

    attr_1d_base = np.copy(attr_1d[0])
    attr_1d_base -= np.amin(attr_1d_base)
    attr_1d_base /= np.amax(attr_1d_base)
    attr_1d_top_ix = np.argsort(attr_1d_base)[-top:]

    plt.scatter(plt_x, plt_y, c="gray", s=15, picker=True, pickradius=0.3)
    plt.scatter(plt_x[0], plt_y[0], c="black", marker="x", picker=False)
    for i in attr_1d_top_ix:
        plt.plot(
            [plt_x[0], plt_x[i]],
            [plt_y[0], plt_y[i]],
            c=cm(attr_1d_base[i]),
            picker=False,
        )
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    fig.canvas.mpl_connect("pick_event", plot_attr_on_pick)
    plt.tight_layout()
    plt.show()
    plt.draw()


def plot_attr_conn(group_mat, attr, top=None, alpha_scale=5):
    fig = plt.figure(figsize=(24, 12))
    plt_x = group_mat[:, 1] / DAY_SECONDS
    plt_y = group_mat[:, 0] / 1000
    attr = np.sum(attr, axis=-1)
    attr -= np.amin(attr)
    attr /= np.amax(attr)
    attr_alpha = attr**alpha_scale
    attr_alpha /= np.amax(attr_alpha)
    cm = mpl.cm.get_cmap("hot_r")

    for i in range(group_mat.shape[0]):
        if top != None:
            attr_sorted = np.argsort(attr[i])
            for j in attr_sorted[-top:]:
                plt.plot(
                    [plt_x[i], plt_x[j]],
                    [plt_y[i], plt_y[j]],
                    c=cm(attr[i, j]),
                    alpha=attr_alpha[i, j],
                )
        else:
            for j in range(group_mat.shape[0]):
                plt.plot(
                    [plt_x[i], plt_x[j]],
                    [plt_y[i], plt_y[j]],
                    c=cm(attr[i, j]),
                    alpha=attr_alpha[i, j],
                )
    plt.scatter(plt_x, plt_y, c="black", s=10)
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.tight_layout()
    plt.show()


def plot_attr_feature(group_mat, attr_1d):
    var_attr = np.mean(attr_1d, axis=(0, 1))
    labels = ["weight (Kg)", "sin(time)", "cos(time)"]
    plt.figure(figsize=(10, 10))
    plt.boxplot(var_attr, labels=labels)
    plt.tight_layout()
    plt.show()

    feature_count = attr_1d.shape[3]
    group_vars = np.zeros((group_mat.shape[0], feature_count))
    group_vars[:, 0] = group_mat[:, 0] / 1000
    group_vars[:, 1] = np.sin((group_mat[:, 1] / 86400) / 180)
    group_vars[:, 2] = np.cos((group_mat[:, 1] / 86400) / 180)
    fig, ax = plt.subplots(nrows=feature_count, ncols=feature_count, figsize=(20, 10))
    for i in range(feature_count):
        for j in range(feature_count):
            ax[i, j].scatter(var_attr[:, i], var_attr[:, j], s=15)
            ax[i, j].set_xlabel(labels[i])
            ax[i, j].set_ylabel(labels[j])
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pident2 Interpret")
    parser.add_argument(
        "-i", default="clod50e-s", dest="model_id", help="Model ID (str)"
    )
    parser.add_argument(
        "-t",
        default="ig",
        dest="type",
        choices=["ig", "dl", "svs", "gs", "oc"],
        help="Method type (str)",
    )
    parser.add_argument(
        "-b",
        default="random-f",
        dest="baseline",
        choices=[
            "zero",
            "zero-c",
            "zero-w",
            "random",
            "random-f",
            "random-t",
            "random-s",
        ],
        help="Baseline values (str)",
    )
    parser.add_argument(
        "-p",
        default="pair",
        dest="plot_type",
        choices=["pair", "corr", "pred-corr", "feature", "dist"],
        help="Plot type (str)",
    )
    parser.add_argument(
        "-a",
        default=False,
        action="store_true",
        dest="absolute",
        help="Use absolute values (flag only)",
    )
    parser.add_argument("-m", default="pident2", dest="model", help="Model type (str)")
    parser.add_argument(
        "-g", default=0, dest="group_id", type=int, help="Group index (int)"
    )
    parser.add_argument(
        "-f", default=0, dest="cv_fold", type=int, help="Cross-validation fold (int)"
    )
    parser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        dest="cpu",
        help="Use CPU for inference (flag only)",
    )
    parser.add_argument(
        "-l",
        default=False,
        action="store_true",
        dest="load",
        help="Load saved attributions (flag only)",
    )
    args = vars(parser.parse_args())

    if args["load"]:
        load_attr(
            args["model_id"],
            args["baseline"],
            args["type"],
            args["group_id"],
            args["plot_type"],
            args["absolute"],
        )
        sys.exit(0)

    file_path = "metrics/" + args["model_id"] + "/" + args["model_id"]
    fold_n = args["cv_fold"]
    with open(file_path + "_cv" + str(fold_n) + "_metrics.dat", "rb") as fh:
        fold_metrics = pickle.load(fh)

    m_args = fold_metrics["args"]
    m_args["params"] = fold_metrics["params"]
    if not args["cpu"] and torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
        print("CUDA available - using GPU")
    else:
        torch_device = torch.device("cpu")

    data_loader = DataLoader(
        batch_size=1,
        series_lengths=m_args["sim_length"],
        train_size=m_args["train_count"],
        eval_size=m_args["eval_count"],
        pig_counts=m_args["sim_animals"],
        sample_rates=m_args["sim_rate"],
        shuffle=False,
        length_limit=-1,
        cv_fold_total=m_args["cv_fold_total"],
        nested_fold_total=m_args["nested_fold_total"],
        pre_load=False,
        time_shift=m_args["time_shift"],
        time_weight=m_args["time_weight"],
        use_feed=m_args["use_feed"],
        cluster_label=m_args["cluster_loss"],
        day_trim=-1,
    )
    feature_size = data_loader.get_feature_size()
    pident_net = pident_models[args["model"]](
        m_args["params"], feature_size, torch_device, attn_map=False, batch_first=True
    ).to(torch_device)

    model_path = file_path + "_cv" + str(fold_n) + "_state.dat"
    pident_net.load_state_dict(torch.load(model_path, map_location=torch_device))
    pident_net.eval()

    model_pred(
        args["model_id"],
        fold_n,
        data_loader,
        pident_net,
        torch_device,
        baseline=args["baseline"],
        method=args["type"],
        group_index=args["group_id"],
    )


if __name__ == "__main__":
    main()
