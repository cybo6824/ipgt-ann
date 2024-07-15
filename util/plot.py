import itertools
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy as sp
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
import seaborn as sns

MODEL_COLORS = {
    "gb": "red",
    "mlp": "blue",
    "rf": "limegreen",
    "svm_nys": "purple",
    "linear": "green",
    "nn": "orange",
    "nn-mse": "salmon",
}
ALT_COLORS = ["brown", "darkblue", "deepskyblue", "crimson"]

DAY_SECONDS = 86400
PLOT_COLORS = [
    "red",
    "blue",
    "limegreen",
    "purple",
    "green",
    "orange",
    "salmon",
    "black",
    "tan",
    "crimson",
    "teal",
    "cyan",
    "mediumslateblue",
    "fuchsia",
    "pink",
]


def plot_series_basic(group_mat):
    plt.figure(figsize=(12, 4))
    plt.scatter(group_mat[:, 1] / DAY_SECONDS, group_mat[:, 0] / 1000, c="r", s=5)
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()


def plot_full_comparison(
    mean_results,
    err_results,
    var_names,
    var_ranges,
    metric="rmse",
    x_index=0,
    subplot_index=1,
):
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
    ax_l = ax.reshape(-1)
    series_length = 35
    all_scores = list()
    all_errors = list()
    alt_col_i = 0

    for model_name, model_results in mean_results.items():
        model_errors = err_results[model_name]
        try:
            plot_col = MODEL_COLORS[model_name]
        except KeyError:
            plot_col = ALT_COLORS[alt_col_i]
            alt_col_i += 1
        for i, subplot_value in enumerate(var_ranges[subplot_index]):
            plot_x = list()
            plot_values = list()
            plot_errors = list()
            for x_value in var_ranges[x_index]:
                result_code = (
                    str(x_value)
                    + "-"
                    + str(subplot_value).replace(".", ",")
                    + "-"
                    + str(series_length)
                )
                try:
                    plot_values.append(model_results[result_code][metric])
                    all_scores.append(model_results[result_code][metric])
                    plot_errors.append(model_errors[result_code][metric])
                    all_errors.append(model_errors[result_code][metric])
                    plot_x.append(x_value)
                except KeyError:
                    continue
            try:
                ax_l[i].errorbar(
                    plot_x,
                    plot_values,
                    yerr=plot_errors,
                    c=plot_col,
                    label=model_name,
                    capsize=5.0,
                )
                ax_l[i].title.set_text(
                    var_names[subplot_index] + ": " + str(subplot_value)
                )
                ax_l[i].yaxis.grid(True)
                ax_l[i].set_xticks([10, 20, 30, 40, 50])
            except IndexError:
                continue
    all_scores = np.asarray(all_scores)
    all_errors = np.asarray(all_errors)

    min_score = min(all_scores - all_errors)
    max_score = max(all_scores + all_errors)
    if metric == "rmse":
        lim_offset = 0.1
    else:
        lim_offset = 0.03
    for plot_i in range(ax_l.shape[0]):
        ax_l[plot_i].set_ylim(min_score - lim_offset, max_score + lim_offset)

    metric_labels = {
        "rmse": "RMSE (Kg)",
        "jaccard": "Jaccard",
        "r2": "R2",
        "f1": "F1",
        "f05": "F0.5",
        "f2": "F2",
    }

    fig.text(
        0.48, 0.02, var_names[x_index], ha="center", va="center", fontsize="medium"
    )
    fig.text(
        0.02,
        0.5,
        metric_labels[metric],
        ha="center",
        va="center",
        rotation="vertical",
        fontsize="medium",
    )
    handles, labels = ax_l[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, title="Model", fontsize="medium", frameon=False)
    plt.subplots_adjust(left=0.06, right=0.9, top=0.95, bottom=0.06)
    plt.show()


def plot_average_comparison(
    mean_results,
    err_results,
    var_names,
    var_ranges,
    metric="rmse",
    x_index=0,
    subplot_index=1,
):
    plt.figure(figsize=(10, 6))
    series_length = 35
    all_scores = list()
    all_errors = list()
    alt_col_i = 0

    models_full = {
        "rf": "Random forest",
        "gb": "Gradient boosting",
        "linear": "Linear",
        "mlp": "Multilayer perceptron",
        "svm_nys": "Support vector machine",
        "nn": "Attention neural network",
    }

    for model_name, model_results in mean_results.items():
        model_errors = err_results[model_name]
        try:
            plot_col = MODEL_COLORS[model_name]
        except KeyError:
            plot_col = ALT_COLORS[alt_col_i]
            alt_col_i += 1
        plot_values_all = list()
        plot_errors_all = list()
        for i, subplot_value in enumerate(var_ranges[subplot_index]):
            plot_x = list()
            plot_values = list()
            plot_errors = list()
            for x_value in var_ranges[x_index]:
                result_code = (
                    str(x_value)
                    + "-"
                    + str(subplot_value).replace(".", ",")
                    + "-"
                    + str(series_length)
                )
                plot_values.append(model_results[result_code][metric])
                plot_errors.append(model_errors[result_code][metric])
                plot_x.append(x_value)
            plot_values_all.append(np.asarray(plot_values))
            plot_errors_all.append(np.asarray(plot_errors))
        plot_values_all = np.mean(np.stack(plot_values_all, axis=0), axis=0)
        plot_errors_all = np.mean(np.stack(plot_errors_all, axis=0), axis=0)
        all_scores.append(plot_values_all)
        all_errors.append(plot_errors_all)
        plt.errorbar(
            plot_x,
            plot_values_all,
            yerr=plot_errors_all,
            c=plot_col,
            label=models_full[model_name],
            capsize=5.0,
        )
    all_scores = np.concatenate(all_scores)
    all_errors = np.concatenate(all_errors)

    min_score = min(all_scores - all_errors)
    max_score = max(all_scores + all_errors)
    if metric == "rmse":
        lim_offset = 0.1
    else:
        lim_offset = 0.03
    axis = plt.gca()
    axis.set_ylim(min_score - lim_offset, max_score + lim_offset)

    metric_labels = {
        "rmse": "RMSE (Kg)",
        "jaccard": "Jaccard",
        "r2": "R2",
        "f1": "F1",
        "f05": "F0.5",
        "f2": "F2",
    }

    plt.legend(
        loc=7,
        title="Model",
        fontsize="medium",
        frameon=False,
        bbox_to_anchor=(0.85, 0.25, 0.5, 0.5),
    )
    plt.subplots_adjust(left=0.08, right=0.75, top=0.9, bottom=0.1)
    plt.grid(visible=True, axis="y")
    plt.xlabel(var_names[x_index], fontdict={"size": 14})
    plt.ylabel(metric_labels[metric], fontdict={"size": 14})
    plt.xticks([10, 20, 30, 40, 50])
    plt.show()


def plot_single_comparison(
    mean_results,
    err_results,
    var_names,
    var_ranges,
    metric="rmse",
    x_index=0,
    subplot_value=2.0,
):
    plt.figure(figsize=(10, 6))
    series_length = 35
    all_scores = list()
    all_errors = list()
    alt_col_i = 0

    models_full = {
        "rf": "Random forest",
        "gb": "Gradient boosting",
        "linear": "Linear",
        "mlp": "Multilayer perceptron",
        "svm_nys": "Support vector machine",
        "nn": "Attention neural network",
    }

    model_index = 0
    for model_name, model_results in mean_results.items():
        if model_name == "linear" and metric == "jaccard":
            model_index += 1
            continue
        try:
            plot_col = MODEL_COLORS[model_name]
        except KeyError:
            plot_col = ALT_COLORS[alt_col_i]
            alt_col_i += 1

        model_errors = err_results[model_name]
        plot_x = list()
        plot_values = list()
        plot_errors = list()
        for x_value in var_ranges[x_index]:
            result_code = (
                str(x_value)
                + "-"
                + str(subplot_value).replace(".", ",")
                + "-"
                + str(series_length)
            )
            try:
                plot_values.append(model_results[result_code][metric])
                all_scores.append(model_results[result_code][metric])
                plot_errors.append(model_errors[result_code][metric])
                all_errors.append(model_errors[result_code][metric])
                plot_x.append(x_value)
            except KeyError:
                continue
        plt.errorbar(
            plot_x,
            plot_values,
            yerr=plot_errors,
            c=plot_col,
            label=models_full[model_name],
            capsize=5.0,
        )
        model_index += 1
    all_scores = np.asarray(all_scores)
    all_errors = np.asarray(all_errors)

    min_score = min(all_scores - all_errors)
    max_score = max(all_scores + all_errors)
    if metric == "rmse":
        lim_offset = 0.1
    else:
        lim_offset = 0.03
    axis = plt.gca()
    axis.set_ylim(min_score - lim_offset, max_score + lim_offset)

    metric_labels = {
        "rmse": "RMSE (Kg)",
        "jaccard": "Jaccard",
        "r2": "R2",
        "f1": "F1",
        "f05": "F0.5",
        "f2": "F2",
    }

    plt.legend(
        loc=7,
        title="Model",
        fontsize="medium",
        frameon=False,
        bbox_to_anchor=(0.85, 0.25, 0.5, 0.5),
    )
    plt.subplots_adjust(left=0.06, right=0.75, top=0.9, bottom=0.1)
    plt.grid(visible=True, axis="y")
    plt.xlabel(var_names[x_index])
    plt.ylabel(metric_labels[metric])
    plt.xticks([10, 20, 30, 40, 50])
    plt.show()


def plot_prediction_map(y_pred, y_true):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].imshow(y_pred, cmap="viridis_r")
    ax[1].imshow(y_true)
    plt.tight_layout()
    plt.show()


def plot_cluster_map(dist_mat, save_fig=False):
    dist_array = ssd.squareform(dist_mat)
    dist_linkage = hierarchy.linkage(dist_array, method="complete")

    cmap = sns.cm.rocket_r
    sns.clustermap(
        dist_mat, row_linkage=dist_linkage, col_linkage=dist_linkage, cmap=cmap
    )
    if save_fig:
        plt.savefig("plots/pident_cluster_heatmap.png", dpi=500)
    else:
        plt.show()


def plot_attn_map(y_pred, y_true, attn_map):
    fig, ax = plt.subplots(nrows=4, ncols=9, figsize=(24, 14))

    for layer_i, layer_maps in enumerate(attn_map):
        for head_i, head_map in enumerate(layer_maps):
            ax[layer_i, head_i].imshow(head_map["attn"])

    ax[0, -1].imshow(y_pred)
    ax[1, -1].imshow(y_true)
    plt.tight_layout()
    plt.show()


def plot_comparison_combo(true_to_pred, true_series, pred_series, save_fig=False):
    pig_count = len(true_series)
    cm = plt.get_cmap("gist_ncar")
    plt.figure(figsize=(12, 6))
    starting_weights = [true_series[i][0, 0] for i in range(pig_count)]
    true_ids, pred_ids, starting_weights = zip(
        *sorted(
            zip(
                list(true_to_pred.keys()), list(true_to_pred.values()), starting_weights
            ),
            reverse=True,
            key=lambda x: x[2],
        )
    )

    rng = np.random.Generator(np.random.PCG64(890))
    colour_order = np.arange(pig_count)
    rng.shuffle(colour_order)
    colour_set = [cm(1.0 * i / pig_count) for i in colour_order]

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)
    for true_id in true_ids:
        plt.plot(
            true_series[true_id][:, 1] / DAY_SECONDS,
            true_series[true_id][:, 0] / 1000,
            linewidth=1.0,
        )

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)
    for pred_id in pred_ids:
        pred_order = np.argsort(pred_series[pred_id][:, 1])
        plt.plot(
            pred_series[pred_id][pred_order, 1] / DAY_SECONDS,
            pred_series[pred_id][pred_order, 0] / 1000,
            linestyle=":",
            linewidth=2.0,
        )
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.xlim((-1, 36))
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    if save_fig:
        plt.savefig("plots/pident2_comparison_plot_combo.jpg")
    else:
        plt.show()


def plot_full_combo(
    y_pred,
    pred_indices,
    true_indices,
    true_to_pred,
    true_series,
    pred_series,
    save_fig=False,
):
    pig_count = len(true_series)
    cm = plt.get_cmap("gist_ncar")
    starting_weights = [true_series[i][0, 0] for i in range(pig_count)]
    true_ids, pred_ids, starting_weights = zip(
        *sorted(
            zip(
                list(true_to_pred.keys()), list(true_to_pred.values()), starting_weights
            ),
            reverse=True,
            key=lambda x: x[2],
        )
    )

    rng = np.random.Generator(np.random.PCG64(890))
    colour_order = np.arange(pig_count)
    rng.shuffle(colour_order)
    colour_set = [cm(1.0 * i / pig_count) for i in colour_order]

    animal_indices = np.unique(true_indices)
    y_pred = np.tril(y_pred) + np.triu(y_pred.T, 1)
    pred_image_ps = np.zeros(y_pred.shape)
    pred_image_ts = np.zeros(y_pred.shape)
    true_image_ps = np.zeros((y_pred.shape[0], y_pred.shape[0], 4))
    true_image_ts = np.zeros((y_pred.shape[0], y_pred.shape[0], 4))
    count_pred = 0
    count_true = 0
    for i, (true_id, pred_id) in enumerate(zip(true_ids, pred_ids)):
        pred_indices_a = pred_indices == pred_id
        pred_points = np.sum(pred_indices_a)
        pred_image_ps[count_pred : count_pred + pred_points] = y_pred[pred_indices_a]
        true_image_ps[
            count_pred : count_pred + pred_points, pred_indices_a
        ] = colour_set[i]
        count_pred += pred_points

        true_indices_a = true_indices == animal_indices[true_id]
        true_points = np.sum(true_indices_a)
        pred_image_ts[count_true : count_true + true_points] = y_pred[true_indices_a]
        true_image_ts[
            count_true : count_true + true_points, true_indices_a
        ] = colour_set[i]
        count_true += true_points

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(pred_image_ps, cmap=sns.cm.rocket_r)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Sorted by predicted cluster")

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.imshow(true_image_ps)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Predicted labels")

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(pred_image_ts, cmap=sns.cm.rocket_r)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Sorted by true cluster")

    ax4 = fig.add_subplot(gs[1, 3])
    ax4.imshow(true_image_ts)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("True labels")

    ax5 = fig.add_subplot(gs[0, :2])
    for i, true_id in enumerate(true_ids):
        ax5.plot(
            true_series[true_id][:, 1] / DAY_SECONDS,
            true_series[true_id][:, 0] / 1000,
            linewidth=1.0,
            c=colour_set[i],
        )

    for i, pred_id in enumerate(pred_ids):
        pred_order = np.argsort(pred_series[pred_id][:, 1])
        ax5.plot(
            pred_series[pred_id][pred_order, 1] / DAY_SECONDS,
            pred_series[pred_id][pred_order, 0] / 1000,
            linestyle=":",
            linewidth=2.0,
            c=colour_set[i],
        )
    ax5.set_xlabel("Time (days)")
    ax5.set_ylabel("Weight (Kg)")
    ax5.set_xlim((-1, 36))
    ax5.spines["right"].set_visible(False)
    ax5.spines["top"].set_visible(False)

    if save_fig:
        plt.savefig("plots/pident2_full_combo.jpg")
    else:
        plt.show()


def plot_model_var(
    mean_results, err_results, var_names, var_ranges, x_index=0, subplot_index=1
):
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=3)
    ax_l = ax.reshape(-1)
    series_length = 35

    model_index = 0
    scores_all = list()
    model_plot_data = dict()
    for model_name, model_results in mean_results.items():
        model_plot_data[model_name] = defaultdict(list)
        for i, sample_rate in enumerate(var_ranges[subplot_index]):
            scores_sr = list()
            pig_counts_sr = list()
            sample_rates_sr = list()
            for pig_count in var_ranges[x_index]:
                result_code = (
                    str(pig_count)
                    + "-"
                    + str(sample_rate).replace(".", ",")
                    + "-"
                    + str(series_length)
                )
                try:
                    scores_sr.append(model_results[result_code])
                    scores_all.append(model_results[result_code])
                    pig_counts_sr.append(pig_count)
                    sample_rates_sr.append(sample_rate)
                except KeyError:
                    continue
            model_plot_data[model_name]["scores"].append(scores_sr)
            model_plot_data[model_name]["pig_counts"].append(pig_counts_sr)
            model_plot_data[model_name]["sample_rates"].append(sample_rates_sr)

    min_score = np.amin(scores_all)
    max_score = np.amax(scores_all)
    for model_name, plot_data in model_plot_data.items():
        cf = ax_l[model_index].contourf(
            plot_data["pig_counts"],
            plot_data["sample_rates"],
            plot_data["scores"],
            vmin=min_score,
            vmax=max_score,
        )
        ax_l[model_index].title.set_text(model_name)
        ax_l[model_index].set_xticks(var_ranges[x_index])
        ax_l[model_index].set_yticks(var_ranges[subplot_index])
        fig.colorbar(cf, ax=ax_l[model_index])
        model_index += 1

    fig.text(
        0.48, 0.01, var_names[x_index], ha="center", va="center", fontsize="medium"
    )
    fig.text(
        0.007,
        0.5,
        "Sample rate",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize="medium",
    )
    plt.tight_layout()
    plt.show()


def plot_loss_score(mean_results, eval_losses):
    pig_counts = [10, 20, 30, 40, 50]
    sample_rates = [1.0]
    series_lengths = [35]
    var_ranges = [pig_counts, sample_rates, series_lengths]

    losses = list()
    scores = list()
    colours = list()
    for model_name in mean_results.keys():
        losses.append(eval_losses[model_name][0])
        if eval_losses[model_name][1]:
            colours.append("g")
        else:
            colours.append("r")
        model_scores = list()
        for pig_count, sample_rate, series_length in itertools.product(*var_ranges):
            file_code = (
                str(pig_count)
                + "-"
                + str(sample_rate).replace(".", ",")
                + "-"
                + str(series_length)
            )
            model_scores.append(mean_results[model_name][file_code])
        mean_score = np.mean(model_scores)
        scores.append(mean_score)
        print(model_name, eval_losses[model_name], mean_score)
    print("\nPearson correlation coefficient:", sp.stats.pearsonr(losses, scores))

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(losses, scores, c=colours)
    plt.xlabel("Loss (MSE)")
    plt.ylabel("Trajectory mean RMSE (Kg / pig)")
    plt.tight_layout()
    plt.show()


def plot_raw_series(group_mat, mode="scatter"):
    plt.figure(figsize=(12, 4))

    animal_ids = np.unique(group_mat[:, 2])
    pig_count = len(animal_ids)
    cm = plt.get_cmap("gist_ncar")
    rng = np.random.Generator(np.random.PCG64(890))
    colour_order = np.arange(pig_count)
    rng.shuffle(colour_order)
    colour_set = [cm(1.0 * i / pig_count) for i in colour_order]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)

    for i in range(pig_count):
        true_indices = np.squeeze(np.argwhere(group_mat[:, 2] == animal_ids[i]))
        true_series_i = group_mat[true_indices]
        if true_series_i.ndim == 1:
            true_series_i = np.expand_dims(true_series_i, 0)
        true_series_i = true_series_i[np.argsort(true_series_i[:, 1])]
        if mode == "line":
            plt.plot(true_series_i[:, 1] / DAY_SECONDS, true_series_i[:, 0] / 1000)
        elif mode == "scatter":
            plt.scatter(
                true_series_i[:, 1] / DAY_SECONDS, true_series_i[:, 0] / 1000, s=10
            )
        else:
            raise RuntimeError("PlotRawSeries: Unknown mode: " + mode)

    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_true_pred(group_mat, clustering_labels, pig_count):
    animal_ids = np.unique(group_mat[:, 2])
    true_series = list()
    pred_series = list()
    for i in range(pig_count):
        pred_indices = np.squeeze(np.argwhere(clustering_labels == i))
        pred_series_i = group_mat[pred_indices, :2]
        if pred_series_i.ndim == 1:
            pred_series_i = np.expand_dims(pred_series_i, 0)
        pred_series.append(pred_series_i)

    for i in range(pig_count):
        true_indices = np.squeeze(np.argwhere(group_mat[:, 2] == animal_ids[i]))
        true_series_i = group_mat[true_indices]
        if true_series_i.ndim == 1:
            true_series_i = np.expand_dims(true_series_i, 0)
        true_series.append(true_series_i)

    cm = plt.get_cmap("gist_ncar")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[cm(1.0 * i / pig_count) for i in range(pig_count)]
    )
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    ax[0].title.set_text("True growth sequences")
    ax[1].title.set_text("Predicted growth sequences")
    for i in range(pig_count):
        ax[0].plot(true_series[i][:, 1] / DAY_SECONDS, true_series[i][:, 0] / 1000)
        ax[0].set_xlabel("Time (days)")
        ax[0].set_ylabel("Weight (Kg)")
    for i in range(pig_count):
        pred_order = np.argsort(pred_series[i][:, 1])
        ax[1].plot(
            pred_series[i][pred_order, 1] / DAY_SECONDS,
            pred_series[i][pred_order, 0] / 1000,
        )
        ax[1].set_xlabel("Time (days)")
        ax[1].set_ylabel("Weight (Kg)")
    plt.tight_layout()
    plt.show()


def plot_bw_comparison(true_to_pred, true_series, pred_series, scores):
    pig_count = len(true_series)
    fig, ax = plt.subplots(figsize=(12, 9), nrows=2, ncols=1)
    print(scores)
    print(list(true_to_pred.keys()))
    print(list(true_to_pred.values()))

    for true_id in true_to_pred.keys():
        ax[0].plot(
            true_series[true_id][:, 1] / DAY_SECONDS,
            true_series[true_id][:, 0] / 1000,
            linewidth=1.0,
            c="grey",
        )
        ax[1].plot(
            true_series[true_id][:, 1] / DAY_SECONDS,
            true_series[true_id][:, 0] / 1000,
            linewidth=1.0,
            c="grey",
        )

    for pred_id in true_to_pred.values():
        pred_order = np.argsort(pred_series[pred_id][:, 1])
        ax[0].plot(
            pred_series[pred_id][pred_order, 1] / DAY_SECONDS,
            pred_series[pred_id][pred_order, 0] / 1000,
            linestyle=":",
            linewidth=2.0,
            c="grey",
        )
        ax[1].plot(
            pred_series[pred_id][pred_order, 1] / DAY_SECONDS,
            pred_series[pred_id][pred_order, 0] / 1000,
            linestyle=":",
            linewidth=2.0,
            c="grey",
        )

    best_index_true = list(true_to_pred.keys())[np.argmin(scores)]
    best_index_pred = list(true_to_pred.values())[np.argmin(scores)]
    worst_index_true = list(true_to_pred.keys())[np.argmax(scores)]
    worst_index_pred = list(true_to_pred.values())[np.argmax(scores)]
    pred_order = np.argsort(pred_series[best_index_pred][:, 1])
    ax[0].plot(
        true_series[best_index_true][:, 1] / DAY_SECONDS,
        true_series[best_index_true][:, 0] / 1000,
        linewidth=1.0,
        c="c",
    )
    ax[0].plot(
        pred_series[best_index_pred][pred_order, 1] / DAY_SECONDS,
        pred_series[best_index_pred][pred_order, 0] / 1000,
        linestyle=":",
        linewidth=2.0,
        c="g",
    )

    pred_order = np.argsort(pred_series[worst_index_pred][:, 1])
    ax[1].plot(
        true_series[worst_index_true][:, 1] / DAY_SECONDS,
        true_series[worst_index_true][:, 0] / 1000,
        linewidth=1.0,
        c="b",
    )
    ax[1].plot(
        pred_series[worst_index_pred][pred_order, 1] / DAY_SECONDS,
        pred_series[worst_index_pred][pred_order, 0] / 1000,
        linestyle=":",
        linewidth=2.0,
        c="r",
    )

    ax[0].set_xlabel("Time (days)")
    ax[0].set_ylabel("Weight (Kg)")
    txt_1 = plt.text(0.02, 0.8, "A", fontsize=50, transform=ax[0].transAxes)
    txt_1.set_in_layout(False)
    ax[1].set_xlabel("Time (days)")
    ax[1].set_ylabel("Weight (Kg)")
    txt_2 = plt.text(0.02, 0.8, "B", fontsize=50, transform=ax[1].transAxes)
    txt_2.set_in_layout(False)
    plt.xlim((-1, 36))
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    plt.tight_layout()
    plt.show()
