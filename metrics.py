import sys
import os
import pickle
import argparse
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from util.data_loader import DataLoader
from model import pident_models
from util.cluster import PidentCluster
import util.plot as pd_plt

MODELS_COMPARE = ["clod50e", "mse"]
MODEL_ALIASES = {"clod50e": "nn", "mse": "nn-mse"}

COMPARISON_PIG_COUNTS = [10, 20, 30, 40, 50]
COMPARISON_SAMPLE_RATES = [1.0, 2.0, 3.0, 4.0]
COMPARISON_SERIES_LENGTHS = [35]


def get_pident_scores(var_ranges, n_folds=3):
    mean_results = dict()
    err_results = dict()
    df_dict = defaultdict(list)
    series_scores = dict()
    pident_models_full = {
        "rf-ncv": "Random Forest",
        "gb-ncv": "Gradient Boosting",
        "linear": "Linear",
        "mlp-ncv": "Multilayer Perceptron",
        "svm_nys-ncv": "SVM Nystroem",
    }
    pident_models = ["gb-ncv", "mlp-ncv", "rf-ncv", "svm_nys-ncv", "linear"]
    for model in pident_models:
        if model[-4:] == "-ncv":
            model_alias = model[:-4]
        else:
            model_alias = model
        mean_results[model_alias] = dict()
        err_results[model_alias] = dict()
        series_scores[model] = defaultdict(list)
        for series_length, sample_rate, pig_count in itertools.product(
            *reversed(var_ranges)
        ):
            file_code = (
                str(pig_count)
                + "-"
                + str(sample_rate).replace(".", ",")
                + "-"
                + str(series_length)
            )
            mean_results[model_alias][file_code] = dict()
            err_results[model_alias][file_code] = dict()
            fold_scores = defaultdict(list)
            for fold_n in range(n_folds):
                file_name = (
                    model + "_benchmark_" + file_code + "_" + str(fold_n + 1) + ".dat"
                )
                try:
                    with open("pident1_metrics/" + model + "/" + file_name, "rb") as fh:
                        file_in = pickle.load(fh)
                        for series in file_in:
                            if type(series) == list:
                                series_tr = series[-1]
                            else:
                                series_tr = series
                            for score_type, score_value in series_tr.items():
                                fold_scores[score_type].append(score_value)
                except FileNotFoundError:
                    continue
            for score_type, score_values in fold_scores.items():
                series_scores[model][score_type] += score_values
                mean_results[model_alias][file_code][score_type] = np.mean(score_values)
                err_results[model_alias][file_code][score_type] = np.std(
                    score_values
                ) / np.sqrt(len(score_values))
                df_dict[score_type + " mean"].append(
                    mean_results[model_alias][file_code][score_type]
                )
                df_dict[score_type + " standard error"].append(
                    err_results[model_alias][file_code][score_type]
                )
            df_dict["Sample Rate"].append(sample_rate)
            df_dict["Pig Count"].append(pig_count)
            df_dict["Model"].append(pident_models_full[model])
        for score_type, score_values in series_scores.items():
            series_scores[model][score_type] = np.asarray(
                series_scores[model][score_type]
            )
    return mean_results, err_results, df_dict, series_scores


def get_pident2_scores(var_ranges, n_folds=3):
    mean_results = dict()
    err_results = dict()
    loss_results = dict()
    df_dict = defaultdict(list)
    series_scores = dict()
    for model in MODELS_COMPARE:
        model_split = model.split("_")
        model_alias = model
        try:
            model_alias = MODEL_ALIASES[model]
        except KeyError:
            pass
        mean_results[model_alias] = dict()
        err_results[model_alias] = dict()
        loss_results[model_alias] = list()
        series_scores[model] = defaultdict(list)
        model_id = model_split[0]
        train_str = ""
        trim_str = ""
        if len(model_split) > 1:
            for split_str in model_split:
                if split_str == "train":
                    train_str = "_train"
                elif split_str[:2] == "tr":
                    trim_str = "_" + split_str
        file_path = "metrics/" + model_id + "/" + model_id

        for fold_n in range(n_folds):
            with open(file_path + "_cv" + str(fold_n) + "_metrics.dat", "rb") as fh:
                model_metrics = pickle.load(fh)
            try:
                loss_weight = model_metrics["args"]["loss_weight"]
            except KeyError:
                loss_weight = False
            loss_results[model_alias].append(
                model_metrics["cv" + str(fold_n) + "_eval_loss"][-1]
            )
        loss_results[model_alias] = [
            np.mean(loss_results[model_alias]),
            loss_weight,
            model_metrics["args"]["batch_size"],
        ]

        for series_length, sample_rate, pig_count in itertools.product(
            *reversed(var_ranges)
        ):
            file_code = (
                str(pig_count)
                + "-"
                + str(sample_rate).replace(".", ",")
                + "-"
                + str(series_length)
            )
            mean_results[model_alias][file_code] = dict()
            err_results[model_alias][file_code] = dict()
            fold_scores = defaultdict(list)
            file_ext = "_" + file_code + trim_str + train_str + "_cscores.dat"
            try:
                for fold_n in range(n_folds):
                    with open(file_path + "_cv" + str(fold_n) + file_ext, "rb") as fh:
                        n_scores = pickle.load(fh)
                    if type(n_scores[0]) == dict:
                        for series in n_scores.values():
                            for score_type, score_value in series.items():
                                fold_scores[score_type].append(score_value)
                    else:
                        fold_scores["rmse"] += list(n_scores.values())
            except FileNotFoundError:
                continue
            for score_type, score_values in fold_scores.items():
                series_scores[model][score_type] += score_values
                mean_results[model_alias][file_code][score_type] = np.mean(score_values)
                err_results[model_alias][file_code][score_type] = np.std(
                    score_values
                ) / np.sqrt(len(score_values))
                df_dict[score_type + " mean"].append(
                    mean_results[model_alias][file_code][score_type]
                )
                df_dict[score_type + " standard error"].append(
                    err_results[model_alias][file_code][score_type]
                )
            df_dict["Sample Rate"].append(sample_rate)
            df_dict["Pig Count"].append(pig_count)
            df_dict["Model"].append(model_alias)
        for score_type, score_values in series_scores.items():
            series_scores[model][score_type] = np.asarray(
                series_scores[model][score_type]
            )
    return mean_results, err_results, df_dict, series_scores, loss_results


def model_comparison_full(load_pident_scores=False, n_folds=3):
    var_ranges = [
        COMPARISON_PIG_COUNTS,
        COMPARISON_SAMPLE_RATES,
        COMPARISON_SERIES_LENGTHS,
    ]
    var_names = ["Pig count", "Sample rate", "Series length"]

    if load_pident_scores:
        mean_results, err_results, df_dict, s_scores = get_pident_scores(
            var_ranges, n_folds=n_folds
        )
        (
            mean_results_2,
            err_results_2,
            df_dict_2,
            s_scores_2,
            eval_loss,
        ) = get_pident2_scores(var_ranges, n_folds=n_folds)
        mean_results.update(mean_results_2)
        err_results.update(err_results_2)
        df_dict.update(df_dict_2)
        s_scores.update(s_scores_2)
    else:
        mean_results, err_results, df_dict, s_scores, eval_loss = get_pident2_scores(
            var_ranges, n_folds=n_folds
        )

    for model_name, model_scores in s_scores.items():
        print(model_name, "mean RMSE score:", np.mean(model_scores["rmse"]))
        print(model_name, "mean Jaccard score:", np.mean(model_scores["jaccard"]))
        print(model_name, "mean R2 score:", np.mean(model_scores["r2"]))
        print()

    pd_df = pd.DataFrame(data=df_dict)
    pd_df.to_csv("metrics/metrics_comparison.csv")
    pd_plt.plot_full_comparison(
        mean_results, err_results, var_names, var_ranges, metric="rmse"
    )
    pd_plt.plot_full_comparison(
        mean_results, err_results, var_names, var_ranges, metric="jaccard"
    )
    pd_plt.plot_full_comparison(
        mean_results, err_results, var_names, var_ranges, metric="r2"
    )


def get_model_size(file_path):
    state_dict = torch.load(file_path + "_cv0_nf0_state.dat", map_location="cpu")
    model_size = 0
    for param_set in state_dict.values():
        size_tuple = tuple(param_set.size())
        model_size += np.prod(size_tuple)
    return model_size


def ncv_param_analysis(args):
    file_path = "metrics/" + args["model_id"] + "/"
    file_names = os.listdir(file_path)
    model_scores = defaultdict(list)
    for sub_id in file_names:
        if sub_id.find(".") == -1:
            sub_scores = defaultdict(list)
            model_path = file_path + sub_id + "/" + args["model_id"]
            for pig_count in args["sim_animals"]:
                for sample_rate in args["sim_rate"]:
                    file_code = (
                        str(pig_count)
                        + "-"
                        + str(sample_rate).replace(".", ",")
                        + "-"
                        + str(args["sim_length"])
                    )
                    for cv_fold in range(args["cv_folds"]):
                        for nested_fold in range(args["nested_folds"]):
                            file_id = "_cv" + str(cv_fold) + "_nf" + str(nested_fold)
                            s_file = model_path + file_id + "_" + file_code
                            try:
                                with open(s_file + "_cscores.dat", "rb") as fh:
                                    in_scores = pickle.load(fh)
                            except FileNotFoundError:
                                continue
                            for metric in ["rmse", "r2", "jaccard", "f1"]:
                                scores = list()
                                if metric != "f1":
                                    for series in in_scores.values():
                                        scores.append(series[metric])
                                else:
                                    for series in in_scores.values():
                                        f1_score = (
                                            2.0
                                            * (series["r2"] * series["jaccard"])
                                            / (series["r2"] + series["jaccard"])
                                        )
                                        scores.append(f1_score)
                                sub_scores[metric].append(np.mean(scores))
            model_scores["id"].append(sub_id)
            sub_id_split = sub_id.split("_")
            model_scores["batch size"].append(sub_id_split[0])
            model_scores["dim 1d"].append(sub_id_split[1])
            model_scores["dim 2d"].append(sub_id_split[2])
            model_scores["dim conn"].append(sub_id_split[3])
            model_scores["depth"].append(sub_id_split[4])
            model_scores["model_size"].append(get_model_size(model_path))
            for metric in ["rmse", "r2", "jaccard", "f1"]:
                model_scores[metric].append(np.mean(sub_scores[metric]))
            for metric in ["rmse", "r2", "jaccard", "f1"]:
                model_scores[metric + " error"].append(
                    np.std(sub_scores[metric]) / np.sqrt(len(sub_scores[metric]))
                )

    pd_df = pd.DataFrame(data=model_scores)
    pd_df.to_csv("metrics/" + args["model_id"] + "_metrics.csv")


def plot_loss(args, fold_metrics, running_avg=8):
    fig, ax = plt.subplots(
        nrows=args["cv_folds"], ncols=1, figsize=(12, args["cv_folds"] * 4)
    )
    for fold_n in range(args["cv_folds"]):
        if fold_metrics[fold_n] == None:
            continue
        model_metrics = fold_metrics[fold_n]
        metrics_key = "cv" + str(fold_n) + "_"
        if running_avg == None:
            ax[fold_n].plot(model_metrics[metrics_key + "batch_loss"], c="g")
        else:
            batch_losses = list()
            for i in range(
                len(model_metrics[metrics_key + "batch_loss"]) - running_avg
            ):
                batch_losses.append(
                    np.mean(
                        model_metrics[metrics_key + "batch_loss"][i : i + running_avg]
                    )
                )
            ax[fold_n].plot(batch_losses, c="g")
        epoch_series = np.cumsum(model_metrics[metrics_key + "total_batches"])
        train_skew = np.asarray(model_metrics[metrics_key + "total_batches"]) / 2
        ax[fold_n].plot(
            epoch_series - train_skew, model_metrics[metrics_key + "train_loss"], c="b"
        )
        ax[fold_n].plot(epoch_series, model_metrics[metrics_key + "eval_loss"], c="r")
        ax[fold_n].set_xlabel("Batch")
        ax[fold_n].set_ylabel("Loss")
    plt.tight_layout()
    plt.show()


def pident_pred(
    pident_net,
    data_loader,
    fold_n,
    args,
    torch_device,
    cluster_label=False,
    log_freq=20,
):
    total_batches = data_loader.total_batches
    epoch_pred = list()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            x_1d = batch["x_1d"].to(torch_device)
            x_2d = batch["x_2d"].to(torch_device)
            pad_mask = batch["pad_mask"].to(torch_device)
            y = batch["y"].numpy()

            if not args["plot_pred"]:
                net_output = pident_net(x_1d, x_2d, pad_mask)
            else:
                net_output, attn_map = pident_net(x_1d, x_2d, pad_mask)

            if cluster_label:
                net_output = (net_output + net_output.transpose(-1, -2)) / 2.0
                net_output = F.normalize(net_output, dim=(-1, -2))

            for i in range(x_1d.size(1)):
                pred_data = batch["data"][i]
                src_len = pred_data["group_mat"].shape[0]
                pred_data["pred"] = net_output[i][:src_len, :src_len].cpu().numpy()
                if cluster_label:
                    pred_data["pred"] -= np.amin(pred_data["pred"])
                else:
                    pred_data["pred"] -= 1.0
                    pred_data["pred"] *= -1.0
                np.fill_diagonal(pred_data["pred"], 0.0)
                pred_data["true"] = y[i, :src_len, :src_len]
                if args["plot_pred"]:
                    pd_plt.plot_cluster_map(pred_data["pred"])
                    pd_plt.plot_prediction_map(
                        pred_data["pred"], y[i, :src_len, :src_len]
                    )
                    # pd_plt.plot_attn_map(pred_data["pred"], y.numpy(), attn_map[i])
                epoch_pred.append(pred_data)
            if (batch_index + 1) % log_freq == 0 and batch_index > 0:
                percent_progress = ((batch_index + 1) / total_batches) * 100
                print(
                    "Fold: %2d/%2d, Progress: %5.2f%%, Batch: %4d/%4d"
                    % (
                        fold_n + 1,
                        args["cv_folds"],
                        percent_progress,
                        batch_index + 1,
                        total_batches,
                    ),
                    flush=True,
                )
    return epoch_pred


def model_eval(fold_n, args, fold_metrics, nested_fold=None):
    print("Generating predictions for fold", fold_n)
    for pig_count in args["sim_animals"]:
        for sample_rate in args["sim_rate"]:
            file_code = (
                str(pig_count)
                + "-"
                + str(sample_rate).replace(".", ",")
                + "-"
                + str(args["sim_length"])
            )
            train_str = "_train" if args["train"] else ""
            ncv_str = "" if nested_fold == None else "_nf" + str(nested_fold)
            trim_str = "" if args["trim"] == -1 else "_tr" + str(args["trim"])
            file_path = args["file_path"] + "_cv" + str(fold_n) + ncv_str
            score_file = file_path + "_" + file_code + trim_str + train_str
            if args["score"]:
                try:
                    with open(score_file + "_cscores.dat", "rb") as fh:
                        _ = pickle.load(fh)
                    if args["index"]:
                        with open(score_file + "_indices.dat", "rb") as fh:
                            _ = pickle.load(fh)
                    print("Skipping pig count", pig_count, "sample rate", sample_rate)
                    continue
                except (FileNotFoundError, EOFError) as e:
                    pass

            print("pig count", pig_count, "sample rate", sample_rate)
            eval_pred = get_predictions(
                file_path,
                fold_n,
                args,
                fold_metrics[fold_n],
                [pig_count],
                [sample_rate],
                args["sim_length"],
                args["eval_count"],
                nested_fold=nested_fold,
            )
            if args["score"] or args["plot_tj"]:
                eval_scores, eval_indices = score_predictions(
                    eval_pred, args["plot_tj"]
                )

                if args["score"]:
                    with open(score_file + "_cscores.dat", "wb") as fh:
                        pickle.dump(eval_scores, fh)
                    if args["index"]:
                        with open(score_file + "_indices.dat", "wb") as fh:
                            pickle.dump(eval_indices, fh)


def get_predictions(
    file_path,
    fold_n,
    args,
    model,
    pig_count,
    sample_rate,
    series_length,
    eval_count,
    nested_fold=None,
):
    m_args = model["args"]

    data_loader = DataLoader(
        batch_size=args["batch_size"],
        series_lengths=[series_length],
        train_size=m_args["train_count"],
        eval_size=eval_count,
        pig_counts=pig_count,
        sample_rates=sample_rate,
        shuffle=False,
        length_limit=-1,
        cv_fold_total=m_args["cv_fold_total"],
        nested_fold_total=m_args["nested_fold_total"],
        time_shift=False,
        time_weight=0.0,
        use_feed=m_args["use_feed"],
        day_trim=args["trim"],
    )

    if not args["cpu"] and torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
        print("CUDA available - using GPU")
    else:
        torch_device = torch.device("cpu")

    try:
        cluster_loss = m_args["cluster_loss"]
    except KeyError:
        cluster_loss = False

    in_features = data_loader.get_feature_size()
    pident_net = pident_models[args["model"]](
        model["params"], in_features, torch_device, attn_map=args["plot_pred"]
    ).to(torch_device)
    pident_net.load_state_dict(
        torch.load(file_path + "_state.dat", map_location=torch_device)
    )
    pident_net.eval()
    data_loader.set_dataset(
        "eval" if not args["train"] else "train", fold_n, nested_fold
    )
    eval_pred = pident_pred(
        pident_net, data_loader, fold_n, args, torch_device, cluster_loss
    )
    return eval_pred


def score_predictions(pred, plot_mode=False):
    print("Calculating cluster scores...")
    pident_cluster = PidentCluster()
    model_scores = dict()
    model_indices = dict()
    for pred_item in pred:
        group_id = pred_item["group_id"]
        group_score, group_indices = pident_cluster.cluster_score(
            pred_item["pred"],
            pred_item["group_mat"],
            pred_item["pig_count"],
            plot_mode=plot_mode,
        )
        model_scores[group_id] = group_score
        model_indices[group_id] = group_indices
    return model_scores, model_indices


def main():
    parser = argparse.ArgumentParser(description="Pident2 metrics")
    parser.add_argument("-i", "--id", dest="model_id", help="Model ID (str)")
    parser.add_argument(
        "-s", default=None, dest="sub_id", type=str, help="Model sub ID (str)"
    )
    parser.add_argument("-m", default="pident2", dest="model", help="Model type (str)")
    parser.add_argument(
        "-c",
        default=False,
        action="store_true",
        dest="compare",
        help="Model comparison (flag only)",
    )
    parser.add_argument(
        "-n",
        default=False,
        action="store_true",
        dest="ncv",
        help="Nested CV mode (flag only)",
    )
    parser.add_argument(
        "-l",
        default=False,
        action="store_true",
        dest="plot_loss",
        help="Plot loss (flag only)",
    )
    parser.add_argument(
        "-p",
        default=False,
        action="store_true",
        dest="plot_pred",
        help="Plot predictions (flag only)",
    )
    parser.add_argument(
        "--tj",
        default=False,
        action="store_true",
        dest="plot_tj",
        help="Plot trajectories (flag only)",
    )
    parser.add_argument(
        "--score",
        default=False,
        action="store_true",
        dest="score",
        help="Score predictions (flag only)",
    )
    parser.add_argument(
        "--index",
        default=False,
        action="store_true",
        dest="index",
        help="Index predictions (flag only)",
    )
    parser.add_argument(
        "-t",
        default=False,
        action="store_true",
        dest="train",
        help="Use training predictions (flag only)",
    )
    parser.add_argument(
        "-f",
        default=None,
        dest="cv_fold",
        type=int,
        help="Cross-validation fold (integer)",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        dest="batch_size",
        type=int,
        help="Batch size (integer)",
    )
    parser.add_argument(
        "--cv_folds",
        default=3,
        dest="cv_folds",
        type=int,
        help="Cross-validation folds (integer)",
    )
    parser.add_argument(
        "--nested_folds",
        default=5,
        dest="nested_folds",
        type=int,
        help="Nested folds (integer)",
    )
    parser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        dest="cpu",
        help="Use CPU for inference (flag only)",
    )
    parser.add_argument(
        "--threads",
        default=None,
        dest="n_threads",
        type=int,
        help="Number of CPU threads (flag only)",
    )

    parser.add_argument(
        "--sim_animals",
        default=[10, 20, 30, 40, 50],
        dest="sim_animals",
        nargs="+",
        type=int,
        help="Number of animals per group int (int or list(int))",
    )
    parser.add_argument(
        "--sim_rate",
        default=[1.0],
        dest="sim_rate",
        nargs="+",
        type=float,
        help="Sample rate (float or list(float))",
    )
    parser.add_argument(
        "--sim_length",
        default=35,
        dest="sim_length",
        type=int,
        help="Length of group series in days (integer)",
    )
    parser.add_argument(
        "--eval_count",
        default=100,
        dest="eval_count",
        type=int,
        help="Number of group series (integer)",
    )
    parser.add_argument(
        "--trim", default=-1, dest="trim", type=int, help="Day trim (integer)"
    )
    args = vars(parser.parse_args())

    if args["n_threads"] != None:
        torch.set_num_threads(args["n_threads"])
    args["score"] = True if args["index"] else args["score"]

    if args["compare"]:
        if args["ncv"]:
            ncv_param_analysis(args)
        else:
            model_comparison_full(n_folds=args["cv_folds"])
        sys.exit(0)

    if args["sub_id"] == None:
        sub_id = ""
    else:
        sub_id = args["sub_id"] + "/"
    args["file_path"] = "metrics/" + args["model_id"] + "/" + sub_id + args["model_id"]

    if args["ncv"]:
        inner_folds = range(args["nested_folds"])
        outer_folds = range(args["cv_folds"])
    else:
        inner_folds = range(args["cv_folds"])
        outer_folds = [-1]

    for outer_fold in outer_folds:
        fold_metrics = list()
        exist_indices = list()
        for fold_n in inner_folds:
            if args["ncv"]:
                cv_str = "_cv" + str(outer_fold) + "_nf" + str(fold_n)
            else:
                cv_str = "_cv" + str(fold_n)
            try:
                with open(args["file_path"] + cv_str + "_metrics.dat", "rb") as fh:
                    fold_metrics.append(pickle.load(fh))
                exist_indices.append(len(fold_metrics) - 1)
            except FileNotFoundError:
                fold_metrics.append(None)
        if len(exist_indices) == 0:
            raise RuntimeError("Cannot find metric files for " + args["model_id"])

        if args["plot_loss"]:
            plot_loss(args, fold_metrics)

        if args["score"] or args["plot_pred"] or args["plot_tj"]:
            for fold_n in exist_indices:
                if args["cv_fold"] != None and args["cv_fold"] != fold_n:
                    continue
                if args["ncv"]:
                    cv_fold = outer_fold
                    nested_fold = fold_n
                else:
                    cv_fold = fold_n
                    nested_fold = None
                model_eval(cv_fold, args, fold_metrics, nested_fold)


if __name__ == "__main__":
    main()
