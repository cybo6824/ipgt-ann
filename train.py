import sys
import os
import pickle
import time
import ast
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util.data_loader import DataLoader
from model import pident_models


def train_epoch(
    model, optimizer, data_loader, criterion, epoch, args, torch_device, log_freq=10
):
    epoch_start_time = time.time()
    train_epoch_loss = 0.0
    batch_losses = list()
    log_loss = 0.0
    total_batches = data_loader.total_batches
    for batch_index, batch in enumerate(data_loader):
        x_1d = batch["x_1d"].to(torch_device)
        x_2d = batch["x_2d"].to(torch_device)
        pad_mask = batch["pad_mask"].to(torch_device)

        net_output = model(x_1d, x_2d, pad_mask)
        if args["cluster_loss"]:
            net_output = (net_output + net_output.transpose(-1, -2)) / 2.0
            net_output = F.normalize(net_output, dim=(-1, -2))
            if args["time_weight"] > 0.0:
                y_w = batch["y_w"].to(torch_device)
            else:
                y_w = None
            loss = criterion(net_output, batch["y"], weights=y_w)
        else:
            y = batch["y"].to(torch_device)
            y_w = batch["y_w"].to(torch_device)
            loss = torch.mean(criterion(net_output, y) * y_w)

        optimizer.zero_grad()
        loss.backward()
        if args["cluster_loss"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        batch_loss = loss.item()
        train_epoch_loss += batch_loss
        log_loss += batch_loss
        batch_losses.append(batch_loss)
        if (batch_index + 1) % log_freq == 0 and batch_index > 0:
            log_loss /= log_freq
            percent_progress = (
                ((batch_index + 1) + epoch * total_batches)
                / (args["epochs"] * total_batches)
            ) * 100
            print(
                "Progress: %5.2f%%, Epoch: %2d/%2d, Batch: %4d/%4d, Loss: %.3f"
                % (
                    percent_progress,
                    epoch + 1,
                    args["epochs"],
                    batch_index + 1,
                    total_batches,
                    log_loss,
                ),
                flush=True,
            )
            log_loss = 0.0
    train_epoch_loss /= total_batches
    print(
        "Average training loss for epoch",
        epoch + 1,
        "/",
        args["epochs"],
        ":",
        train_epoch_loss,
    )
    print("Training epoch time:", time.time() - epoch_start_time)
    return batch_losses, train_epoch_loss


def eval_epoch(model, data_loader, criterion, epoch, args, torch_device, log_freq=10):
    eval_epoch_loss = 0.0
    total_batches = data_loader.total_batches
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            x_1d = batch["x_1d"].to(torch_device)
            x_2d = batch["x_2d"].to(torch_device)
            pad_mask = batch["pad_mask"].to(torch_device)

            net_output = model(x_1d, x_2d, pad_mask)
            if args["cluster_loss"]:
                net_output = (net_output + net_output.transpose(-1, -2)) / 2.0
                net_output = F.normalize(net_output, dim=(-1, -2))
                if args["time_weight"] > 0.0:
                    y_w = batch["y_w"].to(torch_device)
                else:
                    y_w = None
                loss = criterion(net_output, batch["y"], weights=y_w)
            else:
                y_w = batch["y_w"].to(torch_device)
                y = batch["y"].to(torch_device)
                loss = torch.mean(criterion(net_output, y) * y_w)

            eval_epoch_loss += loss.item()
            if (batch_index + 1) % log_freq == 0 and batch_index > 0:
                percent_progress = ((batch_index + 1) / total_batches) * 100
                print(
                    "Progress: %5.2f%%, Epoch: %2d/%2d, Batch: %4d/%4d"
                    % (
                        percent_progress,
                        epoch + 1,
                        args["epochs"],
                        batch_index + 1,
                        total_batches,
                    ),
                    flush=True,
                )
    if total_batches > 0:
        eval_epoch_loss /= total_batches
        print(
            "Average eval loss for epoch",
            epoch + 1,
            "/",
            args["epochs"],
            ":",
            eval_epoch_loss,
        )
    return eval_epoch_loss


def mem_test(args, in_features, device):
    fold_params = args["params"]
    if args["fold_param_mode"]:
        fold_params = fold_params[0]
    model = pident_models[args["model"]](fold_params, in_features, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    criterion = nn.MSELoss(reduction="none")
    model.train()

    bsz = args["batch_size"]
    seq_len = args["seq_limit"] if args["seq_limit"] != -1 else 280

    x_1d = torch.ones(seq_len, bsz, in_features[0], device=device)
    x_2d = torch.ones(bsz, seq_len, seq_len, in_features[1], device=device)
    pad_mask = torch.ones(bsz, seq_len, device=device)
    y = torch.ones(bsz, seq_len, seq_len, device=device)
    y_w = torch.ones(bsz, seq_len, seq_len, device=device)

    net_output = model(x_1d, x_2d, pad_mask)
    loss = torch.mean(criterion(net_output, y) * y_w)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(torch.cuda.memory_summary(device))


class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, labels, weights=None):
        bsz = input.size(0)
        same_dists = torch.empty(bsz, device=input.device, dtype=float)
        diff_dists = torch.empty(bsz, device=input.device, dtype=float)

        if weights == None:
            for i in range(bsz):
                same_dists[i] = torch.mean(
                    input[i, labels[i]["same_x"], labels[i]["same_y"]]
                )
                diff_dists[i] = torch.mean(
                    input[i, labels[i]["diff_x"], labels[i]["diff_y"]]
                )
        else:
            for i in range(bsz):
                same_dists[i] = torch.mean(
                    input[i, labels[i]["same_x"], labels[i]["same_y"]]
                    * weights[i, labels[i]["same_x"], labels[i]["same_y"]]
                )
                diff_dists[i] = torch.mean(
                    input[i, labels[i]["diff_x"], labels[i]["diff_y"]]
                    / weights[i, labels[i]["diff_x"], labels[i]["diff_y"]]
                )
        return torch.mean(same_dists) - torch.mean(diff_dists)


def get_next_model_index(folder):
    file_indices = list()
    file_list = os.listdir(folder)
    for filename in file_list:
        try:
            file_indices.append(int(filename))
        except ValueError:
            continue
    if len(file_indices) == 0:
        return "%04d" % 0
    return "%04d" % (max(file_indices) + 1)


def get_fold_lists(args):
    if args["cv_fold"] != None and args["cv_fold"] >= args["cv_fold_total"]:
        raise RuntimeError("cv_fold is greater than or equal to cv_fold_total")
    cv_folds = (
        [args["cv_fold"]] if args["cv_fold"] != None else range(args["cv_fold_total"])
    )
    if args["nested_fold_total"] != None:
        if (
            args["nested_fold"] != None
            and args["nested_fold"] >= args["nested_fold_total"]
        ):
            raise RuntimeError(
                "nested_fold is greater than or equal to nested_fold_total"
            )
        nested_folds = (
            [args["nested_fold"]]
            if args["nested_fold"] != None
            else range(args["nested_fold_total"])
        )
    else:
        nested_folds = [None]
    return cv_folds, nested_folds


def parse_model_params(params):
    if params == None:
        return None, None
    param_dict = ast.literal_eval(params)
    assert type(param_dict) == dict
    fold_params = True
    for key in param_dict.keys():
        try:
            int(key)
        except ValueError:
            fold_params = False
            break
    return param_dict, fold_params


def save_metrics(model_id, sub_id, cv_fold, nested_fold, model_metrics, model_state):
    cv_fold_str = "_cv" + str(cv_fold)
    nested_fold_str = ""
    sub_str = ""
    if nested_fold != None:
        nested_fold_str = "_nf" + str(nested_fold)
    if sub_id != None:
        sub_str = sub_id + "/"
    folder_id = "metrics/" + model_id + "/" + sub_str
    file_id = model_id + cv_fold_str + nested_fold_str
    with open(folder_id + file_id + "_metrics.dat", "wb") as fh:
        pickle.dump(model_metrics, fh)
    torch.save(model_state, folder_id + file_id + "_state.dat")


def main():
    parser = argparse.ArgumentParser(description="Pident2 train")
    parser.add_argument(
        "-t",
        default=None,
        dest="n_threads",
        type=int,
        help="Number of CPU threads (integer)",
    )
    parser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        dest="cpu",
        help="Use CPU for training and evaluation (flag only)",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        dest="test",
        help="Test mode - disables output saving (flag only)",
    )
    parser.add_argument(
        "--mem",
        default=False,
        action="store_true",
        dest="mem",
        help="Memory test mode (flag only)",
    )

    parser.add_argument(
        "--epochs",
        default=30,
        dest="epochs",
        type=int,
        help="Number of epochs (integer)",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        dest="batch_size",
        type=int,
        help="Batch size (integer)",
    )
    parser.add_argument(
        "--learn_rate",
        default=0.001,
        dest="learning_rate",
        type=float,
        help="Learning rate (float)",
    )
    parser.add_argument(
        "--nshuffle",
        default=True,
        action="store_false",
        dest="shuffle",
        help="Disable dataset shuffling (flag only)",
    )
    parser.add_argument(
        "--time_shift",
        default=False,
        action="store_true",
        dest="time_shift",
        help="Time shift (flag only)",
    )
    parser.add_argument(
        "--time_weight",
        default=10.0,
        dest="time_weight",
        type=float,
        help="Time-based loss weight (float)",
    )
    parser.add_argument(
        "--feed",
        default=False,
        action="store_true",
        dest="use_feed",
        help="Use feed consumption (flag only)",
    )
    parser.add_argument(
        "--mse",
        default=True,
        action="store_false",
        dest="cluster_loss",
        help="Use MSE loss function (flag only)",
    )

    parser.add_argument(
        "--id", default=None, dest="model_id", type=str, help="Model ID (str)"
    )
    parser.add_argument(
        "-s", default=None, dest="sub_id", type=str, help="Model sub ID (str)"
    )
    parser.add_argument(
        "-m",
        default="pident2",
        dest="model",
        choices=["pident2", "test"],
        help="Model type (str)",
    )
    parser.add_argument(
        "-p", default=None, dest="params", help="Model parameters (dict)"
    )

    parser.add_argument(
        "--sim_animals",
        default=[10],
        dest="sim_animals",
        nargs="+",
        type=int,
        help="Number of animals per group (list(integer))",
    )
    parser.add_argument(
        "--sim_rate",
        default=[1.0],
        dest="sim_rate",
        nargs="+",
        type=float,
        help="Sample rate (list(float))",
    )
    parser.add_argument(
        "--sim_length",
        default=[35],
        dest="sim_length",
        nargs="+",
        type=int,
        help="Length of group series in days (integer)",
    )
    parser.add_argument(
        "--seq_limit",
        default=280,
        dest="seq_limit",
        type=int,
        help="Group series max data points (integer)",
    )
    parser.add_argument(
        "--train_count",
        default=500,
        dest="train_count",
        type=int,
        help="Number of group series (integer)",
    )
    parser.add_argument(
        "--eval_count",
        default=100,
        dest="eval_count",
        type=int,
        help="Number of group series (integer)",
    )

    parser.add_argument(
        "--cv_fold",
        default=None,
        dest="cv_fold",
        type=int,
        help="Cross-validation fold index (integer)",
    )
    parser.add_argument(
        "--cv_fold_total",
        default=3,
        dest="cv_fold_total",
        type=int,
        help="Number of cross-validation folds (integer)",
    )
    parser.add_argument(
        "--nested_fold",
        default=None,
        dest="nested_fold",
        type=int,
        help="Nested cross-validation fold index (integer)",
    )
    parser.add_argument(
        "--nested_fold_total",
        default=None,
        dest="nested_fold_total",
        type=int,
        help="Number of nested cross-validation folds (integer)",
    )
    args = vars(parser.parse_args())

    if not args["test"] and not args["mem"]:
        if args["model_id"] != None and args["model_id"].find("_") != -1:
            raise ValueError("'Model ID' argument cannot contain any underscores")
        os.makedirs("metrics", exist_ok=True)
        if args["model_id"] == None or args["model_id"] == "None":
            args["model_id"] = get_next_model_index("metrics/")
        os.makedirs("metrics/" + args["model_id"], exist_ok=True)
        if args["sub_id"] != None:
            os.makedirs(
                "metrics/" + args["model_id"] + "/" + args["sub_id"], exist_ok=True
            )

    data_loader = DataLoader(
        batch_size=args["batch_size"],
        series_lengths=args["sim_length"],
        train_size=args["train_count"],
        eval_size=args["eval_count"],
        pig_counts=args["sim_animals"],
        sample_rates=args["sim_rate"],
        shuffle=args["shuffle"],
        length_limit=args["seq_limit"],
        cv_fold_total=args["cv_fold_total"],
        nested_fold_total=args["nested_fold_total"],
        pre_load=True,
        time_shift=args["time_shift"],
        time_weight=args["time_weight"],
        use_feed=args["use_feed"],
        cluster_label=args["cluster_loss"],
    )

    if torch.cuda.is_available() and not args["cpu"]:
        torch_device = torch.device("cuda:0")
        print("CUDA available - using GPU")
    else:
        torch_device = torch.device("cpu")

    if args["n_threads"] != None:
        torch.set_num_threads(args["n_threads"])

    cv_folds, nested_folds = get_fold_lists(args)
    args["params"], args["fold_param_mode"] = parse_model_params(args["params"])
    if args["cluster_loss"]:
        criterion = ClusterLoss()
    else:
        criterion = nn.MSELoss(reduction="none")
    in_features = data_loader.get_feature_size()

    if args["mem"]:
        mem_test(args, in_features, torch_device)
        sys.exit(0)

    for cv_fold in cv_folds:
        fold_params = args["params"]
        if args["fold_param_mode"]:
            fold_params = fold_params[cv_fold]

        for nested_fold in nested_folds:
            print("\nCV Fold:", cv_fold, "Nested fold:", nested_fold)
            model_metrics = defaultdict(list)
            metrics_key = "cv" + str(cv_fold) + "_"
            if nested_fold != None:
                metrics_key += "nf" + str(nested_fold) + "_"
            pident_net = pident_models[args["model"]](
                fold_params, in_features, torch_device
            ).to(torch_device)
            model_metrics["params"] = pident_net.get_params()
            model_metrics["args"] = args
            optimizer = optim.Adam(pident_net.parameters(), lr=args["learning_rate"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            for epoch in range(args["epochs"]):
                print("\nTraining epoch", epoch, "/", args["epochs"], "started...")
                pident_net.train()
                data_loader.set_dataset("train", cv_fold, nested_fold)
                batch_losses, train_loss = train_epoch(
                    pident_net,
                    optimizer,
                    data_loader,
                    criterion,
                    epoch,
                    args,
                    torch_device,
                )
                model_metrics[metrics_key + "batch_loss"] += batch_losses
                model_metrics[metrics_key + "train_loss"].append(train_loss)
                model_metrics[metrics_key + "total_batches"].append(
                    data_loader.total_batches
                )

                print("\nEval epoch", epoch, "/", args["epochs"], "started...")
                pident_net.eval()
                data_loader.set_dataset("eval", cv_fold, nested_fold)
                eval_loss = eval_epoch(
                    pident_net, data_loader, criterion, epoch, args, torch_device
                )
                model_metrics[metrics_key + "eval_loss"].append(eval_loss)
                scheduler.step()

            if not args["test"]:
                save_metrics(
                    args["model_id"],
                    args["sub_id"],
                    cv_fold,
                    nested_fold,
                    model_metrics,
                    pident_net.state_dict(),
                )
    if not args["test"]:
        print("\nTraining finished", args["model_id"])


if __name__ == "__main__":
    main()
