import sys
import os
import pickle
import argparse
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment
from util.forecast_util import generate_dataset


def train_cv(dataset, model_name, std_pred=False):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:, :-1])
    x_scaler = StandardScaler()
    data_x = x_scaler.fit_transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000
    cv_folds = PredefinedSplit(dataset["folds"])

    if model_name == "rf":
        rgs = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=555)
    elif model_name == "gb":
        rgs = GradientBoostingRegressor(random_state=555)
        if std_pred:
            rgs_std = GradientBoostingRegressor(random_state=555)
            data_y_mean = data_y[:, 0]
            data_y_std = data_y[:, 1]
    elif model_name == "lasso":
        if not std_pred:
            rgs = Lasso(max_iter=10000, random_state=555)
        else:
            rgs = MultiTaskLasso(random_state=555)
    elif model_name == "dummy":
        rgs = DummyRegressor()

    cv_out = defaultdict(list)
    if std_pred and model_name == "gb":
        cv_results_mean = cross_validate(
            rgs,
            data_x,
            data_y_mean,
            scoring="neg_root_mean_squared_error",
            n_jobs=3,
            return_estimator=True,
            cv=cv_folds,
        )
        cv_results_std = cross_validate(
            rgs_std,
            data_x,
            data_y_std,
            scoring="neg_root_mean_squared_error",
            n_jobs=3,
            return_estimator=True,
            cv=cv_folds,
        )
        cv_out["estimators"] = list()
        cv_out["validation_fold_rmse"] = list()
        for i in range(len(cv_results_mean["estimator"])):
            cv_out["estimators"].append(
                [cv_results_mean["estimator"][i], cv_results_std["estimator"][i]]
            )
            cv_out["validation_fold_rmse"].append(
                (cv_results_mean["test_score"][0] + cv_results_std["test_score"][0])
                * -0.5
            )
        cv_out["validation_mean_rmse"] = np.mean(cv_out["validation_fold_rmse"])
    else:
        cv_results = cross_validate(
            rgs,
            data_x,
            data_y,
            scoring="neg_root_mean_squared_error",
            n_jobs=3,
            return_estimator=True,
            cv=cv_folds,
        )
        cv_out["estimators"] = cv_results["estimator"]
        cv_out["validation_mean_rmse"] = np.mean(cv_results["test_score"]) * -1
        cv_out["validation_fold_rmse"] = [x * -1 for x in cv_results["test_score"]]

    cv_out["model_name"] = model_name
    cv_out["scaler"] = x_scaler
    fold_n = 0
    for train_ix, test_ix in cv_folds.split():
        x_test, y_test = data_x[test_ix], data_y[test_ix]
        if std_pred and model_name == "gb":
            y_pred_mean = cv_out["estimators"][fold_n][0].predict(x_test)
            y_pred_std = cv_out["estimators"][fold_n][1].predict(x_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = cv_out["estimators"][fold_n].predict(x_test)
        cv_out["pred"].append(y_pred)
        cv_out["true"].append(y_test)
        cv_out["groups"].append(dataset["groups"][test_ix])
        cv_out["steps"].append(dataset["steps"][test_ix])
        if std_pred:
            cv_out["counts"].append(dataset["x"][test_ix, -1])
        fold_n += 1
    return cv_out


def train_ncv(dataset, model_name, params=None, std_pred=False, n_threads=5):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:, :-1])
    x_scaler = StandardScaler()
    data_x = x_scaler.fit_transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000

    nested_out = defaultdict(list)
    nested_out["model_name"] = model_name
    nested_out["scaler"] = x_scaler

    cv_outer = PredefinedSplit(dataset["folds"])
    current_fold = 0
    fold_count = np.amax(dataset["folds"]) + 1
    for train_ix, test_ix in cv_outer.split():
        print("Starting fold", current_fold + 1, "/", fold_count)
        X_train, X_test = data_x[train_ix, :], data_x[test_ix, :]
        y_train, y_test = data_y[train_ix], data_y[test_ix]

        if model_name == "rf":
            rgs = RandomForestRegressor(random_state=555)
            param_grid = {
                "n_estimators": [50, 75, 100, 125, 150, 175, 200],
                "max_depth": [5, 10, 15, 20, 25, 30, 35],
            }
        elif model_name == "gb":
            rgs = GradientBoostingRegressor(random_state=555)
            if std_pred:
                rgs_std = GradientBoostingRegressor(random_state=555)
                y_train_mean = y_train[:, 0]
                y_train_std = y_train[:, 1]
            param_grid = {
                "n_estimators": [50, 75, 100, 125, 150, 175, 200],
                "max_depth": [2, 4, 6, 8, 10, 12],
            }
            param_grid_std = param_grid
        elif model_name == "dummy":
            rgs = DummyRegressor()

        if params != None:
            if std_pred and model_name == "gb":
                param_grid = {k: [v] for k, v in params[current_fold][0].items()}
                param_grid_std = {k: [v] for k, v in params[current_fold][1].items()}
            else:
                param_grid = {k: [v] for k, v in params[current_fold].items()}

        if std_pred and model_name == "gb":
            cv_inner_mean = GroupKFold(n_splits=5).split(
                X_train, y_train_mean, dataset["groups"][train_ix]
            )
            cv_inner_std = GroupKFold(n_splits=5).split(
                X_train, y_train_std, dataset["groups"][train_ix]
            )
        else:
            cv_inner = GroupKFold(n_splits=5).split(
                X_train, y_train, dataset["groups"][train_ix]
            )

        if model_name != "lasso":
            if std_pred and model_name == "gb":
                gridsearch_mean = GridSearchCV(
                    rgs,
                    param_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=cv_inner_mean,
                    n_jobs=n_threads,
                    refit=True,
                )
                gridsearch_std = GridSearchCV(
                    rgs_std,
                    param_grid_std,
                    scoring="neg_root_mean_squared_error",
                    cv=cv_inner_std,
                    n_jobs=n_threads,
                    refit=True,
                )
                gs_result_mean = gridsearch_mean.fit(X_train, y_train_mean)
                gs_result_std = gridsearch_std.fit(X_train, y_train_std)
                best_model = [
                    gs_result_mean.best_estimator_,
                    gs_result_std.best_estimator_,
                ]
                nested_out["best_params"].append(
                    [gs_result_mean.best_params_, gs_result_std.best_params_]
                )
                nested_out["gs_results"].append(
                    [gs_result_mean.cv_results_, gs_result_std.cv_results_]
                )
            else:
                gridsearch = GridSearchCV(
                    rgs,
                    param_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=cv_inner,
                    n_jobs=n_threads,
                    refit=True,
                )
                gs_result = gridsearch.fit(X_train, y_train)
                best_model = gs_result.best_estimator_
                nested_out["best_params"].append(gs_result.best_params_)
                nested_out["gs_results"].append(gs_result.cv_results_)
        else:
            if std_pred:
                if params == None:
                    gridsearch = MultiTaskLassoCV(max_iter=10000, random_state=555)
                    gs_result = gridsearch.fit(X_train, y_train)
                    lasso_alpha = gs_result.alpha_
                else:
                    lasso_alpha = params[current_fold]["alpha"]
                best_model = MultiTaskLasso(
                    alpha=lasso_alpha, max_iter=10000, random_state=555
                )
            else:
                if params == None:
                    gridsearch = LassoCV(max_iter=10000, random_state=555)
                    gs_result = gridsearch.fit(X_train, y_train)
                    lasso_alpha = gs_result.alpha_
                else:
                    lasso_alpha = params[current_fold]["alpha"]
                best_model = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=555)
            best_model.fit(X_train, y_train)
            nested_out["best_params"].append({"alpha": lasso_alpha})

        if std_pred and model_name == "gb":
            y_pred_mean = best_model[0].predict(X_test)
            y_pred_std = best_model[1].predict(X_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = best_model.predict(X_test)
        model_score = np.sqrt(np.mean((y_pred - y_test) ** 2))

        nested_out["pred"].append(y_pred)
        nested_out["true"].append(y_test)
        nested_out["groups"].append(dataset["groups"][test_ix])
        nested_out["steps"].append(dataset["steps"][test_ix])
        nested_out["validation_fold_rmse"].append(model_score)
        nested_out["estimators"].append(best_model)
        if std_pred:
            nested_out["counts"].append(dataset["x"][test_ix, -1])
        current_fold += 1

    print("Optimal parameters:", nested_out["best_params"])
    nested_out["validation_mean_rmse"] = np.mean(nested_out["validation_fold_rmse"])
    return nested_out


def test_cv(dataset, model, std_pred=False):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:, :-1])
    data_x = model["scaler"].transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000
    cv_folds = PredefinedSplit(dataset["folds"])

    test_out = defaultdict(list)
    fold_n = 0
    for train_ix, test_ix in cv_folds.split():
        x_test, y_test = data_x[test_ix], data_y[test_ix]
        if std_pred and model["model_name"] == "gb":
            y_pred_mean = model["estimators"][fold_n][0].predict(x_test)
            y_pred_std = model["estimators"][fold_n][1].predict(x_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = model["estimators"][fold_n].predict(x_test)
        model_score = np.sqrt(np.mean((y_pred - y_test) ** 2))
        test_out["test_fold_rmse"].append(model_score)
        if std_pred:
            model_score_meanpred = np.sqrt(np.mean((y_pred[:, 0] - y_test[:, 0]) ** 2))
            model_score_stdpred = np.sqrt(np.mean((y_pred[:, 1] - y_test[:, 1]) ** 2))
            test_out["test_fold_rmse_meanpred"].append(model_score_meanpred)
            test_out["test_fold_rmse_stdpred"].append(model_score_stdpred)
        fold_n += 1
    test_out["test_mean_rmse"] = np.mean(test_out["test_fold_rmse"])
    if std_pred:
        test_out["test_mean_rmse_meanpred"] = np.mean(
            test_out["test_fold_rmse_meanpred"]
        )
        test_out["test_mean_rmse_stdpred"] = np.mean(test_out["test_fold_rmse_stdpred"])
    return test_out


def mean_rel_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_pred))


def get_metrics(y_true_folds, y_pred_folds):
    fold_scores = defaultdict(list)
    for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
        fold_scores["r2"].append(r2_score(fold_true, fold_pred))
        fold_scores["mre"].append(mean_rel_error(fold_true, fold_pred))
        fold_scores["rmse"].append(np.sqrt(np.mean((fold_true - fold_pred) ** 2)))
    print("Matched individual RMSE fold scores:", fold_scores["rmse"])
    print("Matched individual RMSE average score:", np.mean(fold_scores["rmse"]))
    print("Matched individual R2 fold scores:", fold_scores["r2"])
    print("Matched individual R2 average score:", np.mean(fold_scores["r2"]))
    print("Matched individual MRE fold scores:", fold_scores["mre"])
    print("Matched individual MRE average score:", np.mean(fold_scores["mre"]))
    return fold_scores


def match_individual_weights(model):
    all_true = list()
    all_pred = list()
    for true, pred, groups, steps in zip(
        model["true"], model["pred"], model["groups"], model["steps"]
    ):
        fold_true = list()
        fold_pred = list()
        for group_i in np.unique(groups):
            group_indices = groups == group_i
            for step_i in np.unique(steps[group_indices]):
                step_indices = np.logical_and(group_indices, steps == step_i)
                true_step = true[step_indices]
                pred_step = pred[step_indices]
                dist_mat = np.zeros(
                    (true_step.shape[0], pred_step.shape[0]), dtype=float
                )
                for i in range(true_step.shape[0]):
                    for j in range(pred_step.shape[0]):
                        dist_mat[i, j] = (true_step[i] - pred_step[j]) ** 2
                true_indices, pred_indices = linear_sum_assignment(dist_mat)
                fold_true.append(true_step[true_indices])
                fold_pred.append(pred_step[pred_indices])
        all_true.append(np.concatenate(fold_true))
        all_pred.append(np.concatenate(fold_pred))
    return all_true, all_pred


def estimate_individual_weights(model, indv_dataset):
    rng = np.random.Generator(np.random.PCG64(555))

    cv_outer = PredefinedSplit(indv_dataset["folds"])
    indv_true_y = list()
    indv_true_groups = list()
    indv_true_steps = list()
    for train_ix, test_ix in cv_outer.split():
        indv_true_y.append(indv_dataset["y"][test_ix])
        indv_true_groups.append(indv_dataset["groups"][test_ix])
        indv_true_steps.append(indv_dataset["steps"][test_ix])

    all_true = list()
    all_pred = list()
    model_iter = zip(
        indv_true_y,
        indv_true_groups,
        indv_true_steps,
        model["pred"],
        model["groups"],
        model["steps"],
        model["counts"],
    )
    for (
        true,
        true_groups,
        true_steps,
        pred,
        pred_groups,
        pred_steps,
        pred_counts,
    ) in model_iter:
        fold_true = list()
        fold_pred = list()
        for group_i in np.unique(pred_groups):
            group_indices_true = true_groups == group_i
            group_indices_pred = pred_groups == group_i
            for step_i in np.unique(pred_steps[group_indices_pred]):
                step_indices_true = np.logical_and(
                    group_indices_true, true_steps == step_i
                )
                step_indices_pred = np.logical_and(
                    group_indices_pred, pred_steps == step_i
                )
                true_step = true[step_indices_true] / 1000
                pred_step_mean = pred[step_indices_pred, 0]
                pred_step_std = pred[step_indices_pred, 1]
                pred_count = round(pred_counts[step_indices_pred][0])
                pred_step = rng.normal(
                    loc=pred_step_mean, scale=pred_step_std, size=pred_count
                )
                dist_mat = np.zeros(
                    (true_step.shape[0], pred_step.shape[0]), dtype=float
                )
                assert true_step.shape[0] == pred_step.shape[0]

                for i in range(true_step.shape[0]):
                    for j in range(pred_step.shape[0]):
                        dist_mat[i, j] = (true_step[i] - pred_step[j]) ** 2
                true_indices, pred_indices = linear_sum_assignment(dist_mat)
                fold_true.append(true_step[true_indices])
                fold_pred.append(pred_step[pred_indices])
        all_true.append(np.concatenate(fold_true))
        all_pred.append(np.concatenate(fold_pred))
    return all_true, all_pred


def model_comparison(metric="rmse", nested_cv=False):
    figure_colours = [
        "red",
        "blue",
        "limegreen",
        "purple",
        "orange",
        "brown",
        "deepskyblue",
        "gold",
        "crimson",
        "darkred",
        "darkblue",
        "green",
    ]
    format_modes = ["true", "pred_rf", "pred_clod50e7l", "group"]
    format_markers = {"true": "o", "pred_rf": "x", "pred_clod50e7l": "+", "group": "d"}
    metric_full = dict({"rmse": "RMSE (Kg)", "mre": "Mean relative error", "r2": "R2"})
    lim_offsets = {"rmse": 0.5, "mre": 0.005, "r2": 0.025}
    pig_counts = [10, 20, 30, 40, 50]
    sample_rates = [1.0, 2.0, 3.0, 4.0]
    models = ["lasso", "rf", "gb"]
    models_full = {"lasso": "LASSO", "rf": "Random Forest", "gb": "Gradient Boosting"}
    validation_type = "ncv" if nested_cv else "cv"
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
    ax_l = ax.reshape(-1)
    all_scores = list()
    df_dict = defaultdict(list)
    type_scores = defaultdict(list)

    for plot_i, sample_rate in enumerate(sample_rates):
        series_i = 0
        for format_mode, model in itertools.product(*(format_modes, models)):
            plot_label = format_mode + "_" + model
            marker = format_markers[format_mode]
            if format_mode[:5] == "pred_":
                model_dir = format_mode[5:]
                model_name = validation_type + "_" + model + "_pred"
                if format_mode[5:] == "clod50e7l":
                    plot_label = "pred_nn_" + model
            else:
                model_dir = format_mode
                model_name = validation_type + "_" + model + "_" + format_mode

            score_series = list()
            pig_counts_series = list()
            for pig_count in pig_counts:
                file_code = (
                    "_" + str(pig_count) + "-" + str(sample_rate).replace(".", ",")
                )
                try:
                    with open(
                        "metrics_forecast/"
                        + model_dir
                        + "/"
                        + model_name
                        + file_code
                        + ".dat",
                        "rb",
                    ) as fh:
                        model_data = pickle.load(fh)
                except FileNotFoundError:
                    print("Missing model:", model_name + file_code)
                    continue
                score_series.append(model_data["eval_mean_" + metric])
                all_scores.append(model_data["eval_mean_" + metric])
                pig_counts_series.append(pig_count)
                df_dict[str(pig_count) + "-pig " + metric_full[metric]].append(
                    model_data["eval_mean_" + metric]
                )
                type_scores[format_mode].append(model_data["eval_mean_" + metric])

            if len(score_series) > 0:
                df_dict["Sample Rate"].append(sample_rate)
                # df_dict["Pig count"].append(pig_count)
                df_dict["Input Type"].append(format_mode.capitalize())
                df_dict["Model"].append(models_full[model])
                ax_l[plot_i].plot(
                    pig_counts_series,
                    score_series,
                    c=figure_colours[series_i],
                    label=plot_label,
                    marker=marker,
                )
                ax_l[plot_i].title.set_text("Sample rate: " + str(sample_rate))
                ax_l[plot_i].set_xlim((10 - 2, 50 + 2))
                ax_l[plot_i].yaxis.grid(True)
                series_i += 1

    min_score = min(all_scores)
    max_score = max(all_scores)
    for plot_i in range(ax_l.shape[0]):
        ax_l[plot_i].set_ylim(
            min_score - lim_offsets[metric], max_score + lim_offsets[metric]
        )

    plt.setp(ax, xticks=[10, 20, 30, 40, 50])
    fig.text(0.46, 0.02, "Pig count", ha="center", va="center", fontsize="medium")
    txt_lft_algn = 0.02 if metric != "mre" else 0.01
    fig.text(
        txt_lft_algn,
        0.5,
        metric_full[metric],
        ha="center",
        va="center",
        rotation="vertical",
        fontsize="medium",
    )
    handles, labels = ax_l[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, title="Model", fontsize="medium", frameon=False)
    plt.subplots_adjust(left=0.05, right=0.87, top=0.95, bottom=0.06)
    plt.show()

    pd_df = pd.DataFrame(data=df_dict)
    for score_type, score_values in type_scores.items():
        print(score_type + " mean:", np.mean(score_values))
    pd_df.to_csv("metrics_forecast/forecast_comparison_ " + metric + ".csv")


def model_comparison_average(metric="rmse", nested_cv=False):
    figure_colours = [
        "red",
        "blue",
        "limegreen",
        "purple",
        "orange",
        "brown",
        "deepskyblue",
        "gold",
        "crimson",
        "darkred",
        "darkblue",
        "green",
    ]
    format_modes = {
        "true": "RFID",
        "pred_rf": "RF predicted",
        "pred_clod50e7l": "ANN predicted",
        "group": "Group average",
    }
    format_markers = {"true": "o", "pred_rf": "x", "pred_clod50e7l": "+", "group": "d"}
    metric_full = dict({"rmse": "RMSE (Kg)", "mre": "Mean relative error", "r2": "R2"})
    lim_offsets = {"rmse": 0.5, "mre": 0.005, "r2": 0.025}
    pig_counts = [10, 20, 30, 40, 50]
    sample_rates = [1.0, 2.0, 3.0, 4.0]
    models = {"lasso": "Lasso", "rf": "Random Forest", "gb": "Gradient Boosting"}
    validation_type = "ncv" if nested_cv else "cv"
    plt.figure(figsize=(10, 6))
    all_scores = list()

    series_i = 0
    for format_mode, model in itertools.product(*(format_modes.keys(), models.keys())):
        plot_label = format_modes[format_mode] + " " + models[model]
        marker = format_markers[format_mode]
        if format_mode[:5] == "pred_":
            model_dir = format_mode[5:]
            model_name = validation_type + "_" + model + "_pred"
        else:
            model_dir = format_mode
            model_name = validation_type + "_" + model + "_" + format_mode

        model_scores = list()
        for pig_count in pig_counts:
            rate_scores = list()
            for sample_rate in sample_rates:
                file_code = (
                    "_" + str(pig_count) + "-" + str(sample_rate).replace(".", ",")
                )
                with open(
                    "metrics_forecast/"
                    + model_dir
                    + "/"
                    + model_name
                    + file_code
                    + ".dat",
                    "rb",
                ) as fh:
                    model_data = pickle.load(fh)
                rate_scores.append(model_data["eval_mean_" + metric])
            model_scores.append(np.asarray(rate_scores))
        model_scores = np.mean(np.stack(model_scores, axis=0), axis=1)
        all_scores.append(model_scores)

        plt.plot(
            pig_counts,
            model_scores,
            c=figure_colours[series_i],
            label=plot_label,
            marker=marker,
        )
        series_i += 1
    all_scores = np.concatenate(all_scores)

    min_score = np.amin(all_scores)
    max_score = np.amax(all_scores)
    axis = plt.gca()
    axis.set_ylim(min_score - lim_offsets[metric], max_score + lim_offsets[metric])

    plt.legend(
        loc=7,
        title="Model",
        fontsize="medium",
        frameon=False,
        bbox_to_anchor=(0.99, 0.25, 0.5, 0.5),
    )
    plt.subplots_adjust(left=0.08, right=0.7, top=0.9, bottom=0.1)
    plt.grid(visible=True, axis="y")
    plt.xlabel("Pig count", fontdict={"size": 14})
    plt.ylabel(metric_full[metric], fontdict={"size": 14})
    plt.xticks([10, 20, 30, 40, 50])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PIDENT Forecasting")
    parser.add_argument(
        "-f",
        default="true",
        dest="format_mode",
        choices=["true", "pred", "group"],
        help="Trajectory format mode",
    )
    parser.add_argument(
        "-c",
        default=10,
        dest="pig_count",
        choices=[10, 20, 30, 40, 50],
        type=int,
        help="Number of pigs",
    )
    parser.add_argument(
        "-r",
        default=1.0,
        dest="sample_rate",
        choices=[1.0, 2.0, 3.0, 4.0],
        type=float,
        help="Sample rate",
    )
    parser.add_argument(
        "-s",
        default=False,
        action="store_true",
        dest="save_out",
        help="Save model performance (flag only)",
    )
    parser.add_argument(
        "-l",
        default=False,
        action="store_true",
        dest="load",
        help="Load previous parameters (flag only)",
    )
    parser.add_argument(
        "--compare",
        default="none",
        dest="compare",
        choices=["none", "rmse", "mre", "r2"],
        help="Comparison mode",
    )

    parser.add_argument(
        "-m",
        default="rf",
        dest="model",
        choices=["rf", "gb", "lasso", "dummy"],
        help="Prediction model",
    )
    parser.add_argument(
        "-n",
        default=False,
        action="store_true",
        dest="nested_cv",
        help="Nested CV mode (flag only)",
    )
    parser.add_argument(
        "-t", default=3, dest="n_threads", type=int, help="Number of CPU threads"
    )
    parser.add_argument(
        "--outer_folds", default=3, dest="outer_folds", type=int, help="Outer CV folds"
    )

    parser.add_argument(
        "--tj", default="clod50e", dest="tj_model", help="Trajectory model (pred mode)"
    )
    parser.add_argument(
        "--series_count",
        default=100,
        dest="series_count",
        type=int,
        help="Number of series",
    )
    parser.add_argument(
        "--series_length",
        default=35,
        dest="series_length",
        type=int,
        help="Series length",
    )

    parser.add_argument(
        "--step_size",
        default=1.0,
        dest="step_size",
        type=float,
        help="Step size (days)",
    )
    parser.add_argument(
        "--horizon",
        default=14,
        dest="horizon",
        type=int,
        help="Forecast horizon (steps)",
    )
    parser.add_argument(
        "--frag_size",
        default=7,
        dest="frag_size",
        type=int,
        help="Fragment size (steps)",
    )
    parser.add_argument(
        "--min_pigs",
        default=1.0,
        dest="min_pigs",
        type=float,
        help="Minimum pigs ratio (float)",
    )
    args = vars(parser.parse_args())

    file_code = (
        "_" + str(args["pig_count"]) + "-" + str(args["sample_rate"]).replace(".", ",")
    )
    validation_type = "ncv" if args["nested_cv"] else "cv"
    model_name = validation_type + "_" + args["model"] + "_" + args["format_mode"]
    model_dir = (
        args["format_mode"] if args["format_mode"] != "pred" else args["tj_model"]
    )

    if args["compare"] != "none":
        model_comparison_average(metric=args["compare"], nested_cv=True)
        model_comparison(metric=args["compare"], nested_cv=True)
        sys.exit(0)

    print("Generating dataset...")
    dataset = generate_dataset(
        args["format_mode"],
        args["tj_model"],
        series_count=args["series_count"],
        pig_count=args["pig_count"],
        sample_rate=args["sample_rate"],
        series_length=args["series_length"],
        fold_count=args["outer_folds"],
        min_pigs=args["min_pigs"],
        ff_step_size=args["step_size"],
        ff_frag_size=args["frag_size"],
        ff_horizon=args["horizon"],
    )

    if args["load"]:
        with open(
            "metrics_forecast/" + model_dir + "/" + model_name + file_code + ".dat",
            "rb",
        ) as fh:
            loaded_model = pickle.load(fh)
        loaded_params = loaded_model["best_params"]
        print("Old scores:", loaded_model["eval_fold_rmse"])
    else:
        loaded_params = None

    std_pred = args["format_mode"] == "group"
    if args["nested_cv"]:
        print("\nNested cross-validation started...")
        cv_train = train_ncv(
            dataset,
            args["model"],
            params=loaded_params,
            n_threads=args["n_threads"],
            std_pred=std_pred,
        )
    else:
        print("\nCross-validation started...")
        cv_train = train_cv(dataset, args["model"], std_pred=std_pred)

    cv_test = test_cv(dataset, cv_train, std_pred=std_pred)
    cv_test["validation_fold_rmse"] = cv_train["validation_fold_rmse"]
    cv_test["validation_mean_rmse"] = cv_train["validation_mean_rmse"]

    if args["nested_cv"]:
        cv_test["best_params"] = cv_train["best_params"]
        cv_test["pred"] = cv_train["pred"]
        cv_test["true"] = cv_train["true"]
        if args["model"] != "lasso":
            cv_test["gs_results"] = cv_train["gs_results"]

    if args["format_mode"] == "group":
        print("RMSE fold scores:", cv_test["test_fold_rmse"])
        print("RMSE average score:", cv_test["test_mean_rmse"], "\n")
        print("Mean prediction RMSE fold scores:", cv_test["test_fold_rmse_meanpred"])
        print(
            "Mean prediction RMSE average score:",
            cv_test["test_mean_rmse_meanpred"],
            "\n",
        )
        print("STD prediction RMSE fold scores:", cv_test["test_fold_rmse_stdpred"])
        print(
            "STD prediction RMSE average score:",
            cv_test["test_mean_rmse_stdpred"],
            "\n",
        )
        print("Evaluation started - loading individual-level dataset...")
        dataset_indv = generate_dataset(
            "true",
            args["tj_model"],
            series_count=args["series_count"],
            pig_count=args["pig_count"],
            sample_rate=args["sample_rate"],
            series_length=args["series_length"],
            fold_count=args["outer_folds"],
            min_pigs=args["min_pigs"],
            ff_step_size=args["step_size"],
            ff_frag_size=args["frag_size"],
            ff_horizon=args["horizon"],
        )
        indv_true, indv_pred = estimate_individual_weights(cv_train, dataset_indv)
    else:
        indv_true, indv_pred = match_individual_weights(cv_train)

    eval_scores = get_metrics(indv_true, indv_pred)
    for metrics_key, metrics_values in eval_scores.items():
        cv_test["eval_fold_" + metrics_key] = metrics_values
        cv_test["eval_mean_" + metrics_key] = np.mean(metrics_values)

    if args["save_out"]:
        cv_test["args"] = args
        os.makedirs("metrics_forecast/" + model_dir, exist_ok=True)
        with open(
            "metrics_forecast/" + model_dir + "/" + model_name + file_code + ".dat",
            "wb",
        ) as fh:
            pickle.dump(cv_test, fh)


if __name__ == "__main__":
    main()
