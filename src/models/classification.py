"""
DONE:
- Figure out what metrics to compute and save once predicted values are
calculated with the predict() function
- Add a loop to run different models for each train-test split
- Update plotting section for new metrics
- move the params dictionary into the __init__ method of the class
- Update model specs for classification (loss)
- Update dictionary key names in the section that puts params_comb_dict's in the master list
- Update params dictionary with hyperparameters for Keras model
- Review and update command line arguments and their corresponding class parameters
- add code to test different thresholds
(see https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

TO DO:
- Update stop_instance function to allow for it to be run in the background
with threading (https://stackoverflow.com/questions/7168508/background-function-in-python)
- move s3 upload_file() segments into a function

DECIDED TO SKIP FOR NOW:
- Update validation data specification in the model fit() method to accept
multiple validation sets
(see https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras)
(THIS ALSO REQUIRES UPDATE OF PREPROCESSING FUNCTIONS)
"""

from collections import defaultdict
from datetime import datetime
from itertools import product
import logging
import os
from pathlib import Path
import platform
import sys
import time

import boto3
from botocore.exceptions import ClientError
from ec2_metadata import ec2_metadata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    callbacks,
    initializers,
    layers,
    losses,
    metrics,
    optimizers,
)
import tensorflow_addons as tfa

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from features.read_and_split_csvs import (
    month_counter,
    train_test_time_split,
    valid_date,
    valid_frac,
)
from models.bacc import BACC


class KerasClf:
    def __init__(
        self,
        curr_dt_time,
        startmonth,
        n_months_in_first_train_set,
        n_months_in_val_set,
        class_ratio,
        n_val_sets_in_one_month,
        chunksize,
        pipe_steps,
        frac=None,
        add_weather_features=False,
        scaler="s",
        effect_coding=False,
        add_princomps=0,
        add_interactions=False,
    ):
        self.curr_dt_time = curr_dt_time
        self.startmonth = startmonth
        self.n_months_in_first_train_set = n_months_in_first_train_set
        self.n_months_in_val_set = n_months_in_val_set
        self.class_ratio = class_ratio
        self.n_val_sets_in_one_month = n_val_sets_in_one_month
        self.chunksize = chunksize
        self.pipe_steps = pipe_steps
        self.frac = frac if frac is not None else 1.0
        self.add_weather_features = add_weather_features
        self.scaler = scaler
        self.effect_coding = effect_coding
        self.add_princomps = add_princomps
        self.add_interactions = add_interactions

        self.non_x_cols = [
            "shop_id",
            "item_id",
            "sale_date",
            "sid_shop_item_qty_sold_day",
        ]

        if self.n_val_sets_in_one_month > 1:
            raise NotImplementedError(
                "Model class currently does not support more than one validation set."
            )

        self.params = {
            "loss": ["binary_crossentropy"],
            "learning_rate": [0.001],
            "batch_size": [64],
            "epochs": [50],
            "threshold": [0.5],
        }
        assert all(
            [isinstance(v, list) for v in self.params.values()]
        ), "Values provided for each hyperparameter must be enclosed in a list."
        logging.info(
            "Model parameters used: "
            + ", ".join([f"{k}: {v}" for k, v in self.params.items()])
        )

        # determine if there is more than one combination of hyperparameters
        # if only one combination, set get_stats_ flag to True
        self.get_stats_ = (
            len(self.params[max(self.params, key=lambda x: len(self.params[x]))]) == 1
        )

        metric_fns = (balanced_accuracy_score, recall_score, precision_score, f1_score)
        metric_names = ("balanced acc", "recall", "precision", "F1 score")
        self.clf_metrics = dict(zip(metric_names, metric_fns))

    def gridsearch_wfv(self):

        self.all_params_combs = list()

        for train, test, train_counter, last_val_set_ind in train_test_time_split(
            self.startmonth,
            self.n_months_in_first_train_set,
            self.n_months_in_val_set,
            self.class_ratio,
            n_val_sets_in_one_month=self.n_val_sets_in_one_month,
            chunksize=self.chunksize,
            frac=self.frac,
            add_weather_features=self.add_weather_features,
        ):

            if self.get_stats_:
                get_stats = last_val_set_ind
            else:
                get_stats = False

            self.train_counter = train_counter

            # the statement below points the test variable to the first dataframe
            # in the test list, so multiple validation sets are not currently
            # supported
            test = test[0]
            # self.cat_col_cats = [
            #     np.sort(train[col].unique()) for col in train if col.startswith("cat")
            # ]
            self.cat_col_cats = [
                np.sort(np.unique(np.hstack((train[col].unique(), test[col].unique()))))
                for col in train.columns.union(test.columns)
                if col.startswith("cat")
            ]

            self.train_y = train["sid_shop_item_qty_sold_day"].to_numpy()
            self.test_y = test["sid_shop_item_qty_sold_day"].to_numpy()
            self.preprocess_features(train, test).fit_and_eval(get_stats)

        # calculate average metric values across all folds
        for params_comb_dict in self.all_params_combs:

            for metric in ("recall", "precision", "F1_score", "balanced_acc"):
                params_comb_dict[f"avg_{metric}_"] = np.mean(
                    params_comb_dict[f"{metric}_list_"]
                )
                params_comb_dict[f"avg_{metric}_thresh"] = np.mean(
                    params_comb_dict[f"{metric}_thresh_list_"]
                )

        best_params = min(self.all_params_combs, key=lambda x: x["avg_F1_score_"])
        self.best_score_ = best_params["avg_F1_score_"]
        # remove non-parameter key-values from self.best_params (i.e., rmse_list_ and avg_rmse_, etc.)
        self.best_params_ = {k: v for k, v in best_params.items() if k in self.params}

        # save list of parameter-result dictionaries to dataframe and then to CSV
        if self.all_params_combs:
            all_params_combs_df = pd.DataFrame(self.all_params_combs)
            output_csv = "all_params_combs.csv"
            all_params_combs_df.to_csv(output_csv, index=False)

            try:
                key = f"keras_all_params_combs_{self.curr_dt_time}.csv"
                # global s3_client
                s3_client = boto3.client("s3")
                response = s3_client.upload_file(output_csv, "sales-demand-data", key)
                logging.info(
                    "Name of CSV uploaded to S3 and containing all parameter combinations "
                    f"and results is: {key}"
                )
            except ClientError as e:
                logging.exception(
                    "CSV file with Keras parameter combinations and results was not copied to S3."
                )

        else:
            logging.debug(
                "List of parameter-result dictionaries is empty and was not converted to CSV!"
            )

    def preprocess_features(self, train_X, test_X):
        train_X.drop(self.non_x_cols, axis=1, inplace=True)
        train_X.reset_index(drop=True, inplace=True)
        test_X.drop(self.non_x_cols, axis=1, inplace=True)
        test_X.reset_index(drop=True, inplace=True)

        train_pcs_df = None
        test_pcs_df = None
        if self.add_princomps > 0:  # create additional principal component columns
            train_pcs_df, test_pcs_df = self.get_pc_cols(
                train_X[[col for col in train_X if not col.startswith("cat")]],
                test_X[[col for col in test_X if not col.startswith("cat")]],
                self.add_princomps,
            )

        train_interaction_df_list = list()
        test_interaction_df_list = list()
        if self.add_interactions:
            for col in [
                "d_days_after_holiday",
                "d_days_to_holiday",
                "d_eurrub",
                "d_usdrub",
            ]:
                train_interaction_df_list.append(
                    train_X[[cl for cl in train_X if "_7d" in cl]]
                    .multiply(train_X[col], axis=0)
                    .rename(
                        columns={
                            k: f"{k}_{col}"
                            for k in [cl for cl in train_X if "_7d" in cl]
                        }
                    )
                )
                test_interaction_df_list.append(
                    test_X[[cl for cl in test_X if "_7d" in cl]]
                    .multiply(test_X[col], axis=0)
                    .rename(
                        columns={
                            k: f"{k}_{col}"
                            for k in [cl for cl in test_X if "_7d" in cl]
                        }
                    )
                )

        if self.add_princomps > 0 or self.add_interactions:
            train_X = pd.concat(
                [
                    train_X[[col for col in train_X if col.startswith("cat")]],
                    train_pcs_df,
                    *train_interaction_df_list,
                ],
                axis=1,
            )
            test_X = pd.concat(
                [
                    test_X[[col for col in test_X if col.startswith("cat")]],
                    test_pcs_df,
                    *test_interaction_df_list,
                ],
                axis=1,
            )
            del (
                train_pcs_df,
                test_pcs_df,
                train_interaction_df_list,
                test_interaction_df_list,
            )

        rng = np.random.RandomState(304)
        qt = QuantileTransformer(output_distribution="normal", random_state=rng)

        if self.scaler == "s":
            sc = StandardScaler()
        elif self.scaler == "m":
            sc = MinMaxScaler()
        elif self.scaler == "r":
            sc = RobustScaler()

        numeric_features = [col for col in train_X if not col.startswith("cat")]
        categorical_features = [
            col
            for col in train_X.columns.union(test_X.columns)
            if col.startswith("cat")
        ]

        if self.pipe_steps == 0:
            steps = [("qt_normal", qt)]
        elif self.pipe_steps == 1:
            steps = [("scaler", sc)]
        elif self.pipe_steps == 2:
            steps = [("qt_normal", qt), ("scaler", sc)]
        numeric_transformer = Pipeline(steps=steps)

        categorical_transformer = OneHotEncoder(
            categories=self.cat_col_cats, drop="first", dtype="uint8", sparse=False,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.train_X = pd.DataFrame(
            preprocessor.fit_transform(train_X),
            columns=numeric_features
            + preprocessor.named_transformers_["cat"]
            .get_feature_names(categorical_features)
            .tolist(),
            dtype="float32",
        )
        self.test_X = pd.DataFrame(
            preprocessor.transform(test_X),
            columns=numeric_features
            + preprocessor.named_transformers_["cat"]
            .get_feature_names(categorical_features)
            .tolist(),
            dtype="float32",
        )
        if self.effect_coding:
            self.train_X[
                [col for col in self.train_X if col.startswith("cat")]
            ] = self.train_X[
                [col for col in self.train_X if col.startswith("cat")]
            ].replace(
                0.0, -1.0
            )
            self.test_X[
                [col for col in self.test_X if col.startswith("cat")]
            ] = self.test_X[
                [col for col in self.test_X if col.startswith("cat")]
            ].replace(
                0.0, -1.0
            )

        return self

    def get_pc_cols(self, train_X, test_X, n_components):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)

        scaled_train_X = scaler.fit_transform(train_X)
        pca_transformed_train_X = pca.fit_transform(scaled_train_X)
        del scaled_train_X
        # pca_transformed_train_X = np.log(np.abs(pca_transformed_train_X))

        scaled_test_X = scaler.transform(test_X)
        pca_transformed_test_X = pca.transform(scaled_test_X)
        del scaled_test_X
        # pca_transformed_test_X = np.log(np.abs(pca_transformed_test_X))

        pca_transformed_train_X = pd.DataFrame(
            pca_transformed_train_X, columns=[f"pc_{c}" for c in range(n_components)]
        )
        pca_transformed_test_X = pd.DataFrame(
            pca_transformed_test_X, columns=[f"pc_{c}" for c in range(n_components)]
        )

        return pca_transformed_train_X, pca_transformed_test_X

    def fit_and_eval(self, get_stats):

        for params_comb_counter, params_comb_dict in enumerate(
            (
                dict(zip(self.params.keys(), v))
                for v in list(product(*list(self.params.values())))
            ),
            1,
        ):

            # self.params_comb_dict = params_comb_dict.copy()
            # self.params_comb_dict["precision_list_"] = list()
            # self.params_comb_dict["precision_thresh_list_"] = list()
            # self.params_comb_dict["recall_list_"] = list()
            # self.params_comb_dict["recall_thresh_list_"] = list()
            # self.params_comb_dict["F1_score_list_"] = list()
            # self.params_comb_dict["F1_score_thresh_list_"] = list()
            # self.params_comb_dict["balanced_acc_list_"] = list()
            # self.params_comb_dict["balanced_acc_thresh_list_"] = list()
            # self.params_comb_dict["fit_times_list_"] = list()
            # self.params_comb_dict["counter_"] = list()
            self.params_comb_dict = defaultdict(list, params_comb_dict)

            print(f"Shape of train_X is {self.train_X.shape}")

            try:
                self.model = keras.Sequential()

                layer_size = (
                    2 ** (round(np.log(self.train_X.shape[1]) / np.log(2))) // 2
                )

                ki = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123)
                self.model.add(
                    layers.Dense(
                        layer_size,
                        kernel_initializer=ki,
                        bias_initializer="zeros",
                        input_shape=(self.train_X.shape[1],),
                        activation="relu",
                    )
                )
                # self.model.add(layers.BatchNormalization())
                # self.model.add(layers.Dropout(0.2))
                self.model.add(layers.Dense(layer_size, activation="relu"))
                # self.model.add(layers.Dropout(0.2))
                # self.model.add(layers.Dense(64, activation="relu"))
                # self.model.add(
                #     layers.Dense(
                #         layer_size,
                #         kernel_initializer=ki,
                #         bias_initializer="zeros",
                #         input_shape=(self.train_X.shape[1],),
                #         activation="relu",
                #     )
                # )
                # self.model.add(layers.Dense(layer_size // 2, activation="relu"))
                # self.model.add(
                #     layers.Activation("exponential")
                # )  # need to specify number of nodes?
                # self.model.add(layers.BatchNormalization())
                self.model.add(layers.Dense(1, activation="sigmoid"))
                # self.model.add(layers.Dense(1))

                opt = optimizers.Adam(learning_rate=params_comb_dict["learning_rate"])
                # initial_learning_rate = 0.1
                # lr_schedule = optimizers.schedules.ExponentialDecay(
                #     initial_learning_rate,
                #     decay_steps=100_000,
                #     decay_rate=0.96,
                #     staircase=True
                # )
                # opt = optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
                # opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
                callback = callbacks.EarlyStopping(
                    monitor="val_f1_score",
                    patience=10,
                    mode="max",
                    restore_best_weights=True,
                )
                self.model.compile(
                    loss=params_comb_dict["loss"],
                    optimizer=opt,
                    metrics=[
                        metrics.Recall(
                            thresholds=params_comb_dict["threshold"],
                            name="recall_score",
                        ),
                        metrics.Precision(
                            thresholds=params_comb_dict["threshold"],
                            name="precision_score",
                        ),
                        tfa.metrics.F1Score(
                            num_classes=1,
                            threshold=params_comb_dict["threshold"],
                            name="f1_score",
                        ),
                        BACC(
                            thresholds=params_comb_dict["threshold"],
                            name="balanced_acc",
                        ),
                    ],
                )

                print(self.model.summary())

            except Exception:
                logging.exception("Exception occurred while initializing Keras model.")
                sys.exit(1)

            # estimate baseline model
            dummy_clf = DummyClassifier(strategy="constant", constant=1)
            dummy_clf.fit(self.train_X, self.train_y)
            dummy_clf_y_pred = dummy_clf.predict(self.train_X)

            for name, metric_fn in self.clf_metrics.items():
                dummy_clf_metric = metric_fn(self.train_y, dummy_clf_y_pred,)
                print(
                    f"{name} for dummy classifier always predicting the "
                    f"minority class is {dummy_clf_metric}."
                )

            start_time = time.perf_counter()
            history = self.model.fit(
                x=self.train_X,  # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs)
                y=self.train_y,
                validation_data=(self.test_X, self.test_y,),
                epochs=params_comb_dict["epochs"],
                batch_size=params_comb_dict["batch_size"],
                shuffle=True,
                callbacks=[callback],
                verbose=2,
            )

            self.params_comb_dict["fit_times_list_"].append(
                time.perf_counter() - start_time
            )

            pred_probs = self.predict(params_comb_dict["batch_size"])

            for name, metric_fn in self.clf_metrics.items():
                clf_metric, best_threshold = self.best_thresh(metric_fn, pred_probs)

                self.params_comb_dict[f"{name.replace(' ','_')}_list_"].append(
                    clf_metric
                )
                self.params_comb_dict[f"{name.replace(' ','_')}_thresh_list_"].append(
                    best_threshold
                )

            combined_counter = "_" + "_".join(
                [str(params_comb_counter).zfill(2), str(self.train_counter).zfill(2)]
            )
            self.params_comb_dict["counter_"].append(combined_counter)

            if get_stats:

                s3_client = boto3.client("s3")
                for m in (
                    "recall_score",
                    "precision_score",
                    "f1_score",
                    "balanced_acc",
                ):
                    try:
                        plt.plot(history.history[m])
                        plt.plot(history.history[f"val_{m}"])
                        plt.title(f"Model {m.replace('_',' ').title()} Metric")
                        plt.ylabel(f"{m.replace('_',' ').title()}")
                        plt.xlabel("Epoch")
                        n_epochs = len(history.history[m])
                        plt.xticks(
                            np.arange(0, n_epochs, step=1),
                            np.arange(1, n_epochs + 1, step=1),
                        )
                        plt.legend(["Train", "Test"], loc="upper right")

                        png_fname = combined_counter + f"_{m}.png"
                        plt.savefig(png_fname)
                        plt.clf()  # clear current figure
                        key = f"{combined_counter}_{self.curr_dt_time}_{m}.png"
                        response = s3_client.upload_file(
                            png_fname, "sales-demand-data", key
                        )

                    except ClientError as e:
                        logging.exception(
                            f"PNG file with learning curve for {combined_counter} "
                            f"parameter-fold combination "
                            f"and {m} metric was not copied to S3."
                        )

            # (initially, check if list contains anything or is empty)
            # first, check if the master list of dictionaries contains a
            # dictionary with same values of hyperparameters
            # tuple(map(params_comb_dict.get, ('a', 'b', 'c'))) ==
            # tuple(map(DICTIONARY_IN_THE_MASTER_LIST.get, ('a', 'b', 'c')))
            # if it does, append results to key-values of results in that dictionary
            # if it does not, append that dictionary to the master list
            #
            # loop over dictionaries in the list, with enumerate():
            # master_list = [{'a': 100,'b': 200,'c':[0.4]}, {'a': 100, 'b': 300, 'c':[0.5]}, {'a': 100,'b': 200,'c':[0.7]}]
            # test_dict = {'a':100, 'b':200, 'c':[0.6]}

            if not self.all_params_combs:
                self.all_params_combs.append(dict(self.params_comb_dict))
            else:
                found = False
                for idx, d in enumerate(self.all_params_combs):
                    if tuple(map(d.get, tuple(self.params.keys()))) == tuple(
                        map(self.params_comb_dict.get, tuple(self.params.keys()))
                    ):
                        for result in (
                            "precision_list_",
                            "precision_thresh_list_",
                            "recall_list",
                            "recall_thresh_list_",
                            "F1_score_list_",
                            "F1_score_thresh_list_",
                            "balanced_acc_list_",
                            "balanced_acc_thresh_list_",
                            "fit_times_list_",
                            "counter_",
                        ):
                            self.all_params_combs[idx][result].extend(
                                self.params_comb_dict[result]
                            )
                        found = True
                        break
                if not found:
                    self.all_params_combs.append(dict(self.params_comb_dict))

            dict_key_val_str = ", ".join(
                [f"{k}: {v}" for k, v in self.params_comb_dict.items()]
            )
            logging.debug(
                f"params_comb_dict at the end of iteration no. {params_comb_counter} "
                f"over hyperparameter combinations for train set no. {self.train_counter} is "
                f"{dict_key_val_str}."
            )

    def predict(self, batch_size):
        return self.model.predict(self.test_X, batch_size=batch_size,)

    def best_thresh(self, metric_fn, pred_probs, thresholds=np.arange(0, 1, 0.005)):

        # evaluate each threshold
        scores = [
            metric_fn(self.test_y, (pred_probs >= t).astype("int8"),)
            for t in thresholds
        ]
        # get best threshold and score for that threshold
        ix = np.argmax(scores)
        best_threshold = thresholds[ix]
        clf_metric = scores[ix]

        # clf_metric = metric_fn(
        #     self.test_y,
        #     (self.model.predict(
        #         self.test_X, batch_size=params_comb_dict["batch_size"],
        #     ) > 0.5).astype('int8'),
        # )

        return clf_metric, best_threshold


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "startmonth",
        metavar="<startmonth>",
        help="first (earliest) month of data to be used, format: YY-MM",
        type=valid_date,
    )
    parser.add_argument(
        "n_months_in_first_train_set",
        metavar="<n_months_in_first_train_set>",
        help="number of months to be used in first train set during walk-forward validation",
        type=int,
    )
    parser.add_argument(
        "n_months_in_val_set",
        metavar="<n_months_in_val_set>",
        help="number of months to be used in each validation set during walk-forward validation",
        type=int,
    )
    parser.add_argument(
        "class_ratio",
        metavar="<class_ratio>",
        help=(
            "desired ratio of the number of samples in the minority class over "
            "the number of samples in the majority class after resampling"
        ),
        type=valid_frac,
    )
    parser.add_argument(
        "--n_val_sets_in_one_month",
        "-v",
        help="number of validation sets to be used in validating one model (default: 1)",
        default="1",
        type=int,
    )
    parser.add_argument(
        "--chunksize",
        "-z",
        help="chunksize (number of rows) for read_csv(), default is 1,000",
        default="1000",
        type=int,
    )
    parser.add_argument(
        "--frac",
        "-f",
        help="fraction of rows to sample in validation data (default is 1.0 if omitted)",
        default="1.0",
        type=valid_frac,
    )
    parser.add_argument(
        "--weather",
        "-w",
        help="whether to incorporate weather features (if included) or not (if not included)",
        default=False,
        action="store_true",
    )

    scaler_group = parser.add_argument_group(
        "scaler", "Arguments related to feature scaling"
    )

    scaler_group.add_argument(
        "--pipe_steps",
        "-p",
        required=True,
        help="whether to normalize only (0), scale only (1), or normalize and scale (2) the features",
        type=int,
        choices=[0, 1, 2],
    )

    scaler_group.add_argument(
        "--scaler",
        "-s",
        help="whether to apply StandardScaler (s) (default), MinMaxScaler (m), or RobustScaler (r)",
        default="s",
        choices=["s", "m", "r"],
    )

    parser.add_argument(
        "--effect_coding",
        "-e",
        help="whether use -1 and 1 for binary inputs (if included) or 0 and 1 (if not included)",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--add_princomps",
        "-c",
        help="create specified number of additional principal component features (if included), or not (if not included)",
        default="0",
        type=int,
    )

    parser.add_argument(
        "--add_interactions",
        "-t",
        help="create interaction features (if included) or not (if not included)",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    if month_counter(args.startmonth) - args.n_months_in_first_train_set + 1 <= 0:
        raise argparse.ArgumentError(
            "The provided combination of start month and number of months in "
            "first train set is invalid - either not enough months exist to "
            "allow for the provided length of train period, or no months "
            "remain for any validation period."
        )
    elif (
        month_counter(args.startmonth) - args.n_months_in_first_train_set + 1
    ) < args.n_months_in_val_set:
        raise argparse.ArgumentError(
            "The provided combination of start month and number of months in "
            "first train set does not allow for the provided number of months "
            "in validation set."
        )

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("logs")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    curr_dt_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_fname = f"logging_{curr_dt_time}_classification_model.log"
    log_path = log_dir.joinpath(log_fname)

    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format=fmt,
        datefmt=datefmt,
        filename=log_path,
    )

    # statements to suppress irrelevant logging by boto3-related libraries
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("awswrangler").setLevel(logging.DEBUG)

    # # Check if code is being run on EC2 instance (vs locally)
    my_user = os.environ.get("USER")
    is_aws = True if "ec2" in my_user else False
    # Log EC2 instance name and type metadata
    if is_aws:
        instance_metadata = dict()
        instance_metadata["EC2 instance ID"] = ec2_metadata.instance_id
        instance_metadata["EC2 instance type"] = ec2_metadata.instance_type
        instance_metadata["EC2 instance public hostname"] = ec2_metadata.public_hostname

        f = lambda x: ": ".join(x)
        r = list(map(f, list(instance_metadata.items())))
        nl = "\n" + " " * 55
        logging.info(
            f"Script is running on EC2 instance with the following metadata: "
            f"{nl}{nl.join(r)}"
        )
    else:
        logging.info("Script is running on local machine, not on EC2 instance.")

    from matplotlib import __version__ as mpl_version
    from sklearn import __version__ as sk_version
    from tensorflow import __version__ as tf_version

    logging.info(f"The Python version is {platform.python_version()}.")
    logging.info(f"The pandas version is {pd.__version__}.")
    logging.info(f"The numpy version is {np.__version__}.")
    logging.info(f"The matplotlib version {mpl_version}.")
    logging.info(f"The sklearn version is {sk_version}.")
    logging.info(f"The tensorflow version is {tf_version}.")

    s3_client = boto3.client("s3")

    logging.info(
        f"Running Keras classification model with "
        f"startmonth: {args.startmonth}, n_months_in_first_train_set: {args.n_months_in_first_train_set}, "
        f"n_months_in_val_set: {args.n_months_in_val_set}, frac: {args.frac}, "
        f"weather features: {args.weather}, chunksize: {args.chunksize}, "
        f"class_ratio: {args.class_ratio}, "
        f"n_val_sets_in_one_month: {args.n_val_sets_in_one_month}, "
        f"pipe_steps: {args.pipe_steps}, scaler: {args.scaler}, effect_coding: {args.effect_coding}, "
        f"add_princomps: {args.add_princomps}, and add_interactions: {args.add_interactions}...",
    )

    model = KerasClf(
        curr_dt_time,
        args.startmonth,
        args.n_months_in_first_train_set,
        args.n_months_in_val_set,
        args.class_ratio,
        args.n_val_sets_in_one_month,
        args.chunksize,
        args.pipe_steps,
        frac=args.frac,
        add_weather_features=args.weather,
        scaler=args.scaler,
        effect_coding=args.effect_coding,
        add_princomps=args.add_princomps,
        add_interactions=args.add_interactions,
    )
    model.gridsearch_wfv()

    # copy log file to S3 bucket
    try:
        response = s3_client.upload_file(
            f"./logs/{log_fname}", "my-ec2-logs", log_fname
        )
    except ClientError as e:
        logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()
