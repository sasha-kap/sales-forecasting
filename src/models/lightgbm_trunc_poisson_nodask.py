# to do:
# - calculate train score
# - learning curve plot (vary training examples used and examine the effect on train and validation set scores)
#       - https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#       - add sampling in the model code
# - finish creating lists of hyperparameter values
#       - https://neptune.ai/blog/lightgbm-parameters-guide
# - make sure log is not copied to S3 if the program crashes
# - check if LightGBM and Dask logging needs to be disabled - LightGBM probably sends all output to stdout
# - plot learning curves? https://stackoverflow.com/questions/60132246/how-to-plot-the-learning-curves-in-lightgbm-and-python
# - write docstrings and header
# - best_iteration - needed? (can be used while saving model)

# "s3://sales-demand-data/parquet_dataset/"

# save_model(filename, num_iteration=None, start_iteration=0, importance_type='split')[source]
# Save Booster to file.
#
# Parameters
# filename (string or pathlib.Path) – Filename to save Booster.
# num_iteration (int or None, optional (default=None)) – Index of the iteration that should be saved. If None, if the best iteration exists, it is saved; otherwise, all iterations are saved. If <= 0, all iterations are saved.
# start_iteration (int, optional (default=0)) – Start index of the iteration that should be saved.
# importance_type (string, optional (default="split")) – What type of feature importance should be saved. If “split”, result contains numbers of times the feature is used in a model. If “gain”, result contains total gains of splits which use the feature.
#
# Returns
# self – Returns self.
#
# Return type
# Booster

import argparse
from datetime import datetime, timedelta
from itertools import product
import logging
import os
from pathlib import Path
import platform
from statistics import mean
import sys
import time

import boto3
from botocore.exceptions import ClientError
import dask as dsk
from dask import array as da, dataframe as dd
from dask.distributed import Client, LocalCluster, performance_report, wait
# from dask_ml.metrics.regression import mean_squared_error
from dateutil.relativedelta import relativedelta
from ec2_metadata import ec2_metadata
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def month_counter(fm, LAST_DAY_OF_TRAIN_PRD=(2015, 10, 31)):
    """Calculate number of months (i.e. month boundaries) between the first
    month of train period and the end month of validation period.

    Parameters:
    -----------
    fm : datetime
        First day of first month of train period

    Returns:
    --------
    Number of months between first month of train period and end month of validation period
    """
    return (
        (datetime(*LAST_DAY_OF_TRAIN_PRD).year - fm.year) * 12
        + datetime(*LAST_DAY_OF_TRAIN_PRD).month
        - fm.month
    )


def calc_rmse(y_true, y_pred, get_stats):
    if get_stats:
        pred_stats_to_csv(y_true, y_pred)
    return mean_squared_error(y_true, y_pred, squared=False)


def pred_stats_to_csv(y_true, y_pred, output_csv="pred_value_stats.csv"):
    y_true_df = pd.DataFrame(y_true, columns=["y_true"])
    y_pred_df = pd.DataFrame(
        y_pred, columns=["y_pred"], index=y_true_df.index
    )  # convert Dask array to Pandas DF
    full_df = pd.concat(
        [y_true_df, y_pred_df], axis=1
    )  # join actual and predicted values
    del y_true_df
    del y_pred_df

    stats_df = (
        full_df.groupby("y_true")
        .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .droplevel(level=0, axis=1)
        .reset_index()
    )
    stats_df.to_csv(output_csv, index=False)
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(output_csv, "sales-demand-data", output_csv)
        logging.info(
            "CSV file with descriptive stats of predicted values "
            "successfully copied to S3."
        )
    except ClientError as e:
        logging.exception(
            "CSV file with descriptive stats of predicted values "
            "was not copied to S3."
        )


def calc_monthly_rmse(y_true_w_id_cols, y_pred):
    y_true_df = y_true_w_id_cols.copy()  # convert Dask dataframe to Pandas DF
    y_pred_df = pd.DataFrame(
        y_pred, columns=["y_pred"], index=y_true_df.index
    )  # convert Dask array to Pandas DF
    full_df = pd.concat(
        [y_true_df, y_pred_df], axis=1
    )  # join actual and predicted values
    del y_true_df
    del y_pred_df
    # calculate sums of actual and predicted values by shop-item-month
    # the code below assumes that same calendar month does not appear across multiple years in validation set
    shop_item_month_df = (
        full_df.groupby([full_df.index.month, "shop_id", "item_id"])
        .agg("sum")
        .reset_index()
    )
    # calculate RMSE for each month and then take the average of monthly values
    return (
        shop_item_month_df.groupby("sale_date")
        .apply(
            lambda x: np.sqrt(
                np.average((x["sid_shop_item_qty_sold_day"] - x["y_pred"]) ** 2)
            )
        )
        .mean()
    )
    # calculate monthly rmse
    # return np.sqrt(np.average((shop_item_df['sid_shop_item_qty_sold_day'] - shop_item_df['y_pred'])**2))


def valid_frac(s):
    """Convert command-line fraction argument to float value.

    Parameters:
    -----------
    s : str
        Command-line argument for fraction of rows to sample

    Returns:
    --------
    float

    Raises:
    -------
    ArgumentTypeError
        if input string cannot be converted to float or if the resulting float
        is a negative value
    """
    try:
        f = float(s)
    except ValueError:
        msg = f"Not a valid fraction value: {s}. Enter a value between 0.0 and 1.0."
        raise argparse.ArgumentTypeError(msg)
    else:
        if f < 0:
            msg = f"{f} is an invalid positive float value. Enter a value between 0.0 and 1.0."
            raise argparse.ArgumentTypeError(msg)
        return f


def valid_date(s):
    """Convert command-line date argument to YY-MM datetime value.

    Parameters:
    -----------
    s : str
        Command-line argument for first month of data to be used

    Returns:
    --------
    datetime object (format: %y-%m)

    Raises:
    -------
    ArgumentTypeError
        if input string cannot be parsed according to %y-%m strptime format
    """
    try:
        return datetime.strptime(s, "%y-%m")
    except ValueError:
        msg = f"Not a valid date: {s}."
        raise argparse.ArgumentTypeError(msg)


# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# Deal with Over-fitting
# Use small max_bin
# Use small num_leaves
# Use min_data_in_leaf and min_sum_hessian_in_leaf
# Use bagging by set bagging_fraction and bagging_freq
# Use feature sub-sampling by set feature_fraction
# Use bigger training data
# Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
# Try max_depth to avoid growing deep tree
# Try extra_trees
# Try increasing path_smooth

# boosting_type = 'gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100,
# subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
# min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
# colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1,
# silent=True, importance_type='split', client=None, **kwargs

# param_names = ('num','let')
# for params_dict in (dict(zip(param_names,v)) for v in product([1,2,3],('a','b'))):
#     print(params_dict)
# convert the generator expression into a list that's saved to instance variable,
# make the loop numbered (with enumerate),
# update the instance variable (one dictionary at a time) while looping over sets of hyperparameters,
# at the end, convert the full list of dictionaries into a table that can be exported to CSV

# params = {
#     'objective' : ['tweedie', 'regression', 'regression_l1', 'poisson'],
#     'metric' : ['rmse'], # tweedie, poisson, rmse, l1, l2
#     'boosting_type' : ['gdbt', 'dart', 'rf'],
#     # num_leaves - sets the maximum number of nodes per tree. Decrease num_leaves to reduce training time.
#     'num_leaves' : [31, 62, 124], # max number of leaves in one tree, 31 is default
#     # max_depth - this parameter is an integer that controls the maximum distance between the root node of each tree and a leaf node. Decrease max_depth to reduce training time. -1 is default (no limit)
#     'max_depth' : [5, 10],
#     # num_iterations - number of boosting iterations, default is 100 (alias: n_estimators)
#     'num_iterations' : [50, 75, 100],
#     # min_child_samples - minimal number of data in one leaf. Can be used to deal with over-fitting, 20 is default, aka min_data_in_leaf
#     'min_child_samples' : [2, 100, 1000],
#     # learning_rate: default is 0.1
#     'learning_rate' : [0.1, 0.05, 0.01],
#     # max_bin - max number of bins that feature values will be bucketed in, use larger value for better accuracy (may be slower), smaller value helps deal with over-fitting, default is 255
#     'max_bin' : [128, 255],
#     # subsample_for_bin - number of data that sampled to construct feature discrete bins, default: 200000
#     'subsample_for_bin' : [200000],
#     # bagging_fraction - for random selection of part of the data, without resampling, default: 1.0, constraints: 0.0 < bagging_fraction <= 1.0
#     'bagging_fraction' : [1.0],
#     # bagging_freq - frequency for bagging, 0 means disable bagging; k means perform bagging at every k iteration. default: 0
#     'bagging_freq' : [0],
#     # feature_fraction - LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0, default: 1.0, constraints: 0.0 < feature_fraction <= 1.0
#     # colsample_bytree (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.
#     'colsample_bytree' : [1.0]
# }
params = {
    # "objective": ["tweedie"],
    "metric": ["rmse"],  # tweedie, poisson, rmse, l1, l2
    "boosting_type": ["gbdt"],
    # num_leaves - sets the maximum number of nodes per tree. Decrease num_leaves to reduce training time.
    "num_leaves": [28],  # max number of leaves in one tree, 31 is default
    # max_depth - this parameter is an integer that controls the maximum distance between the root node of each tree and a leaf node. Decrease max_depth to reduce training time. -1 is default (no limit)
    # To keep in mind: "Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31)."
    "max_depth": [5],
    # num_iterations - number of boosting iterations, default is 100 (alias: n_estimators)
    "num_iterations": [200, 500, 1000],
    # min_child_samples - minimal number of data in one leaf. Can be used to deal with over-fitting, 20 is default, aka min_data_in_leaf
    "min_child_samples": [200],
    # learning_rate: default is 0.1
    "learning_rate": [0.001, 0.01],
    # max_bin - max number of bins that feature values will be bucketed in, use larger value for better accuracy (may be slower), smaller value helps deal with over-fitting, default is 255
    "max_bin": [128, 255],
    # subsample_for_bin - number of data that sampled to construct feature discrete bins, default: 200000
    # "subsample_for_bin": [200000],
    # bagging_fraction - for random selection of part of the data, without resampling, default: 1.0, constraints: 0.0 < bagging_fraction <= 1.0
    "bagging_fraction": [0.6],
    # bagging_freq - frequency for bagging, 0 means disable bagging; k means perform bagging at every k iteration. default: 0
    "bagging_freq": [0],
    # feature_fraction - LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0, default: 1.0, constraints: 0.0 < feature_fraction <= 1.0
    # colsample_bytree (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.
    "colsample_bytree": [1.],
    # "tweedie_variance_power": [1.4],
    "weight_for_zeros": [1.],
}
# additional parameters
# pre_partition: https://lightgbm.readthedocs.io/en/latest/Parameters.html#pre_partition
# default: false
# used for distributed learning (excluding the feature_parallel mode)
# true if training data are pre-partitioned, and different machines use different partitions

# tweedie_variance_power: https://lightgbm.readthedocs.io/en/latest/Parameters.html#tweedie_variance_power
# default: 1.5, constraints: 1.0 <= tweedie_variance_power < 2.0
# used only in tweedie regression application
# used to control the variance of the tweedie distribution
# set this closer to 2 to shift towards a Gamma distribution
# set this closer to 1 to shift towards a Poisson distribution

# poisson_max_delta_step: https://lightgbm.readthedocs.io/en/latest/Parameters.html#poisson_max_delta_step
# default: 0.7, constraints: poisson_max_delta_step > 0.0
# used only in poisson regression application
# parameter for Poisson regression to safeguard optimization

# distributed learning
# num_threads: https://lightgbm.readthedocs.io/en/latest/Parameters.html#num_threads
# number of threads for LightGBM, default: 0
# for the best speed, set this to the number of real CPU cores, not the number of threads (most CPUs use hyper-threading to generate 2 threads per CPU core)
# for distributed learning, do not use all CPU cores because this will cause poor performance for the network communication

# n_jobs (int, optional (default=-1)) – Number of parallel threads.
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.DaskLGBMRegressor.html#lightgbm.DaskLGBMRegressor

# parameters specific to objective
# lambda_l1 - L1 regularization, default: 0.0, constraints: lambda_l1 >= 0.0
# lambda_l2 - L2 regularization, default: 0.0, constraints: lambda_l2 >= 0.0
# for DaskLGBMRegressor: reg_alpha – L1 regularization term on weights, reg_lambda  – L2 regularization term on weights.

# b = {'objective' : ['tweedie', 'regression_l1', 'poisson'], 'boosting_type' : ['gdbt', 'dart', 'rf']}
# # >>> list(product(*list(b.values())))
# [('tweedie', 'gdbt'), ('tweedie', 'dart'), ('tweedie', 'rf'), ('regression_l1', 'gdbt'), ('regression_l1', 'dart'), ('regression_l1', 'rf'), ('poisson', 'gdbt'), ('poisson', 'dart'), ('poisson', 'rf')]
#
# [dict(zip(b.keys(), v)) for v in list(product(*list(b.values())))]


class LightGBMDaskLocal:
    # https://github.com/Nixtla/mlforecast/blob/main/nbs/distributed.forecast.ipynb
    """
    persist call: data = self.client.persist(data)
    (assignment replaces old lazy array, as persist does not change the
    input in-place)

    To reduce the risk of hitting memory limits,
    consider restarting each worker process before running any data loading or training code.
    self.client.restart()
        - This function will restart each of the worker processes, clearing out anything
        they’re holding in memory. This function does NOT restart the actual machines of
        your cluster, so it runs very quickly.
        - should the workers just be killed regardless of whether the whole process
        was successful or unsuccessful (sort of a clean up action)? can restarting
        be that cleanup action?

    loop over hyperparameter values (method that accepts hyperparameters as a dictionary -
        initializes self.model = DaskLGBMRegressor() with each set of parameters and
        calls the method that loops over )
    loop over train-valdation sets
    run model's fit method and compute predicted values and RMSE
    """

    def __init__(
        self,
        curr_dt_time,
        n_workers,
        s3_path,
        startmonth,
        n_months_in_first_train_set,
        n_months_in_val_set,
        frac=None,
    ):
        self.curr_dt_time = curr_dt_time
        self.startmonth = startmonth
        self.n_months_in_first_train_set = n_months_in_first_train_set
        self.n_months_in_val_set = n_months_in_val_set
        self.frac = frac if frac is not None else 1.0

        cluster = LocalCluster(n_workers=n_workers)
        self.client = Client(cluster)
        self.client.wait_for_workers(n_workers)
        print(f"***VIEW THE DASHBOARD HERE***: {cluster.dashboard_link}")
        # self.pca_transformed = ___ # call PCA code that returns numpy array here
        # (rename self.pca_transformed to self.full_dataset)
        # numpy array can also be created from the saved (pickle) file

        # for data:
        # instead of first looping over hyperparameter values and then over different
        # train-validation sets, is it better to do it in the opposite order
        # to allow for one set of train-validation data to be created only once?

        try:
            # this commented out code did not work without the meta= argument,
            # meta= was not tried as it needs all other columns listed, in
            # addition to the ones being recast
            # self.full_dataset = self.client.persist(
            #     dd.read_parquet(
            #         s3_path, index=False, engine="pyarrow"
            #     )
            #     .sample(frac=self.frac, random_state=42)
            #     .map_partitions(
            #         self.cast_types,
            #         meta={
            #             'sid_shop_item_qty_sold_day': 'i2',
            #             **{f'cat{n}': 'i2' for n in range(1,23)}
            #         }
            #     )
            #     .map_partitions(self.drop_neg_qty_sold)
            #     .set_index(
            #         "sale_date", sorted=False, npartitions="auto"
            #     )
            #     .repartition(partition_size="100MB")
            # )

            # create Dask dataframe from partitioned Parquet dataset on S3 and persist it to cluster
            self.full_dataset = dd.read_parquet(
                s3_path, index=False, engine="pyarrow"
            ).sample(frac=self.frac, random_state=42)
            self.full_dataset["sale_date"] = self.full_dataset["sale_date"].astype(
                "datetime64[ns]"
            )
            self.full_dataset["sid_shop_item_qty_sold_day"] = self.full_dataset[
                "sid_shop_item_qty_sold_day"
            ].astype("int16")
            for col in self.full_dataset:
                if col.startswith("cat"):
                    self.full_dataset[col] = self.full_dataset[col].astype("int16")

            logging.debug(
                f"# of rows in full dataframe before removal of negative target values: {len(self.full_dataset)}"
            )
            self.full_dataset = self.full_dataset[
                self.full_dataset.sid_shop_item_qty_sold_day > 0
            ]
            # call dataframe.set_index(), then repartition, then persist
            # https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.set_index.html
            # set_index(sorted=False, npartitions='auto')
            # df = df.repartition(npartitions=df.npartitions // 100)

            # self.full_dataset = self.client.persist(self.full_dataset)
            # _ = wait([self.full_dataset])

            # https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.repartition.html
            # self.full_dataset = self.full_dataset.repartition(partition_size="100MB")
            self.full_dataset = self.full_dataset.set_index(
                # "sale_date", sorted=False, npartitions="auto", partition_size=100_000_000,
                "sale_date", sorted=False, npartitions="auto",
            )
            # partition_size for set_index: int, optional, desired size of
            # eaach partition in bytes (to be used with npartitions='auto')

            self.full_dataset = self.cull_empty_partitions(self.full_dataset)

            # self.full_dataset = self.client.persist(self.full_dataset)
            # _ = wait([self.full_dataset])
            # logging.debug(
            #     f"# of rows in full dataframe after removal of negative target values: {len(self.full_dataset)}"
            # )
            # logging.debug(
            #     f"Earliest and latest dates in full dataframe are : {dd.compute(self.full_dataset.index.min(), self.full_dataset.index.max())}"
            # )
            # logging.debug(
            #     f"Data types of full Dask dataframe are: {self.full_dataset.dtypes}"
            # )
            self.full_dataset = self.full_dataset.compute()

        except Exception:
            logging.exception(
                "Exception occurred while creating Dask dataframe and persisting it on the cluster."
            )
            # kill all active work, delete all data on the network, and restart the worker processes.
            self.client.restart()
            sys.exit(1)

        # finally:
        #     self.client.restart()
        #     sys.exit(1)

        # https://stackoverflow.com/questions/58437182/how-to-read-a-single-large-parquet-file-into-multiple-partitions-using-dask-dask
        # Parquet datasets can be saved into separate files.
        # Each file may contain separate row groups.
        # Dask Dataframe reads each Parquet row group into a separate partition.

        # I DON'T WANT TO KEEP THE NUMPY ARRAY IN MEMORY, SO IT NEEDS TO BE
        # DELETED AFTER DASK ARRAY IS CREATED
        # MIGHT BE BETTER TO CREATE DASK ARRAY FROM FILE ON S3, TO AVOID
        # HAVING BOTH NUMPY ARRAY AND PERSISTED DASK ARRAY IN MEMORY
        # I ALSO WANT TO SPLIT THAT NUMPY ARRAY INTO MULTIPLE TRAIN AND VALIDATION
        # SETS, SO WHAT'S THE BEST WAY TO DO THAT?
        # SEND THE ENTIRE ARRAY TO THE CLUSTER AT ONCE - PROBABLY NOT, OR
        # SEND TRAIN AND VALIDATION SETS ONE BY ONE AND DELETE?
        # BUT THAT WILL REQUIRE SENDING DATA TO THE CLUSTER MULTIPLE TIMES -
        # NOT IF THE DATA BEING SENT ARE DIFFERENT EACH TIME
        # THEY ARE NOT GOING TO BE COMPLETELY DIFFERENT BECAUSE TRAIN DATA WILL
        # JUST CONTINUE TO MERGE WITH VALIDATION SETS AND GROW
        # CREATE FIRST DASK ARRAY AND SEND TO CLUSTER, THEN APPEND TO IT?
        # IT DOES NOT LOOK LIKE DASK WOULD ALLOW THAT (SEE
        # https://github.com/dask/distributed/issues/1676 -
        # "You should also be aware that the task/data model underlying dask
        # arrays is immutable. You should never try to modify memory in-place.")
        # SO PROBABLY SEND ALL OF THE DATA TO THE CLUSTER AT THE BEGINNING,
        # THEN TAKE CHUNKS OF IT FOR WALK-FORWARD VALIDATION

        # PROBABLY SHOULD RELY ON LOADING DATA FROM FILE USING DELAYED /
        # FROM_DELAYED
        # SEE https://stackoverflow.com/questions/45941528/how-to-efficiently-send-a-large-numpy-array-to-the-cluster-with-dask-array)

        # can I use a function to read multiple files into one Dask array?

        # either figure out how to read multiple files (saved on S3) into one
        # Dask array, or
        # figure out how to save one array of PCA results to S3 (need disk space
        # to save it locally before transfer to S3 and need a method that can
        # handle transfer of more than 5GB - multipart transfer to S3)

        # try to write PCA-transformed data directly to zarr array (stored in memory)
        # then upload it to S3 (directly from memory)
        # then create dask array from that zarr array in S3

        # try to write PCA-transformed data to xarray then upload it to S3 as zarr

        # save numpy array to parquet file, upload that file to S3 (using upload_file),
        # then read that file into a Dask dataframe
        # write data to parquet on S3 from pandas dataframe and append to it using awswrangler library?
        # (https://github.com/awslabs/aws-data-wrangler/blob/main/tutorials/004%20-%20Parquet%20Datasets.ipynb)
        # df = dd.read_parquet('s3://bucket/my-parquet-data')
        # (https://docs.dask.org/en/latest/generated/dask.dataframe.read_parquet.html#dask.dataframe.read_parquet)
        # from above link:
        # engine argument: If ‘pyarrow’ or ‘pyarrow-dataset’ is specified, the ArrowDatasetEngine (which leverages the pyarrow.dataset API) will be used.
        # read partitioned parquet dataset with Dask:
        # https://stackoverflow.com/questions/67222212/read-partitioned-parquet-dataset-written-by-spark-using-dask-and-pyarrow-dataset

    # def cast_types(self, df):
    #     df = df.copy()
    #     df['sale_date'] = df["sale_date"].astype(
    #         "datetime64[ns]"
    #     )
    #     for col in df:
    #         if col.startswith("cat") or (col == "sid_shop_item_qty_sold_day"):
    #             df[col] = df[col].astype("int16")
    #     return df
    #
    # def drop_neg_qty_sold(self, df):
    #     return df[df.sid_shop_item_qty_sold_day >= 0].copy()

    # function from https://stackoverflow.com/questions/47812785/remove-empty-partitions-in-dask
    def cull_empty_partitions(self, ddf):
        ll = list(ddf.map_partitions(len).compute())
        ddf_delayed = ddf.to_delayed()
        ddf_delayed_new = list()
        pempty = None
        for ix, n in enumerate(ll):
            if 0 == n:
                pempty = ddf.get_partition(ix)
            else:
                ddf_delayed_new.append(ddf_delayed[ix])
        if pempty is not None:
            ddf = dd.from_delayed(ddf_delayed_new, meta=pempty)
        return ddf

    def truncated_poisson(self, y_true, y_pred):
        # function needs to return grad and hess
        # y_true array-like of shape = [n_samples]
        # The target values.
        #
        # y_pred array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
        # The predicted values. Predicted values are returned before any transformation, e.g. they are raw margin instead of probability of positive class for binary task.
        #
        # group array-like
        # Group/query data. Only used in the learning-to-rank task. sum(group) = n_samples. For example, if you have a 100-document dataset with group = [10, 20, 40, 10, 10, 10], that means that you have 6 groups, where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        #
        # grad array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
        # The value of the first order derivative (gradient) of the loss with respect to the elements of y_pred for each sample point.
        #
        # hess array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
        # The value of the second order derivative (Hessian) of the loss with respect to the elements of y_pred for each sample point.

        # per https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/statug/statug_fmm_details07.htm
        loss = np.log(np.exp(np.exp(y_pred)) - 1) - y_true * y_pred
        grad = (np.exp(y_pred + np.exp(y_pred))) / (np.exp(np.exp(y_pred)) - 1) - y_true
        hess = (
            (np.exp(y_pred + np.exp(y_pred)) * (np.exp(np.exp(y_pred)) - np.exp(y_pred) - 1)) /
            (np.exp(np.exp(y_pred)) - 1)**2
        )
        return grad, hess

    def gridsearch_wfv(self, params):
        # self.hyperparameters = hyperparameters
        # self.rmse_results = defaultdict(list) # replace this variable by creating a key-value in
        # the self.hyper_dict dictionary with value containing list of RMSE values
        self.all_params_combs = list()
        # determine if there is more than one combination of hyperparameters
        # if only one combination, set get_stats_ flag to True
        self.get_stats_ = (
            len(params[max(params, key=lambda x: len(params[x]))]) == 1
        )
        for params_comb_dict in (
            dict(zip(params.keys(), v)) for v in list(product(*list(params.values())))
        ):
            # for self.hyper_dict in hyperparameters:
            # self.params_combs_list.append(params_comb_dict)
            self.params_comb_dict = params_comb_dict.copy()
            self.params_comb_dict["rmse_list_"] = list()
            self.params_comb_dict["monthly_rmse_list_"] = list()
            self.params_comb_dict["fit_times_list_"] = list()
            try:
                self.model = lgb.LGBMRegressor(
                    # client=self.client,
                    objective=self.truncated_poisson,
                    random_state=42,
                    silent=False,
                    # tree_learner="data",
                    # force_row_wise=True,
                    **params_comb_dict,
                )
            except Exception:
                logging.exception("Exception occurred while initializing Dask model.")
                # kill all active work, delete all data on the network, and restart the worker processes.
                self.client.restart()
                sys.exit(1)

            # call method that loops over train-validation sets
            # with performance_report(filename=f"dask_report_{self.curr_dt_time}.html"):
            for train, test, get_stats in self.train_test_time_split():
                self.fit(train).predict(test).rmse_all_folds(test, get_stats)

            self.params_comb_dict["avg_rmse_"] = mean(
                self.params_comb_dict["rmse_list_"]
            )
            self.params_comb_dict["monthly_avg_rmse_"] = mean(
                self.params_comb_dict["monthly_rmse_list_"]
            )
            self.all_params_combs.append(self.params_comb_dict)

        best_params = max(self.all_params_combs, key=lambda x: x["monthly_avg_rmse_"])
        self.best_score_ = best_params["monthly_avg_rmse_"]
        # remove non-parameter key-values from self.best_params (i.e., rmse_list_ and avg_rmse_, etc.)
        self.best_params_ = {k: v for k, v in best_params.items() if k in params}

        # save list of parameter-result dictionaries to dataframe and then to CSV
        if self.all_params_combs:
            all_params_combs_df = pd.DataFrame(self.all_params_combs)
            output_csv = "all_params_combs.csv"
            all_params_combs_df.to_csv(output_csv, index=False)

            try:
                key = f"lightgbm_all_params_combs_{self.curr_dt_time}.csv"
                # global s3_client
                s3_client = boto3.client("s3")
                response = s3_client.upload_file(output_csv, "sales-demand-data", key)
                logging.info(
                    "Name of CSV uploaded to S3 and containing all parameter combinations "
                    f"and results is: {key}"
                )
            except ClientError as e:
                logging.exception(
                    "CSV file with LightGBM parameter combinations and results was not copied to S3."
                )

        else:
            logging.debug(
                "List of parameter-result dictionaries is empty and was not converted to CSV!"
            )

            # probably do the opposite:
            # loop over train-validation splits (persisting that data in memory)
            # and run different models on one
            # split, saving the results that can later be aggregated

            # is it possible to read the full range of dates needed for time
            # series validation and then drop/delete rows from array or
            # move some rows to another array:
            # start with July-September (train) + October (validation),
            # then remove October and move September from train to validation

    # def time_split(self):
    #     return (
    #         self.full_dataset.loc[:self.end_date],
    #         self.full_dataset.loc[self.end_date + timedelta(days=1):self.end_date + relativedelta(months=self.n_months_in_val_set, day=31)]
    #         # self.full_dataset[date > self.end_date & date <= self.end_date + relativedelta(months=n_months_in_val_set, day=31)]
    #         # less than or equal to last day of month currently used for validation
    #     )

    def train_test_time_split(self):
        # first (earliest) month: July 2015
        # number of months in first train set: 1
        # number of months in validation set: 2
        #
        # number of months between Oct 2015 and July 2015: 3
        # 3 - (2 - 1) = 2 (two 2-month intervals inside a 3-month interval)
        # (where 2 is the number of months in validation set)

        # (3 - n_months_in_first_train_set + 1) - (2 - 1)
        n_val_sets = (
            month_counter(self.startmonth)  # self.startmonth is e.g. July 1, 2015
            - self.n_months_in_first_train_set
            + 1
        ) - (self.n_months_in_val_set - 1)

        for m in range(n_val_sets):
            end_date = self.startmonth + relativedelta(
                months=m + self.n_months_in_first_train_set - 1, day=31
            )
            if self.get_stats_:
                get_stats = m == n_val_sets - 1
            else:
                get_stats = False
            yield (
                self.full_dataset.loc[:end_date],
                self.full_dataset.loc[
                    end_date
                    + timedelta(days=1) : end_date
                    + relativedelta(months=self.n_months_in_val_set, day=31)
                ],
                get_stats
            )
            # self.train, self.test = self.time_split(self.full_dataset, self.end_date)

    def get_sample_weights(self, train):
        weights_arr = train["sid_shop_item_qty_sold_day"].to_numpy().astype('float32')
        weights_arr = np.where(weights_arr == 0, self.params_comb_dict['weight_for_zeros'], 1.)
        return weights_arr

    def fit(self, train):
        try:
            start_time = time.perf_counter()
            logging.debug(
                f"train X dtypes are {train[[col for col in train if col.startswith(('pc','cat'))]].dtypes}"
            )
            logging.debug(
                f"train y type is {train['sid_shop_item_qty_sold_day'].dtype}"
            )
            self.model.fit(
                train[[col for col in train if col.startswith(("pc","cat"))]].to_numpy(),
                train["sid_shop_item_qty_sold_day"].to_numpy(),
                sample_weight=self.get_sample_weights(train),
                feature_name=[col for col in train if col.startswith(("pc","cat"))],
                categorical_feature=[col for col in train if col.startswith("cat")],
            )
            assert self.model.fitted_
            self.params_comb_dict["fit_times_list_"].append(
                time.perf_counter() - start_time
            )

            return self

        except Exception:
            logging.exception(
                "Exception occurred while fitting model on train data during walk-forward validation."
            )
            # kill all active work, delete all data on the network, and restart the worker processes.
            self.client.restart()
            sys.exit(1)

    def predict(self, test):
        try:
            self.y_pred = np.exp(
                self.model.predict(
                    test[[col for col in test if col.startswith(("pc","cat"))]]
                )
            )
            return self
        except Exception:
            logging.exception(
                "Exception occurred while computing predicted values on the test data."
            )
            # kill all active work, delete all data on the network, and restart the worker processes.
            self.client.restart()
            sys.exit(1)

    def rmse_all_folds(self, test, get_stats):
        try:
            # logging.debug(f"Data type of test['sid_shop_item_qty_sold_day'] is: {type(test['sid_shop_item_qty_sold_day'])}")
            # logging.debug(f"Data type of self.y_pred is: {type(self.y_pred)}")
            # logging.debug(f"Shape of test['sid_shop_item_qty_sold_day'] is: {test['sid_shop_item_qty_sold_day'].compute().shape}")
            # logging.debug(f"Shape of self.y_pred is: {self.y_pred.compute().shape}")
            self.params_comb_dict["rmse_list_"].append(
                calc_rmse(
                    test["sid_shop_item_qty_sold_day"].to_numpy(),
                    self.y_pred,
                    get_stats,
                )
            )
            # self.rmse_results[json.dumps(self.hyper_dict)].append(calc_rmse(test[["sid_shop_item_qty_sold_day"]], self.y_pred))

            self.params_comb_dict["monthly_rmse_list_"].append(
                calc_monthly_rmse(
                    test[["shop_id", "item_id", "sid_shop_item_qty_sold_day"]],
                    self.y_pred,
                )
            )

        except Exception:
            logging.exception(
                "Exception occurred while computing RMSE on the test data."
            )
            # kill all active work, delete all data on the network, and restart the worker processes.
            self.client.restart()
            sys.exit(1)

    def refit_and_save(self, model_path):
        """
        https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc/55209076
        """
        try:
            self.best_model = lgb.LGBMRegressor(
                # client=self.client,
                objective=self.truncated_poisson,
                random_state=42,
                silent=False,
                # tree_learner="data",
                # force_row_wise=True,
                **self.best_params_,
            )
            self.best_model.fit(
                self.full_dataset[
                    [col for col in self.full_dataset if col.startswith(("pc","cat"))]
                ].to_numpy(),
                self.full_dataset["sid_shop_item_qty_sold_day"].to_numpy(),
                sample_weight=self.get_sample_weights(self.full_dataset),
                feature_name=[col for col in self.full_dataset if col.startswith(("pc","cat"))],
                categorical_feature=[col for col in self.full_dataset if col.startswith("cat")],
            )
            output_txt = str(model_path).split("/")[-1]
            booster = self.best_model.booster_.save_model(output_txt)

            # output_txt = str(model_path).split('/')[-1]
            # global s3_client
            s3_client = boto3.client("s3")
            response = s3_client.upload_file(
                output_txt, "sales-demand-data", output_txt
            )
            logging.info(f"Name of saved model uploaded to S3 is: {output_txt}")

        except (Exception, ClientError):
            logging.exception(
                "Exception occurred while fitting model on the full dataset and saving the booster to file on S3."
            )
            # kill all active work, delete all data on the network, and restart the worker processes.
            self.client.restart()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "n_workers", metavar="<n_workers>", help="number of Dask workers", type=int,
    )

    parser.add_argument(
        "s3_path",
        metavar="<s3_path>",
        help="path to S3 folder containing PCA-transformed data in Parquet dataset format",
    )

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
        "--frac",
        "-f",
        help="fraction of rows to sample (default is 1.0 if omitted)",
        default="1.0",
        type=valid_frac,
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
    log_fname = f"logging_{curr_dt_time}_lightgbm.log"
    log_path = log_dir.joinpath(log_fname)

    model_dir = Path.cwd()
    model_fname = f"lgbr_model_{curr_dt_time}.txt"
    model_path = model_dir.joinpath(model_fname)

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

    # statements to suppress some of the logging messages from dask
    # more info here: https://docs.dask.org/en/latest/debugging.html
    logging.getLogger("dask").setLevel(logging.WARNING)
    logging.getLogger("distributed").setLevel(logging.WARNING)

    # also suppress s3fs messages
    logging.getLogger("s3fs").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    # Check if code is being run on EC2 instance (vs locally)
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

    logging.info(f"The Python version is {platform.python_version()}.")
    logging.info(f"The pandas version is {pd.__version__}.")
    logging.info(f"The Dask version is {dsk.__version__}.")
    logging.info(f"The LightGBM version is {lgb.__version__}.")

    s3_client = boto3.client("s3")

    logging.info(
        f"Running LightGBM model with n_workers: {args.n_workers}, s3_path: {args.s3_path}, "
        f"startmonth: {args.startmonth}, n_months_in_first_train_set: {args.n_months_in_first_train_set}, "
        f"n_months_in_val_set: {args.n_months_in_val_set}, and frac: {args.frac}..."
    )

    model = LightGBMDaskLocal(
        curr_dt_time,
        args.n_workers,
        args.s3_path,
        args.startmonth,
        args.n_months_in_first_train_set,
        args.n_months_in_val_set,
        frac=args.frac,
    )
    model.gridsearch_wfv(params)
    model.refit_and_save(model_path)

    # copy log file to S3 bucket
    try:
        response = s3_client.upload_file(
            f"./logs/{log_fname}", "my-ec2-logs", log_fname
        )
    except ClientError as e:
        logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()


#
# # set up client with 2 workers, each having two threads and each having a 2GB memory limit
# client = Client(n_workers=2, threads_per_worker=2, memory_limit='2GB')
# # this set ups local cluster on your local machine
#
# # https://distributed.dask.org/en/latest/api.html#client
# # class distributed.Client()
# # Connect to and submit computation to a Dask cluster
# # It is also common to create a Client without specifying the scheduler address ,
# # like Client(). In this case the Client creates a LocalCluster in the background
# # and connects to that. Any extra keywords are passed from Client to LocalCluster
# # in this case. See the LocalCluster documentation for more information.
#
# # https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
# # Create local Scheduler and Workers
# # This creates a “cluster” of a scheduler and workers running on the local machine.
#
# # https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html#dask
# # from distributed import Client, LocalCluster
# # cluster = LocalCluster(n_workers=3)
# # client = Client(cluster)
#
#
# # Set up a local Dask cluster
# # https://github.com/tvquynh/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
# # original repo: https://github.com/jameslamb/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
# # Create a cluster with 3 workers. Since this is a LocalCluster, those workers are just 3 local processes.
# from dask.distributed import Client, LocalCluster
#
# n_workers = 3
# cluster = LocalCluster(n_workers=n_workers)
#
# client = Client(cluster)
# client.wait_for_workers(n_workers)
#
# print(f"View the dashboard: {cluster.dashboard_link}")
# # Click the link above to view a diagnostic dashboard while you run the training code below.
#
# # Train a model
# # https://github.com/tvquynh/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
# from lightgbm.dask import DaskLGBMRegressor
#
# # The DaskLGBMRegressor class from lightgbm accepts any parameters that can be
# # passed to lightgbm.LGBRegressor, with one exception: num_thread.
# # Any value for num_thread that you pass will be ignored, since the Dask estimators
# # reset num_thread to the number of logical cores on each Dask worker.
# # (https://saturncloud.io/docs/examples/machinelearning/lightgbm-training/)
# dask_reg = DaskLGBMRegressor(
#     client=client,
#     max_depth=5,
#     objective="regression_l1",
#     learning_rate=0.1,
#     tree_learner="data",
#     n_estimators=100,
#     min_child_samples=1,
# )
#
# dask_reg.fit(
#     X=dX,
#     y=dy,
# )
#
# # Evaluate the model
# # https://github.com/tvquynh/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
# from dask_ml.metrics.regression import mean_absolute_error
# mean_absolute_error(preds, dy)
#
# # https://saturncloud.io/docs/examples/machinelearning/lightgbm-training/
# from dask_ml.metrics import mean_absolute_error
# mae = mean_absolute_error(
#     y_true=holdout_labels,
#     y_pred=preds,
#     compute=true
# )
# print(f"Mean Absolute Error: {mae}")
#
# # manual function for calculating error
# # https://github.com/Nixtla/mlforecast/blob/main/nbs/distributed.forecast.ipynb
# def mse_from_dask_dataframe(ddf):
#     ddf['sq_err'] = (ddf['y'] - ddf['y_pred'])**2
#     mse = ddf['sq_err'].mean()
#     return mse.compute()
#
#
# if __name__ == "__main__":
#     print("loading data")
#
#     X, y = make_regression(n_samples=1000, n_features=50)
#
#     print("initializing a Dask cluster")
#
#     cluster = LocalCluster(n_workers=2)
#     client = Client(cluster)
#
#     print("created a Dask LocalCluster")
#
#     print("distributing training data on the Dask cluster")
#
#     dX = da.from_array(X, chunks=(100, 50))
#     dy = da.from_array(y, chunks=(100,))
#
#     print("beginning training")
#
#     dask_model = lgb.DaskLGBMRegressor(n_estimators=10)
#     dask_model.fit(dX, dy)
#     assert dask_model.fitted_
#
#     print("done training")
