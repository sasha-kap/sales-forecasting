"""
Incremental Standard Scaling and Principal Component Analysis of Data Stored
in Large CSV Files in a AWS S3 Bucket

Includes Functions to Plot Explained Variance of Principal Components and to
Perform Necessary Preprocessing of Some of the Features

General Workflow:
1. Read CSV files from S3 bucket
2. Iterate over CSV chunks and apply StandardScaler (two passes: fit and transform)
(per https://stackoverflow.com/questions/52642940/feature-scaling-for-a-big-dataset)
3. Pass scaled data to IncrementalPCA() (partial_fit, then transform on the third pass
over CSV chunks)

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # Get a list of CSV files (as a preliminary check)
    python incremental_pca.py list 15-07

    # Run PCA with 10000 chunksize, 50% sampling rate, and 10 PCs
    python incremental_pca.py pca 15-10 -s 10000 -f 0.5 -c 10

    # Run PCA with default 1000 chunksize and 100% sampling rate, plotting
    # cumulative explained variance of all principal components
    python incremental_pca.py pca 15-09

    # Run same PCA as above, but include binary features in StandardScaler
    python incremental_pca.py pca 15-08 -b
"""

import argparse
import datetime
import io

# from itertools import chain, tee
import json
import logging
import os
from pathlib import Path
import pickle as pk
import platform
import sys
import time

import awswrangler as wr
import boto3
from botocore.exceptions import ClientError
from dateutil.relativedelta import relativedelta
from ec2_metadata import ec2_metadata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

s3_client = boto3.client("s3")

# in addition to Prefix=, can also use:
# StartAfter (string) -- StartAfter is where you want Amazon S3 to start listing from.
# Amazon S3 starts listing after this specified key. StartAfter can be any key in the bucket.
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2
# (keys are like 'shops15_10.csv')
def list_csvs(bucket="my-rds-exports", prefix="shops", first_mon=""):
    """List CSV files in S3 bucket.

    Parameters:
    -----------
    bucket : str
        S3 bucket containing CSV files (default: 'my-rds-exports')
    prefix : str
        Prefix of CSV file name (S3 key) (default: 'shops')
    first_mon : str
        First month of CSV data to be included in output (default: '', which
        produces a list of all files)

    Returns:
    --------
    csv_list : list
        List of CSV file names (S3 keys)
    """
    try:
        csv_list = [
            key["Key"]
            for key in s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, StartAfter=first_mon
            )["Contents"]
        ]
        return csv_list
    except ClientError:
        logging.exception(
            "Exception occurred during execution of list_csvs() function."
        )
        sys.exit(1)


# response1 = s3_client.get_object(Bucket=bucket, Key=file_name1)
# status1 = response1.get("ResponseMetadata", {}).get("HTTPStatusCode")
#
# response2 = s3_client.get_object(Bucket=bucket, Key=file_name2)
# status2 = response2.get("ResponseMetadata", {}).get("HTTPStatusCode")

# if status1 == 200 and status2 == 200:
#     print(f"Successful S3 get_object response. Status1 - {status1}. Status2 - {status2}.")
# reader1, reader2, reader3 = tee(chain(pd.read_csv(response1.get("Body"), chunksize=100),
#     pd.read_csv(response2.get("Body"), chunksize=100)), 3)

# add to read_csv code: encoding, data types dictionary
# also add code to select dtypes that can be used for PCA
# preprocessing: 1) sqrt any features that are counts. log any feature with a heavy tail.
# Need to add code to save numpy array of transformed data to file
#   save compressed? https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
#   upload file to S3
#   need to make sure that I can later match rows in this transformed dataset to rows in original data
# save mean_, var_ (arrays) (NO, THOSE SHOULD JUST BE 0 AND 1)
# change print() commands to logging commands
#   for iterator progress, keep print() but add datetime.datetime.strftime(datetime.datetime.now(), format="%Y-%m-%d %H:%M:%S")

# if needed:
# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
# shows how to dump PCA object to pickle file


def plot_pca_components(fit_pca, first_mon, frac, scale_bins, curr_dt_time):
    """Plot cumulative explained variance of all PCs from PCA results.

    Parameters:
    -----------
    fit_pca : IncrementalPCA partial_fit() object
        Results of PCA fitting
    first_mon : str
        First month of CSV data included in PCA
    frac : float
        Fraction of rows that was sampled from CSVs
    scale_bins : bool
        Whether binary columns were scaled with StandardScaler prior to PCA
    curr_dt_time : str
        Date-time string created at start of script and to be used to insert
        consistent timestamp in filenames

    Returns:
    --------
    None
    """
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.cumsum(fit_pca.explained_variance_ratio_))
    ax.set_title("Cumulative Explained Variance Ratio of Principal Components")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1)

    # place a text box in lower right in axes coords
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # get datetime value of first_mon
    m = datetime.datetime.strptime(
        first_mon.replace("shops_", "").replace("_addl", "").split(".")[0], "%y_%m"
    )
    # add one month
    m = m + relativedelta(months=1)
    # convert to MMM YYYY string format
    fm = datetime.datetime.strftime(m, format="%b %Y")

    textstr = "\n".join(
        (
            f"First month: {fm}",
            f"Fraction: {frac:.2f}",
            f"# Samples: {fit_pca.n_samples_seen_:,}",
            f"# Components: {fit_pca.n_components_}",
            f"Binary cols scaled: {scale_bins}",
        )
    )
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax.grid(True)
    plt.savefig("pca_components.png")
    try:
        key = f"pca_components_{curr_dt_time}.png"
        response = s3_client.upload_file("pca_components.png", "sales-demand-data", key)
    except ClientError as e:
        logging.exception("PCA explained variance plot file was not copied to S3.")


def preprocess_chunk(df, extract_cat_cols, null_col_dict, index, iteration=None):
    """Perform necessary preprocessing steps on each chunk of CSV data prior
    to passing data to StandardScaler.

    Parameters:
    -----------
    df : DataFrame
        Chunk of CSV data
    extract_cat_cols : bool
        keep binary and categorical features along with PCs in final
        Parquet dataset (if True) or not (if False)
    null_col_dict : dict
        Dictionary of columns that have null values in CSVs, with their
        data types that need to be assigned after nulls are filled with 0's
    index : int
        Chunk counter
    iteration: int
        counter of iterations over data in pca() function

    Returns:
    --------
    DataFrame
        Dataframe with preprocessed features
    """
    # drop string columns, quantity sold columns that include quantity from current day,
    # and row identification columns (shop, item, date)
    # also drop sid_coef_var_price, which cannot be reliably constructed for test data
    cols_to_drop = (
        [
            "i_item_category_id",
            "id_item_category_id",
            "sid_item_category_id",
            "i_item_cat_grouped_by_game_console",
            "i_item_category_name",
            "i_item_name",
            "s_city",
            "s_shop_name",
            "sid_coef_var_price",
        ]
        + [col for col in df.columns if col.endswith("_qty_sold_day")]
        + ["d_day_total_qty_sold"]
        + ["shop_id", "item_id", "sale_date"]
    )
    # errors='ignore' is added to suppress error when sid_coef_var_price is not found
    # among existing labels
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    # fill columns with null values and change data type from float to the type
    # previously determined for each column
    for col, dtype_str in null_col_dict.items():
        # sid_shop_item_qty_sold_day is already dropped above, so it can be excluded
        # from this step
        if not col.endswith("_qty_sold_day") and col != "sid_coef_var_price":
            df[col].fillna(0, inplace=True)
            df[col] = df[col].astype(dtype_str)

    # fix data type of some columns
    df["d_modern_education_share"] = df["d_modern_education_share"].apply(
        lambda x: float(x.replace(",", "."))
    )
    df["d_old_education_build_share"] = df["d_old_education_build_share"].apply(
        lambda x: float(x.replace(",", "."))
    )

    # replace previously missed negative infinity value in one of the columns
    df["id_cat_qty_sold_per_item_last_7d"] = df[
        "id_cat_qty_sold_per_item_last_7d"
    ].replace(-np.inf, 0)

    # cast to int the column that was for some reason cast to float in PostgreSQL
    df["id_num_unique_shops_prior_to_day"] = df[
        "id_num_unique_shops_prior_to_day"
    ].astype("int16")

    broad_cats = [
        "Аксессуары",
        "Кино",
        "Служебные",
        "Программы",
        "Музыка",
        "PC",
        "Подарки",
        "Игровые",
        "Элементы",
        "Доставка",
        "Билеты",
        "Карты",
        "Игры",
        "Чистые",
        "Книги",
    ]
    mons_of_first_sale = [x for x in range(13)]
    years = [2013, 2014, 2015]
    dow = [x for x in range(7)]
    months = [x for x in range(12)]
    quarters = [x for x in range(1, 5)]

    # encode categorical features as dummies
    prefix = "i_item_cat_broad"
    df_cats = pd.get_dummies(df.i_item_category_broad, prefix=prefix)
    cols = df_cats.columns.union([prefix + "_" + x for x in broad_cats])
    df_cats = df_cats.reindex(cols, axis=1, fill_value=0).astype("uint8")

    prefix = "i_item_first_mon"
    df_first_months = pd.get_dummies(df.i_item_mon_of_first_sale, prefix=prefix)
    cols = df_first_months.columns.union(
        [prefix + "_" + str(x) for x in mons_of_first_sale]
    )
    df_first_months = df_first_months.reindex(cols, axis=1, fill_value=0).astype(
        "uint8"
    )

    prefix = "d_year"
    df_years = pd.get_dummies(df.d_year, prefix=prefix)
    cols = df_years.columns.union([prefix + "_" + str(x) for x in years])
    df_years = df_years.reindex(cols, axis=1, fill_value=0).astype("uint8")

    prefix = "d_day_of_week"
    df_dow = pd.get_dummies(df.d_day_of_week, prefix=prefix)
    cols = df_dow.columns.union([prefix + "_" + str(x) for x in dow])
    df_dow = df_dow.reindex(cols, axis=1, fill_value=0).astype("uint8")

    prefix = "d_month"
    df_months = pd.get_dummies(df.d_month, prefix=prefix)
    cols = df_months.columns.union([prefix + "_" + str(x) for x in months])
    df_months = df_months.reindex(cols, axis=1, fill_value=0).astype("uint8")

    prefix = "d_quarter"
    df_quarters = pd.get_dummies(df.d_quarter_of_year, prefix=prefix)
    cols = df_quarters.columns.union([prefix + "_" + str(x) for x in quarters])
    df_quarters = df_quarters.reindex(cols, axis=1, fill_value=0).astype("uint8")
    # d_week_of_year (1 to 53) - skipped for get_dummies because of high cardinality

    if extract_cat_cols and (iteration == 3):

        cat_col_names = [
            "i_item_category_broad",
            "i_item_mon_of_first_sale",
            "d_year",
            "d_day_of_week",
            "d_month",
            "d_quarter_of_year",
            "d_week_of_year",
        ]
        df_with_extra_cat_cols = pd.concat(
            [
                ordinal_encode(
                    df[cat_col_names].copy(),
                    broad_cats,
                    mons_of_first_sale,
                    years,
                    dow,
                    months,
                    quarters,
                ),
                df.select_dtypes(include="uint8"),
            ],
            axis=1,
        )

    df = pd.concat(
        [
            df.drop(
                [
                    "i_item_category_broad",
                    "i_item_mon_of_first_sale",
                    "d_year",
                    "d_day_of_week",
                    "d_month",
                    "d_quarter_of_year",
                ],
                axis=1,
            ),
            df_cats,
            df_first_months,
            df_years,
            df_dow,
            df_months,
            df_quarters,
        ],
        axis=1,
    )

    # check each chunk for infinity values and stop script if any values are found
    if df[df.isin([np.inf, -np.inf])].count().any():
        cts = df[df.isin([np.inf, -np.inf])].count().to_dict()
        non_zero_cts = {k: v for k, v in cts.items() if v > 0}
        logging.debug(
            f"Chunk {index} has columns with infinity values: " f"{non_zero_cts}"
        )
        sys.exit(1)

    if extract_cat_cols and (iteration == 3):
        return df, df_with_extra_cat_cols
    return df


def ordinal_encode(df, broad_cats, mons_of_first_sale, years, dow, months, quarters):

    enc = OrdinalEncoder(
        categories=[
            broad_cats,
            mons_of_first_sale,
            years,
            dow,
            months,
            quarters,
            [x for x in range(1, 54)],  # for d_week_of_year
        ],
        dtype="uint8",
    )
    # convert category types to uint8/16
    df["i_item_mon_of_first_sale"] = df["i_item_mon_of_first_sale"].astype("uint8")
    df["d_year"] = df["d_year"].astype("uint16")
    cat_cols = enc.fit_transform(df)
    cat_cols_df = pd.DataFrame(cat_cols, columns=df.columns, index=df.index)

    return cat_cols_df


def pca(
    bucket="my-rds-exports",
    chunksize=1000,
    n_components=None,
    first_mon="",
    frac=1.0,
    scale_bins=False,
    extract_cat_cols=False,
    curr_dt_time="",
):
    """Implement StandardScaler and IncrementalPCA with partial_fit() methods.

    Parameters:
    -----------
    bucket : str
        S3 bucket containing CSV data (default: 'my-rds-exports')
    chunksize : int
        Number of rows to include in each iteration of read_csv() (default: 1000)
    n_components: None or int
        Number of principal components to compute (all, if None (default))
    first_mon : str
        First month of CSV data included in PCA (all, if '' (default))
    frac : float
        Fraction of rows to sample from CSVs (default: 1.0)
    scale_bins : bool
        Whether binary columns are to be scaled with StandardScaler prior to PCA
        (default: False)
    extract_cat_cols : bool
        keep binary and categorical features along with PCs in final
        Parquet dataset (if True) or not (if False)
    curr_dt_time : str
        Date-time string created at start of script and to be used to insert
        consistent timestamp in filenames

    Returns:
    --------
    None
    """
    csv_list = list_csvs(first_mon=first_mon)
    assert isinstance(csv_list, list), f"csv_list is not a list, but {type(csv_list)}!"

    with open("./features/pd_types_from_psql_mapping.json", "r") as f:
        pd_types = json.load(f)

    del pd_types["sale_date"]  # remove sale_date as it will be included in parse_dates=

    # change types of integer columns to floats (float32) for columns that contain nulls
    # change the dictionary values, while also extracting the key-value pairs and
    # putting them into a separate dictionary to pass to the preprocess_chunk function
    # so columns can be changed to appropriate types after null values are filled.
    null_col_dict = dict()
    null_col_list = [
        "sid_shop_cat_qty_sold_last_7d",
        "sid_cat_sold_at_shop_before_day_flag",
        "sid_shop_item_rolling_7d_max_qty",
        "sid_shop_item_rolling_7d_min_qty",
        "sid_shop_item_rolling_7d_avg_qty",
        "sid_shop_item_rolling_7d_mode_qty",
        "sid_shop_item_rolling_7d_median_qty",
        "sid_shop_item_expand_qty_max",
        "sid_shop_item_expand_qty_mean",
        "sid_shop_item_expand_qty_min",
        "sid_shop_item_expand_qty_mode",
        "sid_shop_item_expand_qty_median",
        "sid_shop_item_date_avg_gap_bw_sales",
        "sid_shop_item_date_max_gap_bw_sales",
        "sid_shop_item_date_min_gap_bw_sales",
        "sid_shop_item_date_mode_gap_bw_sales",
        "sid_shop_item_date_median_gap_bw_sales",
        "sid_shop_item_date_std_gap_bw_sales",
        "sid_shop_item_cnt_sale_dts_last_7d",
        "sid_shop_item_cnt_sale_dts_last_30d",
        "sid_shop_item_cnt_sale_dts_before_day",
        "sid_expand_cv2_of_qty",
        "sid_shop_item_days_since_first_sale",
        "sid_days_since_max_qty_sold",
        "sid_shop_item_qty_sold_day",
        "sid_shop_item_first_month",
        "sid_shop_item_last_qty_sold",
        "sid_shop_item_first_week",
        "sid_shop_item_expanding_adi",
        "sid_shop_item_date_diff_bw_last_and_prev_qty",
        "sid_shop_item_days_since_prev_sale",
        "sid_shop_item_qty_sold_7d_ago",
        "sid_qty_median_abs_dev",
        "sid_coef_var_price",
        "sid_shop_item_qty_sold_2d_ago",
        "sid_qty_mean_abs_dev",
        "sid_shop_item_qty_sold_1d_ago",
        "sid_shop_item_qty_sold_3d_ago",
    ]
    for col in null_col_list:
        # these 3 columns have nulls, but they will need to be set to uint8
        # after nulls are filled, not to their signed data type
        if col in [
            "sid_cat_sold_at_shop_before_day_flag",
            "sid_shop_item_first_month",
            "sid_shop_item_first_week",
        ]:
            null_col_dict[col] = "uint8"
        else:
            null_col_dict[col] = pd_types[col]
        pd_types[col] = "float32"

    # change types of binary features to 'uint8'
    # do not include the three sid_ features in bin_features here, but add them to the
    # null_col_dict dictionary with uint8 type, which will change the type to uint8
    # after null values are filled in
    bin_features = (
        "d_holiday",
        "d_is_weekend",
        "d_major_event",
        "d_ps4_game_release_dt",
        "d_ps4_game_release_dt_plus_2",
        "i_digital_item",
        "id_item_first_month",
        "id_item_first_week",
        "id_item_had_spike_before_day",
        "s_online_store",
        "sd_shop_first_month",
        "sd_shop_first_week",
    )
    pd_types = {k: "uint8" if k in bin_features else v for k, v in pd_types.items()}

    select_dtypes_params = {"include": None, "exclude": None}
    if scale_bins:
        select_dtypes_params["include"] = "number"  # all numeric types
    else:
        select_dtypes_params["exclude"] = "uint8"  # binary columns

    # reader1, reader2, reader3 = tee(
    #     chain.from_iterable(
    #         [
    #             pd.read_csv(
    #                 s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body"),
    #                 encoding="utf-8",
    #                 dtype=pd_types,
    #                 chunksize=chunksize,
    #             )
    #             for csv_file in csv_list
    #         ]
    #     ),
    #     3,
    # )

    # def process_result_s3_chunks(bucket, key, chunksize):
    #     csv_obj = s3_client.get_object(Bucket=bucket, Key=key)
    #     body = csv_obj['Body']
    #     for df in pd.read_csv(body, chunksize=chunksize):
    #         process(df)

    scaler = StandardScaler()
    sklearn_pca = IncrementalPCA(n_components=n_components)

    print("Starting first iteration over CSVs - StandardScaler partial fit...")
    start_time = current_time = time.perf_counter()
    global_idx = -1
    for csv_file in csv_list:
        csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
        for index, chunk in enumerate(
            pd.read_csv(
                csv_body, chunksize=chunksize, dtype=pd_types, parse_dates=["sale_date"]
            ),
            global_idx + 1,
        ):
            if index == 0:
                logging.debug(
                    f"Columns in DF converted from CSV: {list(enumerate(chunk.columns))}"
                )
            # if chunk.isna().any().any():
            #     logging.debug(
            #         f"Chunk {index} has {', '.join(chunk.columns[chunk.isna().any()])} columns with nulls"
            #     )
            if index % 25 == 0:
                print(
                    f"current index is {index} and current time is "
                    f"{datetime.datetime.strftime(datetime.datetime.now(), format='%Y-%m-%d %H:%M:%S')}"
                )
                print(
                    f"elapsed time since last check (in secs): {round(time.perf_counter() - current_time, 2)}"
                )
                print(
                    f"total elapsed time (in secs): {round(time.perf_counter() - start_time, 2)}"
                )
                current_time = time.perf_counter()
            # scaler.partial_fit(chunk.sample(frac=0.5, random_state=42)['x'].to_numpy().reshape(1,-1))

            # things to think about:
            # which transformations to apply to which features (identify count data - manually?)
            # apply square root / log transform transformations *and* standard scale (or just one of the two)?
            # how to deal with negative values if decide to apply square root / log transforms

            # take the sample of chunk
            # perform necessary transformations on necessary features
            # also, drop any necessary features
            # pass transformed features to scaler.partial_fit()
            # binary columns are to be scaled with an optional boolean parameter
            # if they are not to be scaled, they are to be excluded from StandardScaler
            try:
                scaler.partial_fit(
                    preprocess_chunk(
                        chunk.sample(frac=frac, random_state=42).sort_values(
                            by=["shop_id", "item_id", "sale_date"]
                        ),
                        extract_cat_cols,
                        null_col_dict,
                        index,
                        iteration=1,
                        # scale_bins=scale_bins,
                    ).select_dtypes(**select_dtypes_params)
                )
                global_idx = index
            except ValueError:
                unique_dict = {col: chunk[col].unique() for col in chunk.columns}
                logging.debug(
                    f"Unique values in chunk that produced ValueError: {unique_dict}"
                )
                sys.exit(1)
            # they will just need to be added to scaled data for PCA

            # print(scaler.mean_, scaler.var_)

    # all_scaled_data = []
    # reader = pd.read_csv(response.get("Body"), chunksize=100)
    print(
        "Starting second iteration over CSVs - StandardScaler transform and IncrementalPCA partial fit..."
    )
    global_idx = -1
    for csv_file in csv_list:
        csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
        for index, chunk in enumerate(
            pd.read_csv(
                csv_body, chunksize=chunksize, dtype=pd_types, parse_dates=["sale_date"]
            ),
            global_idx + 1,
        ):
            if index % 25 == 0:
                print(
                    f"current index is {index} and current time is "
                    f"{datetime.datetime.strftime(datetime.datetime.now(), format='%Y-%m-%d %H:%M:%S')}"
                )
                print(
                    f"elapsed time since last check (in secs): {round(time.perf_counter() - current_time, 2)}"
                )
                print(
                    f"total elapsed time (in secs): {round(time.perf_counter() - start_time, 2)}"
                )
                current_time = time.perf_counter()
            # sample the chunk again
            # again perform the same necessary transformations
            # pass transformed features (and all other features that will be going into PCA) to scaler.transform()
            preprocessed_chunk = preprocess_chunk(
                chunk.sample(frac=frac, random_state=42).sort_values(
                    by=["shop_id", "item_id", "sale_date"]
                ),
                extract_cat_cols,
                null_col_dict,
                index,
                iteration=2,
                # scale_bins=scale_bins,
            )
            scaled_data = scaler.transform(
                preprocessed_chunk.select_dtypes(**select_dtypes_params)
            )
            # print(
            #     "Columns in chunk after preprocessed_chunk.select_dtypes() are: "
            #     f"{preprocessed_chunk.select_dtypes(**select_dtypes_params).columns.to_list()}."
            # )
            # print(
            #     "Number of columns in chunk after preprocessed_chunk.select_dtypes() is: "
            #     f"{len(preprocessed_chunk.select_dtypes(**select_dtypes_params).columns.to_list())} "
            #     f"and shape of scaled_data is {scaled_data.shape}."
            # )
            # pass standard-scaled data (plus binary features if they were not scaled) to PCA partial fit
            if not scale_bins:
                scaled_data = np.hstack(
                    (
                        scaled_data,
                        (preprocessed_chunk.select_dtypes(include="uint8").to_numpy()),
                    )
                )
            # print(
            #     "Columns in chunk with only binary columns are: "
            #     f"{preprocessed_chunk.select_dtypes(include='uint8').columns.to_list()}"
            # )
            # print(
            #     "Shape of chunk after preprocessed_chunk.select_dtypes(include='uint8') is: "
            #     f"{preprocessed_chunk.select_dtypes(include='uint8').to_numpy().shape} and "
            #     "shape of entire array passed to sklearn_pca.partial_fit() is "
            #     f"{scaled_data.shape}."
            # )
            sklearn_pca.partial_fit(scaled_data)
            global_idx = index

    if n_components is None:
        logging.info(
            f"The estimated number of principal components: {sklearn_pca.n_components_}"
        )
        logging.info(f"The total number of samples seen: {sklearn_pca.n_samples_seen_}")
        plot_pca_components(sklearn_pca, first_mon, frac, scale_bins, curr_dt_time)

    else:
        print("Starting third iteration over CSVs - IncrementalPCA transform...")
        pca_transformed = None
        shape_list = list()
        overall_shape = (0, None)
        overall_nbytes = 0
        global_idx = -1
        for csv_file in csv_list:
            csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
            for index, chunk in enumerate(
                pd.read_csv(
                    csv_body,
                    chunksize=chunksize,
                    dtype=pd_types,
                    parse_dates=["sale_date"],
                ),
                global_idx + 1,
            ):
                if index % 25 == 0:
                    print(
                        f"current index is {index} and current time is "
                        f"{datetime.datetime.strftime(datetime.datetime.now(), format='%Y-%m-%d %H:%M:%S')}"
                    )
                    print(
                        f"elapsed time since last check (in secs): {round(time.perf_counter() - current_time, 2)}"
                    )
                    print(
                        f"total elapsed time (in secs): {round(time.perf_counter() - start_time, 2)}"
                    )
                    current_time = time.perf_counter()
                chunk_sub = chunk.sample(frac=frac, random_state=42).sort_values(
                    by=["shop_id", "item_id", "sale_date"]
                )
                id_cols_arr = chunk_sub[
                    ["shop_id", "item_id", "sale_date", "sid_shop_item_qty_sold_day"]
                ].copy()
                # fill in null values of sid_shop_item_qty_sold_day as this data is not passed through
                # the preprocess_chunk function
                id_cols_arr["sid_shop_item_qty_sold_day"].fillna(0, inplace=True)
                id_cols_arr["sid_shop_item_qty_sold_day"] = id_cols_arr[
                    "sid_shop_item_qty_sold_day"
                ].astype("int16")
                id_cols_arr = id_cols_arr.to_numpy()
                # after chunk is sampled (above), apply the same transformation and standard scaling
                # pass the scaled data to PCA transform()
                if extract_cat_cols:
                    preprocessed_chunk, df_with_extra_cat_cols = preprocess_chunk(
                        chunk_sub,
                        extract_cat_cols,
                        null_col_dict,
                        index,
                        iteration=3,
                        # scale_bins=scale_bins,
                    )
                    extra_cat_cols_arr = df_with_extra_cat_cols.to_numpy()
                else:
                    preprocessed_chunk = preprocess_chunk(
                        chunk_sub,
                        extract_cat_cols,
                        null_col_dict,
                        index,
                        iteration=3,
                        # scale_bins=scale_bins,
                    )
                scaled_data = scaler.transform(
                    preprocessed_chunk.select_dtypes(**select_dtypes_params)
                )
                if not scale_bins:
                    scaled_data = np.hstack(
                        (
                            scaled_data,
                            (
                                preprocessed_chunk.select_dtypes(
                                    include="uint8"
                                ).to_numpy()
                            ),
                        )
                    )

                tx_chunk = sklearn_pca.transform(scaled_data)
                if pca_transformed is None:
                    if extract_cat_cols:
                        pca_transformed = np.hstack(
                            (id_cols_arr, extra_cat_cols_arr, tx_chunk)
                        )
                    else:
                        pca_transformed = np.hstack((id_cols_arr, tx_chunk))
                    shape_list.append(pca_transformed.shape)
                else:
                    if extract_cat_cols:
                        tx_chunk = np.hstack(
                            (id_cols_arr, extra_cat_cols_arr, tx_chunk)
                        )
                    else:
                        tx_chunk = np.hstack((id_cols_arr, tx_chunk))
                    shape_list.append(tx_chunk.shape)
                    pca_transformed = np.vstack((pca_transformed, tx_chunk))

                if extract_cat_cols:
                    column_names = (
                        [
                            "shop_id",
                            "item_id",
                            "sale_date",
                            "sid_shop_item_qty_sold_day",
                        ]
                        + [f"cat{x}" for x in range(1, extra_cat_cols_arr.shape[1] + 1)]
                        + [
                            f"pc{x}"
                            for x in range(
                                1,
                                pca_transformed.shape[1]
                                - 3
                                - extra_cat_cols_arr.shape[1],
                            )
                        ]
                    )
                    dtype_dict = {
                        **{
                            k: "float"
                            for k in [
                                f"pc{x}"
                                for x in range(
                                    1,
                                    pca_transformed.shape[1]
                                    - 3
                                    - extra_cat_cols_arr.shape[1],
                                )
                            ]
                        },
                        **{
                            k: "smallint"
                            for k in [
                                f"cat{x}"
                                for x in range(1, extra_cat_cols_arr.shape[1] + 1)
                            ]
                        },
                        **{
                            "shop_id": "smallint",
                            "item_id": "int",
                            "sid_shop_item_qty_sold_day": "smallint",
                        },
                    }
                    s3_path = f"s3://sales-demand-data/parquet_dataset_w_cat_cols/"
                else:
                    column_names = [
                        "shop_id",
                        "item_id",
                        "sale_date",
                        "sid_shop_item_qty_sold_day",
                    ] + [f"pc{x}" for x in range(1, pca_transformed.shape[1] - 3)]
                    dtype_dict = {
                        **{
                            k: "float"
                            for k in [
                                f"pc{x}" for x in range(1, pca_transformed.shape[1] - 3)
                            ]
                        },
                        **{
                            "shop_id": "smallint",
                            "item_id": "int",
                            "sid_shop_item_qty_sold_day": "smallint",
                        },
                    }
                    s3_path = f"s3://sales-demand-data/parquet_dataset/"

                if pca_transformed.nbytes > 100_000_000:
                    # convert the array to pandas dataframe and upload it to S3
                    # as a parquet file/dataset
                    wr.s3.to_parquet(
                        df=pd.DataFrame(pca_transformed, columns=column_names,),
                        path=s3_path,
                        index=False,
                        dataset=True,
                        mode="append",
                        partition_cols=["sale_date"],
                        # https://docs.aws.amazon.com/athena/latest/ug/data-types.html
                        dtype=dtype_dict,
                    )

                    # also update combined shape of PCA-transformed data
                    overall_shape = (
                        overall_shape[0] + pca_transformed.shape[0],
                        pca_transformed.shape[1],
                    )

                    # also update total bytes consumed by PCA-transformed data
                    overall_nbytes += pca_transformed.nbytes

                    # also, reset pca_transformed to None
                    pca_transformed = None

                global_idx = index

        if pca_transformed is not None:
            # convert the array to pandas dataframe and upload it to S3
            # as a parquet file/dataset
            wr.s3.to_parquet(
                df=pd.DataFrame(pca_transformed, columns=column_names,),
                path=s3_path,
                index=False,
                dataset=True,
                mode="append",
                partition_cols=["sale_date"],
                dtype=dtype_dict,
            )

            # also update combined shape of PCA-transformed data
            overall_shape = (
                overall_shape[0] + pca_transformed.shape[0],
                pca_transformed.shape[1],
            )

            # also update total bytes consumed by PCA-transformed data
            overall_nbytes += pca_transformed.nbytes

        print(f"Final shape of PCA-transformed data: {overall_shape}")
        print(
            f"Total bytes consumed by elements of PCA-transformed array: {overall_nbytes:,}"
        )

        shape_arr = np.array(shape_list)
        print(f"Size of shape_arr: {shape_arr.shape}")

        if np.max(shape_arr[:, 1]) != np.min(shape_arr[:, 1]):
            logging.debug("Different chunks have different counts of columns!!!")
        if (np.sum(shape_arr[:, 0]), np.min(shape_arr[:, 1])) != overall_shape:
            logging.debug(
                "Final shape of PCA-transformed data does not "
                "match the combined shape of individual chunks!!!"
            )

        print("Saving StandardScaler object and IncrementalPCA object to files...")
        # save the scaler
        with open("scaler.pkl", "wb") as fp:
            pk.dump(scaler, fp)
        # save the PCA object
        with open("pca.pkl", "wb") as fp:
            pk.dump(sklearn_pca, fp)

        try:
            key = f"scaler_{curr_dt_time}.pkl"
            response = s3_client.upload_file("scaler.pkl", "sales-demand-data", key)
        except ClientError as e:
            logging.exception(
                "Pickle file with dump of StandardScaler object was not copied to S3."
            )
        try:
            key = f"pca_{curr_dt_time}.pkl"
            response = s3_client.upload_file("pca.pkl", "sales-demand-data", key)
        except ClientError as e:
            logging.exception(
                "Pickle file with dump of PCA object was not copied to S3."
            )

            # save transformed data array to npy file on S3
            # IS IT OKAY TO BUILD UP A LARGE COMPLETE PCA_TRANSFORMED ARRAY?
            # SAY 20 PCs AND 10 MLN ROWS > 200 MLN * 8 BYTES = 1.6GB
            # MAYBE SPLIT PCA_TRANSFORMED WHEN WRITING TO FILE (OR TO MEMORY)
            # np.save('pca_transformed_data.npy', pca_transformed)


def valid_date(s):
    """Convert command-line date argument to YY-MM datetime value.

    Parameters:
    -----------
    s : str
        Command-line argument for first month of data to be used

    Returns:
    --------
    Datetime.datetime object (format: %y-%m)

    Raises:
    -------
    ArgumentTypeError
        if input string cannot be parsed according to %y-%m strptime format
    """
    try:
        return datetime.datetime.strptime(s, "%y-%m")
    except ValueError:
        msg = f"Not a valid date: {s}."
        raise argparse.ArgumentTypeError(msg)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", metavar="<command>", help="'list' or 'pca'",
    )
    parser.add_argument(
        "startmonth",
        metavar="<startmonth>",
        help="first (earliest) month of data to be used, format: YY-MM",
        type=valid_date,
    )
    parser.add_argument(
        "--chunksize",
        "-s",
        help="chunksize (number of rows) for read_csv(), default is 1,000",
        default="1000",
        type=int,
    )
    parser.add_argument(
        "--frac",
        "-f",
        help="fraction of rows to sample (default is 1.0 if omitted)",
        default="1.0",
        type=valid_frac,
    )
    parser.add_argument(
        "--comps",
        "-c",
        help="number of principal components to compute (all, if omitted)",
        type=int,
    )
    parser.add_argument(
        "--scale_bins",
        "-b",
        default=False,
        action="store_true",
        help="scale binary features (if included) or not (if not included)",
    )
    parser.add_argument(
        "--keep_cats",
        "-k",
        default=False,
        action="store_true",
        help=(
            "keep binary and categorical features along with PCs in final "
            "Parquet dataset (if included) or not (if not included)"
        ),
    )

    args = parser.parse_args()

    # if args.command == 'pca' and args.frac is None:
    #     parser.error("pca command requires --frac.")

    if args.command not in ["list", "pca"]:
        print("'{}' is not recognized. " "Use 'list' or 'pca'".format(args.command))

    else:

        fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        log_dir = Path.cwd().joinpath("logs")
        path = Path(log_dir)
        path.mkdir(exist_ok=True)
        curr_dt_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        log_fname = f"logging_{curr_dt_time}_{args.command}.log"
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
        # also, suppress irrelevant logging by matplotlib
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
        logging.getLogger("awswrangler").setLevel(logging.DEBUG)

        # Check if code is being run on EC2 instance (vs locally)
        my_user = os.environ.get("USER")
        is_aws = True if "ec2" in my_user else False
        # Log EC2 instance name and type metadata
        if is_aws:
            instance_metadata = dict()
            instance_metadata["EC2 instance ID"] = ec2_metadata.instance_id
            instance_metadata["EC2 instance type"] = ec2_metadata.instance_type
            instance_metadata[
                "EC2 instance public hostname"
            ] = ec2_metadata.public_hostname

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
        logging.info(f"The numpy version is {np.__version__}.")

        if args.command == "list":
            d_prev_mon = args.startmonth - relativedelta(months=1)
            first_mon = (
                "shops_"
                + datetime.datetime.strftime(d_prev_mon, format="%y_%m")
                + "_addl"
                + ".csv"
            )
            logging.info(
                f"Running list function with first_month: {args.startmonth}..."
            )
            print(list_csvs(first_mon=first_mon))

        elif args.command == "pca":
            d_prev_mon = args.startmonth - relativedelta(months=1)
            first_mon = (
                "shops_"
                + datetime.datetime.strftime(d_prev_mon, format="%y_%m")
                + "_addl"
                + ".csv"
            )
            logging.info(
                f"Running PCA function with chunksize: {args.chunksize}, n_components: {args.comps}, "
                f"first_month: {args.startmonth}, frac: {args.frac}, scale_bins: {args.scale_bins}, "
                f"keep_cats: {args.keep_cats}..."
            )
            pca(
                chunksize=args.chunksize,
                n_components=args.comps,
                first_mon=first_mon,
                frac=args.frac,
                scale_bins=args.scale_bins,
                extract_cat_cols=args.keep_cats,
                curr_dt_time=curr_dt_time,
            )

        # copy log file to S3 bucket
        try:
            response = s3_client.upload_file(
                f"./logs/{log_fname}", "my-ec2-logs", log_fname
            )
        except ClientError as e:
            logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()
