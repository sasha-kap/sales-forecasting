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
from sklearn.preprocessing import StandardScaler

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


def plot_pca_components(fit_pca, first_mon, frac, scale_bins):
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
        first_mon.replace("shops_", "").split(".")[0], "%y_%m"
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
        key = f"pca_components_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.png"
        response = s3_client.upload_file("pca_components.png", "sales-demand-data", key)
    except ClientError as e:
        logging.exception("PCA explained variance plot file was not copied to S3.")


def preprocess_chunk(df, index, scale_bins=False):
    """Perform necessary preprocessing steps on each chunk of CSV data prior
    to passing data to StandardScaler.

    Parameters:
    -----------
    df : DataFrame
        Chunk of CSV data
    index : int
        Chunk counter
    scale_bins : bool
        Whether binary columns are to be included in returned dataframe and
        passed to StandardScaler (default: False)

    Returns:
    --------
    DataFrame
        Dataframe with preprocessed features
    """
    # drop string columns, quantty sold columns that include quantity from current day,
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
    df.drop(cols_to_drop, axis=1, inplace=True)

    # fix data type of some columns
    df["d_modern_education_share"] = df["d_modern_education_share"].apply(
        lambda x: float(x.replace(",", "."))
    )
    df["d_old_education_build_share"] = df["d_old_education_build_share"].apply(
        lambda x: float(x.replace(",", "."))
    )

    # replace previously missed negative infinity value in one of the columns
    df['id_cat_qty_sold_per_item_last_7d'] = df['id_cat_qty_sold_per_item_last_7d'].replace(-np.inf, 0)

    # cast to int the column that was for some reason cast to float in PostgreSQL
    df["id_num_unique_shops_prior_to_day"] = df[
        "id_num_unique_shops_prior_to_day"
    ].astype("int16")

    # encode categorical features as dummies
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
    prefix = "i_item_cat_broad_"
    df_cats = pd.get_dummies(df.i_item_category_broad, prefix=prefix)
    cols = df_cats.columns.union([prefix + x for x in broad_cats])
    df_cats = df_cats.reindex(cols, axis=1, fill_value=0).astype('uint8')

    mons_of_first_sale = [x for x in range(13)]
    prefix = "i_item_first_mon_"
    df_first_months = pd.get_dummies(df.i_item_mon_of_first_sale, prefix=prefix)
    cols = df_first_months.columns.union([prefix + str(x) for x in mons_of_first_sale])
    df_first_months = df_first_months.reindex(cols, axis=1, fill_value=0).astype('uint8')

    years = [2013, 2014, 2015]
    prefix = "d_year_"
    df_years = pd.get_dummies(df.d_year, prefix=prefix)
    cols = df_years.columns.union([prefix + str(x) for x in years])
    df_years = df_years.reindex(cols, axis=1, fill_value=0).astype('uint8')

    dow = [x for x in range(7)]
    prefix = "d_day_of_week_"
    df_dow = pd.get_dummies(df.d_day_of_week, prefix=prefix)
    cols = df_dow.columns.union([prefix + str(x) for x in dow])
    df_dow = df_dow.reindex(cols, axis=1, fill_value=0).astype('uint8')

    months = [x for x in range(12)]
    prefix = "d_month_"
    df_months = pd.get_dummies(df.d_month, prefix=prefix)
    cols = df_months.columns.union([prefix + str(x) for x in months])
    df_months = df_months.reindex(cols, axis=1, fill_value=0).astype('uint8')

    quarters = [x for x in range(1, 5)]
    prefix = "d_quarter_"
    df_quarters = pd.get_dummies(df.d_quarter_of_year, prefix=prefix)
    cols = df_quarters.columns.union([prefix + str(x) for x in quarters])
    df_quarters = df_quarters.reindex(cols, axis=1, fill_value=0).astype('uint8')
    # d_week_of_year (1 to 53) - skipped for get_dummies because of high cardinality

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
        logging.debug(f"Chunk {index} has columns with infinity values: "
            f"{non_zero_cts}")
        sys.exit(1)

    # drop binary columns if they are not to be scaled with StandardScaler
    if not scale_bins:
        return df.select_dtypes(exclude="uint8")
    return df


def pca(
    bucket="my-rds-exports",
    chunksize=1000,
    n_components=None,
    first_mon="",
    frac=1.0,
    scale_bins=False,
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

    Returns:
    --------
    None
    """
    csv_list = list_csvs(first_mon=first_mon)
    assert isinstance(csv_list, list), f"csv_list is not a list, but {type(csv_list)}!"

    with open("./features/pd_types_from_psql_mapping.json", "r") as f:
        pd_types = json.load(f)
        pd_types.pop("sale_date")
    # change types of binary features to 'uint8'
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
        "sid_cat_sold_at_shop_before_day_flag",
        "sid_shop_item_first_month",
        "sid_shop_item_first_week",
    )
    pd_types = {k: "uint8" if v in bin_features else v for k, v in pd_types.items()}

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
            if chunk.isna().any().any():
                logging.debug(
                    f"Chunk {index} has {', '.join(chunk.columns[chunk.isna().any()])} columns with nulls"
                )
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
                        index,
                        scale_bins=scale_bins,
                    )
                )
                global_idx = index
            except ValueError:
                unique_dict = {col: chunk[col].unique() for col in chunk.columns}
                logging.debug(f"Unique values in chunk that produced ValueError: {unique_dict}")
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
            scaled_data = scaler.transform(
                preprocess_chunk(
                    chunk.sample(frac=frac, random_state=42).sort_values(
                        by=["shop_id", "item_id", "sale_date"]
                    ),
                    index,
                    scale_bins=scale_bins,
                )
            )
            # pass standard-scaled data (plus binary features if they were not scaled) to PCA partial fit
            if not scale_bins:
                scaled_data = np.hstack(
                    (
                        scaled_data,
                        (
                            chunk.sample(frac=frac, random_state=42)
                            .sort_values(by=["shop_id", "item_id", "sale_date"])
                            .select_dtypes(include="uint8")
                            .to_numpy()
                        ),
                    )
                )
            sklearn_pca.partial_fit(scaled_data)
            global_idx = index

    if n_components is None:
        logging.info(
            f"The estimated number of principal components: {sklearn_pca.n_components_}"
        )
        logging.info(f"The total number of samples seen: {sklearn_pca.n_samples_seen_}")
        plot_pca_components(sklearn_pca, first_mon, frac, scale_bins)

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
                id_cols_arr = chunk_sub[["shop_id", "item_id", "sale_date"]].values
                # after chunk is sampled (above), apply the same transformation and standard scaling
                # pass the scaled data to PCA transform()
                scaled_data = scaler.transform(
                    preprocess_chunk(chunk_sub, index, scale_bins=scale_bins)
                )
                if not scale_bins:
                    scaled_data = np.hstack(
                        (
                            scaled_data,
                            chunk_sub.select_dtypes(include="uint8").to_numpy(),
                        )
                    )

                tx_chunk = sklearn_pca.transform(scaled_data)
                if pca_transformed is None:
                    pca_transformed = np.hstack((id_cols_arr, tx_chunk))
                    shape_list.append(pca_transformed.shape)
                else:
                    tx_chunk = np.hstack((id_cols_arr, tx_chunk))
                    shape_list.append(tx_chunk.shape)
                    pca_transformed = np.vstack((pca_transformed, tx_chunk))

                if pca_transformed.nbytes > 100_000_000:
                    # convert the array to pandas dataframe and upload it to S3
                    # as a parquet file/dataset
                    wr.s3.to_parquet(
                        df=pd.DataFrame(pca_transformed, columns=[
                            "shop_id", "item_id", "sale_date"] +
                            [f"pc{x}" for x in range(1, pca_transformed.shape[1]-2)]
                        ),
                        path="s3://sales-demand-data/parquet_dataset/",
                        index=False,
                        dataset=True,
                        mode="append",
                        partition_cols=["sale_date"],
                    )

                    # also update combined shape of PCA-transformed data
                    overall_shape = (overall_shape[0] + pca_transformed.shape[0],
                    pca_transformed.shape[1])

                    # also update total bytes consumed by PCA-transformed data
                    overall_nbytes += pca_transformed.nbytes

                    # also, reset pca_transformed to None
                    pca_transformed = None

                global_idx = index

        if pca_transformed is not None:
            # convert the array to pandas dataframe and upload it to S3
            # as a parquet file/dataset
            wr.s3.to_parquet(
                df=pd.DataFrame(pca_transformed, columns=[
                    "shop_id", "item_id", "sale_date"] +
                    [f"pc{x}" for x in range(1, pca_transformed.shape[1]-2)]
                ),
                path="s3://sales-demand-data/parquet_dataset/",
                index=False,
                dataset=True,
                mode="append",
                partition_cols=["sale_date"],
            )

            # also update combined shape of PCA-transformed data
            overall_shape = (overall_shape[0] + pca_transformed.shape[0],
            pca_transformed.shape[1])

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
        log_fname = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_{args.command}.log"
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
                + ".csv"
            )
            logging.info(
                f"Running PCA function with chunksize: {args.chunksize}, n_components: {args.comps}, "
                f"first_month: {args.startmonth}, frac: {args.frac}, scale_bins: {args.scale_bins}..."
            )
            pca(
                chunksize=args.chunksize,
                n_components=args.comps,
                first_mon=first_mon,
                frac=args.frac,
                scale_bins=args.scale_bins,
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
