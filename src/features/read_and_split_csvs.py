"""
Purpose:
Create train and validation sets for classification (sale/no sale) models.

Perform necessary preprocessing of features in CSV files, merge in weather
features from queried SQL table, convert target (quantity sold) into binary
format (no quantity / positive quantity), perform desired downsampling of
majority class data (0 quantity sold), and split the data into desired number
of train and validation sets, saving the results in separate dataframes.

Copyright (c) 2022 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # Get a list of CSV files (as a preliminary check)
    python read_and_split_csvs.py list 15-07

    # Run full code with Apr 2015 as first month of train data,
    # two months in the first train set, one month in each validation set,
    # 0.2 minority-to-majority class desired ratio, 2 validation sets to be created
    # for validating one model, 50000 chunksize for reading CSVs, 50% sampling rate
    # for creating validation set from CSVs, and with weather features included
    python read_and_split_csvs.py 15-04 2 1 0.2 -v 2 -s 50000 -f 0.5 -w

"""

import argparse
from collections import Counter
from datetime import datetime
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
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from features.dateconstants import LAST_DAY_OF_TRAIN_PRD
from features.weather_features import get_weather_data

s3_client = boto3.client("s3")


# in addition to Prefix=, can also use:
# StartAfter (string) -- StartAfter is where you want Amazon S3 to start listing from.
# Amazon S3 starts listing after this specified key. StartAfter can be any key in the bucket.
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2
# (keys are like 'shops15_10.csv')
def list_csvs(startmonth, bucket="my-rds-exports", prefix="shops"):
    """List CSV files in S3 bucket.

    Parameters:
    -----------
    startmonth : datetime
        First month of CSV data to be included in output
    bucket : str
        S3 bucket containing CSV files (default: 'my-rds-exports')
    prefix : str
        Prefix of CSV file name (S3 key) (default: 'shops')

    Returns:
    --------
    csv_list : list
        List of CSV file names (S3 keys)
    """
    d_prev_mon = startmonth - relativedelta(months=1)
    first_mon = (
        "shops_" + datetime.strftime(d_prev_mon, format="%y_%m") + "_addl" + ".csv"
    )

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


def month_counter(fm):
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


def preprocess_chunk(
    df,
    null_col_dict,
    index,
    cat_col_name_dict,
    train_data,
    sampler=None,
    class_ratio=None,
):
    """Perform necessary preprocessing steps on each chunk of CSV data.

    Parameters:
    -----------
    df : DataFrame
        Chunk of CSV data
    null_col_dict : dict
        Dictionary of columns that have null values in CSVs, with their
        data types that need to be assigned after nulls are filled with 0's
    index : int
        Chunk counter
    cat_col_name_dict : dict
        Dictionary of categorical columns (identified using uint8 data type),
        with keys being original column names and values being same
        names with 'cat_#_' prefix (e.g., cat_1_..., cat_2_..., etc.)
    train_data : bool
        Indicator for whether train (True) or validation (False) data is being
        processed
    sampler : sampling object implementing fit_resample() method (default: None)
        e.g., imblearn.under_sampling.RandomUnderSampler() object
    class_ratio : float
        desired ratio of the number of samples in the minority class over the
        number of samples in the majority class after resampling

    Returns:
    --------
        Dataframe with preprocessed features
        Dictionary mapping original categorical column names to names with
        cat_#_ prefix

    Raises:
    -------
    ValueError
        if sampler object is not provided when train data is passed to the
        function.
    """
    # check that a sampler is provided if working with train data
    if train_data and sampler is None:
        raise ValueError("If processing train data, sampler argument cannot be None.")

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
        + [
            col
            for col in df.columns
            if col.endswith("_qty_sold_day") and col != "sid_shop_item_qty_sold_day"
        ]
        + ["d_day_total_qty_sold"]
        # + ["shop_id", "item_id", "sale_date"]
    )
    # errors='ignore' is added to suppress error when sid_coef_var_price is not found
    # among existing labels
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    # fill columns with null values and change data type from float to the type
    # previously determined for each column
    for col, dtype_str in null_col_dict.items():
        # sid_shop_item_qty_sold_day is NOT dropped above, so it is kept here
        # so null values in that column can also be filled in
        if (
            not col.endswith("_qty_sold_day") or col == "sid_shop_item_qty_sold_day"
        ) and col != "sid_coef_var_price":
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

    # replace previously missed infinity values with 0s in one of the columns
    df["sid_shop_item_expanding_adi"] = df["sid_shop_item_expanding_adi"].replace(
        [-np.inf, np.inf], 0
    )

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

    cat_col_names = [
        "i_item_category_broad",
        "i_item_mon_of_first_sale",
        "d_year",
        "d_day_of_week",
        "d_month",
        "d_quarter_of_year",
        "d_week_of_year",
    ]
    df = pd.concat(
        [
            df.drop(cat_col_names, axis=1,),
            ordinal_encode(
                df[cat_col_names].copy(),
                broad_cats,
                mons_of_first_sale,
                years,
                dow,
                months,
                quarters,
            ),
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

    # rename columns with uint8 type to start with 'cat_#_' prefix
    if cat_col_name_dict is None:
        cat_col_name_dict = {
            col: f"cat_{i}_{col}"
            for i, col in enumerate(df.columns, 1)
            if col in df.select_dtypes("uint8").columns
        }
    df.rename(
        cat_col_name_dict, axis=1, inplace=True,
    )

    # convert quantity values to binary column (1 for positive quantity sold, 0 otherwise)
    # NOTE: this keeps shop-item-dates with negative quantity sold but converts negative
    # values to 0's
    df["sid_shop_item_qty_sold_day"] = (df.sid_shop_item_qty_sold_day > 0).astype(
        "int16"
    )

    # if processing train data
    if train_data:

        # if the cbunk has both classes
        if np.array_equal(df["sid_shop_item_qty_sold_day"].unique(), np.array([0, 1]),):

            # temporarily convert sale_date column from datetime type to int so resampling
            # of numpy array can work
            df["sale_date"] = df["sale_date"].map(datetime.toordinal)

            # downsample majority class (samples with 0 quantity sold)
            resampled_df, _ = sampler.fit_resample(
                df, df["sid_shop_item_qty_sold_day"],
            )

            # convert sale_date back to datetime
            resampled_df["sale_date"] = resampled_df["sale_date"].map(
                datetime.fromordinal
            )

        # if the chunk only has the majority class samples,
        # just sample the inverse of the class_ratio value
        else:
            resampled_df = df.sample(n=int(1 / class_ratio), random_state=42)

        return resampled_df, cat_col_name_dict

    # if processing validation data, do not perform downsampling
    return df, cat_col_name_dict


def ordinal_encode(df, broad_cats, mons_of_first_sale, years, dow, months, quarters):
    """Transforms values in passed dataframe containing only categorical columns
    using OrdinalEncoder.

    Parameters:
    -----------
    df : DataFrame
        Dataframe with categorical columns
    broad_cats : list
        List of item categories
    mons_of_first_sale : list
        List of numeric values of calendar months (0 to 12), with 0 representing
        a special group of items that according to the data were on sale as of
        the first month of available data (Jan 2013), and 1 to 12 representing
        Jan to Dec
    years : list
        List of calendar years (2013-2015)
    dow : list
        List of numeric values of day of week (0 to 6)
    months : list
        List of numeric values of month of year (0 to 11)
    quarters : list
        List of numeric values of quarter of year (1 to 4)

    Returns:
    --------
    Dataframe with ordinal encoder transformed columns
    """

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


def train_test_time_split(
    startmonth,
    n_months_in_first_train_set,
    n_months_in_val_set,
    class_ratio,
    n_val_sets_in_one_month=1,
    bucket="my-rds-exports",
    chunksize=1000,
    frac=1.0,
    add_weather_features=False,
):
    """Read CSVs from S3 and perform time-series train-test split.

    Parameters:
    -----------
    startmonth : datetime
        First month of CSV data to be included in train data
    n_months_in_first_train_set : int
        number of months to be used in first train set during walk-forward validation
    n_months_in_val_set : int
        number of months to be used in each validation set during walk-forward validation
    class_ratio : float
        desired ratio of the number of samples in the minority class over the
        number of samples in the majority class after resampling
    n_val_sets_in_one_month : int
        number of validation sets to be used in validating one model (default: 1)
    bucket : str
        S3 bucket containing CSV data (default: 'my-rds-exports')
    chunksize : int
        Number of rows to include in each iteration of read_csv() (default: 1000)
    frac : float
        Fraction of rows to sample from validation CSVs (default: 1.0)
    add_weather_features : bool
        Indicator for whether to query and merge in weather-related features
        (default: False)

    Yields:
    --------
    accumulated_train_data : DataFrame
        train data dataframe
    processed_val_data_list : list of DataFrames
        list of validation data dataframes
    """
    csv_list = list_csvs(startmonth)
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

    cat_col_name_dict = None
    types_map = {
        "int64": "bigint",
        "int32": "int",
        "int16": "smallint",
        "float32": "float",
        "float64": "float",
        "uint8": "smallint",
    }

    n_val_sets = (
        month_counter(startmonth)  # startmonth is e.g. May 1, 2015
        - n_months_in_first_train_set
        + 1
    ) - (n_months_in_val_set - 1)
    logging.debug(
        f"Train-test split function will generate {n_val_sets} train-test splits."
    )

    # variable to hold names of CSVs read in for train data
    csvs_already_read = set()

    # variable to hold accumulated train data read in from multiple train-test splits
    accumulated_train_data = None

    # variable to hold names of columns in train/validation data that start with 'cat'
    cat_cols = None

    # variable to hold dataframe with weather features
    weather_df = None

    # initialize undersampler (of negative class) for train data
    under = RandomUnderSampler(sampling_strategy=class_ratio, random_state=42)

    # loop over however many train-validation splits need to be done
    logging.info("Starting loop over train-validation splits...")
    for m in range(n_val_sets):
        # variable to hold processed validatiaon data
        processed_val_data = None

        # identify CSV files that have not been read yet for train set and read them in
        end_date = startmonth + relativedelta(
            months=m + n_months_in_first_train_set - 1, day=31
        )
        logging.debug(f"End date of current train set is {end_date.date()}")
        full_train_csv_list = list(
            filter(
                lambda x: startmonth <= datetime.strptime(x[6:11], "%y_%m") <= end_date,
                csv_list,
            )
        )
        csvs_to_get = [
            csv for csv in full_train_csv_list if csv not in csvs_already_read
        ]
        logging.debug(
            f"CSVs that will be read from S3 for current train set are "
            f"{', '.join(csvs_to_get)}."
        )

        # identify CSV files that need to be read in for validation set
        val_set_csvs = list(
            filter(
                lambda x: end_date
                < datetime.strptime(x[6:11], "%y_%m")
                <= end_date + relativedelta(months=n_months_in_val_set, day=31),
                csv_list,
            )
        )
        logging.debug(
            f"CSVs that will be read from S3 for current validation set are "
            f"{', '.join(val_set_csvs)}."
        )

        # loop over CSV files that need to be read for current train set
        train_data = True
        start_time = time.perf_counter()
        for csv_file in csvs_to_get:
            logging.debug(f"Starting to read {csv_file} file for train set from S3...")
            csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
            # read each CSV file in chunks, process each chunk and concatenate all chunks
            for index, chunk in enumerate(
                pd.read_csv(
                    csv_body,
                    chunksize=chunksize,
                    dtype=pd_types,
                    parse_dates=["sale_date"],
                ),
            ):
                preprocessed_chunk, cat_col_name_dict = preprocess_chunk(
                    chunk.sort_values(by=["shop_id", "item_id", "sale_date"]),
                    null_col_dict,
                    index,
                    cat_col_name_dict,
                    train_data,
                    sampler=under,
                    class_ratio=class_ratio,
                )

                if accumulated_train_data is None:
                    accumulated_train_data = preprocessed_chunk
                else:
                    accumulated_train_data = pd.concat(
                        [accumulated_train_data, preprocessed_chunk],
                        axis=0,
                        ignore_index=True,
                    )

            logging.info(
                f"{csv_file} file for train set was read and processed in "
                f"{round(time.perf_counter() - start_time, 2)} seconds."
            )
            start_time = time.perf_counter()

        # add names of CSVs read in above to list of all CSVs already read in
        csvs_already_read.update(csvs_to_get)
        logging.debug(
            f"CSV files already read for train data are "
            f"{', '.join(sorted(list(csvs_already_read)))}."
        )

        # merge in weather features
        if add_weather_features:
            if cat_cols is None:
                cat_cols = [
                    col for col in accumulated_train_data if col.startswith("cat")
                ]
            if weather_df is None:
                weather_df = get_weather_data(startmonth, datetime(*LAST_DAY_OF_TRAIN_PRD), cat_cols)
            cols_to_drop = [
                col for col in weather_df.columns.tolist() if col not in ("shop_id", "sale_date")
            ]
            accumulated_train_data = accumulated_train_data.drop(
                cols_to_drop,
                axis=1,
                errors='ignore', # this is needed before the first merge with weather_df
            ).merge(weather_df, on=["shop_id", "sale_date"], how="left",)

        # check that only 1's and 0's are in the target column in train dataset
        if not np.array_equal(
            accumulated_train_data["sid_shop_item_qty_sold_day"].unique(),
            np.array([0, 1]),
        ):
            logging.debug(
                f"Target column in train data has some extraneous values: "
                f"{Counter(accumulated_train_data['sid_shop_item_qty_sold_day'])}."
            )
            sys.exit(1)

        logging.debug(
            f"Shape of prepared train dataset is {accumulated_train_data.shape}."
        )
        logging.debug(
            f"Memory usage of prepared train dataset is "
            f"{accumulated_train_data.memory_usage(deep=True).sum()}."
        )
        logging.debug(
            f"Counts of majority and minority classes in processed and sampled "
            f"train data: {Counter(accumulated_train_data['sid_shop_item_qty_sold_day'])}"
        )

        # loop over CSV files that need to be read for current validation set
        train_data = False
        start_time = time.perf_counter()
        for csv_file in val_set_csvs:
            logging.debug(
                f"Starting to read {csv_file} file for validation set from S3..."
            )
            csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
            # read each CSV file in chunks, process each chunk and concatenate all chunks
            for index, chunk in enumerate(
                pd.read_csv(
                    csv_body,
                    chunksize=chunksize,
                    dtype=pd_types,
                    parse_dates=["sale_date"],
                ),
            ):
                preprocessed_chunk, cat_col_name_dict = preprocess_chunk(
                    chunk.sample(frac=frac, random_state=42).sort_values(
                        by=["shop_id", "item_id", "sale_date"]
                    ),
                    null_col_dict,
                    index,
                    cat_col_name_dict,
                    train_data,
                )

                if processed_val_data is None:
                    processed_val_data = preprocessed_chunk
                else:
                    processed_val_data = pd.concat(
                        [processed_val_data, preprocessed_chunk],
                        axis=0,
                        ignore_index=True,
                    )

            logging.info(
                f"{csv_file} file for validation set was read and processed in "
                f"{round(time.perf_counter() - start_time, 2)} seconds."
            )
            start_time = time.perf_counter()

        # merge in weather features
        if add_weather_features:
            processed_val_data = processed_val_data.merge(
                weather_df, on=["shop_id", "sale_date"], how="left",
            )

        # check that only 1's and 0's are in the target column in validation data
        if not np.array_equal(
            processed_val_data["sid_shop_item_qty_sold_day"].unique(), np.array([0, 1])
        ):
            logging.debug(
                f"Target column in validation data has some extraneous values: "
                f"{Counter(processed_val_data['sid_shop_item_qty_sold_day'])}."
            )
            sys.exit(1)

        # check that train and validation sets have same columns
        if sorted(accumulated_train_data.columns.tolist()) != sorted(
            processed_val_data.columns.tolist()
        ):
            logging.debug(
                "Train and validation sets have different columns, see below."
            )
            logging.debug(
                f"Columns in train data, but not in validation data: "
                f"{', '.join(accumulated_train_data.columns.difference(processed_val_data.columns))}."
            )
            logging.debug(
                f"Columns in validation data, but not in train data: "
                f"{', '.join(processed_val_data.columns.difference(accumulated_train_data.columns))}."
            )
            sys.exit(1)

        logging.debug(
            f"Shape of prepared validation dataset is {processed_val_data.shape}."
        )
        logging.debug(
            f"Memory usage of prepared validation dataset is "
            f"{processed_val_data.memory_usage(deep=True).sum()}."
        )

        # if need more than one validation set, shuffle and split validation data into
        # specified number of chunks
        if n_val_sets_in_one_month > 1:
            shuffled_val_data = processed_val_data.sample(frac=1).reset_index(drop=True)
            processed_val_data_list = np.array_split(
                shuffled_val_data, n_val_sets_in_one_month
            )
        else:
            processed_val_data_list = list()
            processed_val_data_list.append(processed_val_data)

        logging.debug(
            f"Counts of majority and minority classes in processed validation data: "
            f"{', '.join([str(Counter(df['sid_shop_item_qty_sold_day'])) for df in processed_val_data_list])}"
        )

        yield accumulated_train_data, processed_val_data_list


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


def valid_frac(s):
    """Convert command-line fraction argument to float value.

    Parameters:
    -----------
    s : str
        Command-line argument for fraction of rows to sample or
        ratio of minority class samples to majority class samples

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
        "command", metavar="<command>", help="'list' or 'process'",
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
        "class_ratio",
        metavar="<class_ratio>",
        help="desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling",
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
        "-s",
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

    if args.command not in ["list", "process"]:
        print("'{}' is not recognized. " "Use 'list' or 'process'".format(args.command))

    else:

        fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        log_dir = Path.cwd().joinpath("logs")
        path = Path(log_dir)
        path.mkdir(exist_ok=True)
        curr_dt_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
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
            logging.info(
                f"Running list function with first_month: {args.startmonth}..."
            )
            print(list_csvs(args.startmonth))

        elif args.command == "process":
            logging.info(
                f"Running process function with chunksize: {args.chunksize}, "
                f"first_month: {args.startmonth}, frac: {args.frac}, "
                f"n_months_in_first_train_set: {args.n_months_in_first_train_set}, "
                f"n_months_in_val_set: {args.n_months_in_val_set}, "
                f"class_ratio: {args.class_ratio}, "
                f"n_val_sets_in_one_month: {args.n_val_sets_in_one_month}, "
                f"weather: {args.weather}..."
            )
            train_test_time_split(
                args.startmonth,
                args.n_months_in_first_train_set,
                args.n_months_in_val_set,
                args.class_ratio,
                n_val_sets_in_one_month=args.n_val_sets_in_one_month,
                chunksize=args.chunksize,
                frac=args.frac,
                add_weather_features=args.weather,
            )

        # copy log file to S3 bucket
        try:
            s3_client.upload_file(f"./logs/{log_fname}", "my-ec2-logs", log_fname)
        except ClientError:
            logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()
