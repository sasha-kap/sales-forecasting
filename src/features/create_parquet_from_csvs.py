"""
Perform necessary preprocessing of features in CSV files, subset data as
necessary based on values of target (quantity sold) and write resulting
dataset to date-partitioned Parquet dataset in AWS s3 bucket.

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # Get a list of CSV files (as a preliminary check)
    python create_parquet_from_csvs.py list 15-07

    # Run full code with 10000 chunksize, 50% sampling rate, and subsetting
    # dataset to quantity sold values greater than 0
    python create_parquet_from_csvs.py process 15-10 -s 10000 -b gt0 -f 0.5

"""

import argparse
import datetime
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

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


def preprocess_chunk(df, null_col_dict, index, cat_col_name_dict):
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

    Returns:
    --------
        Dataframe with preprocessed features
        Dictionary mapping original categorical column names to names with
        cat_#_ prefix
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
        # sid_shop_item_qty_sold_day is already dropped above, so it can be excluded
        # from this step
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
    df["sid_shop_item_expanding_adi"] = df[
        "sid_shop_item_expanding_adi"
    ].replace([-np.inf, np.inf], 0)

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

    return df, cat_col_name_dict


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


def save_to_parquet(
    bucket="my-rds-exports",
    chunksize=1000,
    first_mon="",
    subset="",
    frac=1.0,
):
    """.

    Parameters:
    -----------
    bucket : str
        S3 bucket containing CSV data (default: 'my-rds-exports')
    chunksize : int
        Number of rows to include in each iteration of read_csv() (default: 1000)
    first_mon : str
        First month of CSV data included in PCA (all, if '' (default))
    subset : str
        Optional, used to subset dataset to desired range of target values
    frac : float
        Fraction of rows to sample from CSVs (default: 1.0)

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

    start_time = current_time = time.perf_counter()
    processed_data = None
    overall_shape = (0, None)
    shape_list = list()
    cat_col_name_dict = None
    s3_path = f"s3://sales-demand-data/parquet_dataset_w_cat_cols_{subset}/"
    types_map = {
        "int64": "bigint",
        "int32": "int",
        "int16": "smallint",
        "float32": "float",
        "float64": "float",
        "uint8": "smallint",
    }

    for csv_file in csv_list:
        csv_body = s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body")
        for index, chunk in enumerate(
            pd.read_csv(
                csv_body, chunksize=chunksize, dtype=pd_types, parse_dates=["sale_date"]
            ),
        ):
            if index == 0:
                logging.debug(
                    f"Columns in DF converted from CSV: {list(enumerate(chunk.columns))}"
                )
            if index % 100 == 0:
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

            # try:
            if subset == "gt0":
                chunk = chunk[chunk.sid_shop_item_qty_sold_day > 0]
            elif subset == "gt0_lt6":
                chunk = chunk[
                    (chunk.sid_shop_item_qty_sold_day > 0)
                    & (chunk.sid_shop_item_qty_sold_day < 6)
                ]
            if chunk.shape[0] == 0:
                continue

            preprocessed_chunk, cat_col_name_dict = preprocess_chunk(
                chunk.sample(frac=frac, random_state=42).sort_values(
                    by=["shop_id", "item_id", "sale_date"]
                ),
                null_col_dict,
                index,
                cat_col_name_dict,
            )
            # except ValueError:
            #     unique_dict = {col: chunk[col].unique() for col in chunk.columns}
            #     logging.debug(
            #         f"Unique values in chunk that produced ValueError: {unique_dict}"
            #     )
            #     sys.exit(1)

            if processed_data is None:
                processed_data = preprocessed_chunk
            else:
                processed_data = pd.concat([processed_data, preprocessed_chunk], axis=0)
            shape_list.append(preprocessed_chunk.shape)

            if processed_data.memory_usage(deep=True).sum() > 100_000_000:
                # upload dataframe to S3
                # as a parquet dataset
                dtype_dict = {
                    k: types_map[v]
                    for k, v in processed_data.dtypes.map(str).to_dict().items()
                    if v != 'datetime64[ns]'
                }
                wr.s3.to_parquet(
                    df=processed_data,
                    path=s3_path,
                    index=False,
                    dataset=True,
                    mode="append",
                    partition_cols=["sale_date"],
                    # https://docs.aws.amazon.com/athena/latest/ug/data-types.html
                    dtype=dtype_dict,
                )

                # also update combined shape of preprocessed data
                overall_shape = (
                    overall_shape[0] + processed_data.shape[0],
                    processed_data.shape[1],
                )

                # also, reset processed_data to None
                processed_data = None

    if processed_data is not None:
        # upload dataframe to S3
        # as a parquet dataset
        dtype_dict = {
            k: types_map[v] for k, v in processed_data.dtypes.map(str).to_dict().items()
            if v != 'datetime64[ns]'
        }
        wr.s3.to_parquet(
            df=processed_data,
            path=s3_path,
            index=False,
            dataset=True,
            mode="append",
            partition_cols=["sale_date"],
            # https://docs.aws.amazon.com/athena/latest/ug/data-types.html
            dtype=dtype_dict,
        )

        # also update combined shape of preprocessed data
        overall_shape = (
            overall_shape[0] + processed_data.shape[0],
            processed_data.shape[1],
        )

    print(f"Final shape of preprocessed data: {overall_shape}")

    shape_arr = np.array(shape_list)
    print(f"Size of shape_arr: {shape_arr.shape}")

    if np.max(shape_arr[:, 1]) != np.min(shape_arr[:, 1]):
        logging.debug("Different chunks have different counts of columns!!!")
    if (np.sum(shape_arr[:, 0]), np.min(shape_arr[:, 1])) != overall_shape:
        logging.debug(
            "Final shape of preprocessed data does not "
            "match the combined shape of individual chunks!!!"
        )


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
        "command", metavar="<command>", help="'list' or 'process'",
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
        "--subset",
        "-b",
        help="how to subset the data based on value of target (sid_shop_item_qty_sold_day)",
        choices=["gt0", "gt0_lt6"],
    )
    parser.add_argument(
        "--frac",
        "-f",
        help="fraction of rows to sample (default is 1.0 if omitted)",
        default="1.0",
        type=valid_frac,
    )

    args = parser.parse_args()

    if args.command not in ["list", "process"]:
        print("'{}' is not recognized. " "Use 'list' or 'process'".format(args.command))

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

        elif args.command == "process":
            d_prev_mon = args.startmonth - relativedelta(months=1)
            first_mon = (
                "shops_"
                + datetime.datetime.strftime(d_prev_mon, format="%y_%m")
                + "_addl"
                + ".csv"
            )
            logging.info(
                f"Running process function with chunksize: {args.chunksize}, "
                f"first_month: {args.startmonth}, subset value: {args.subset}, "
                f"frac: {args.frac}..."
            )
            save_to_parquet(
                chunksize=args.chunksize,
                first_mon=first_mon,
                subset=args.subset,
                frac=args.frac,
            )

        # copy log file to S3 bucket
        try:
            s3_client.upload_file(
                f"./logs/{log_fname}", "my-ec2-logs", log_fname
            )
        except ClientError:
            logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()
