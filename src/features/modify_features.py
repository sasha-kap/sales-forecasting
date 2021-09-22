"""
Contains various SQL queries sent to the RDS PostgreSQL instance to modify
previously created features or add additional rows/features.

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # Run one or more queries to create new features or modify existing
    # features or tables
    python modify_features.py query

"""

import argparse
import datetime
import logging
import os
from pathlib import Path
import sys

import psycopg2
from psycopg2.sql import SQL, Identifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.config import config
from utils.rds_instance_mgmt import start_instance, stop_instance
from utils.timer import Timer

@Timer(logger=logging.info)
def run_query(sql_query, params=None):
    """ Connect to the PostgreSQL database server and run specified query.

    Parameters:
    -----------
    sql_query : str
        SQL query to execute
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method

    Returns:
    --------
    None
    """

    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")

        # connect to the PostgreSQL server
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_params)

        # create a cursor to perform database operations
        cur = conn.cursor()

        logging.debug(f"SQL query to be executed: {sql_query}")

        # execute the provided SQL query
        cur.execute(sql_query, params)
        # Make the changes to the database persistent
        conn.commit()
        # Send list of the session's messages sent to the client to log
        for line in conn.notices:
            logging.debug(line.strip('\n'))
        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'summary', 'primary', 'analyze', 'explain', 'query' or 'drop'",
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after querying (if included) or not (if not included)",
    )

    args = parser.parse_args()

    if args.command not in [
        "query",
    ]:
        print(
            "'{}' is not recognized. "
            "Use 'query'".format(args.command)
        )

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("logs")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    log_fname = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
    log_path = log_dir.joinpath(log_fname)

    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format=fmt,
        datefmt=datefmt,
        filename=log_path,
    )

    # statements to suppress irrelevant logging by boto3-related libraries
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)

    start_instance()

    if args.command == "query":
        # update shop_dates table
        #     - revise expanding and rolling quantity stats columns to change
        #       values in rows for first shop-date to 0
        #     - same for first month and first week columns
        sql = (
            "UPDATE shop_dates "
            "SET sd_shop_expand_qty_max = 0, "
            "sd_shop_expand_qty_mean = 0, "
            "sd_shop_expand_qty_median = 0, "
            "sd_shop_expand_qty_min = 0, "
            "sd_shop_expand_qty_mode = 0, "
            "sd_shop_rolling_7d_avg_qty = 0, "
            "sd_shop_rolling_7d_max_qty = 0, "
            "sd_shop_rolling_7d_median_qty = 0, "
            "sd_shop_rolling_7d_min_qty = 0, "
            "sd_shop_rolling_7d_mode_qty = 0, "
            "sd_shop_first_month = 0, "
            "sd_shop_first_week = 0 "
            "WHERE (shop_id, sale_date) IN ("
            "SELECT shop_id, min(sale_date) "
            "FROM shop_dates "
            "GROUP BY shop_id)"
        )
        run_query(sql)

        # update item_dates table
        #     - revise expanding and rolling quantity stats columns to change
        #       values in rows for first item-date to 0
        #     - same for first month and first week columns
        sql = (
            "UPDATE item_dates "
            "SET id_item_expand_qty_max = 0, "
            "id_item_expand_qty_mean = 0, "
            "id_item_expand_qty_median = 0, "
            "id_item_expand_qty_min = 0, "
            "id_item_expand_qty_mode = 0, "
            "id_item_rolling_7d_avg_qty = 0, "
            "id_item_rolling_7d_max_qty = 0, "
            "id_item_rolling_7d_median_qty = 0, "
            "id_item_rolling_7d_min_qty = 0, "
            "id_item_rolling_7d_mode_qty = 0, "
            "id_item_first_month = 0, "
            "id_item_first_week = 0 "
            "WHERE (item_id, sale_date) IN ("
            "SELECT item_id, min(sale_date) "
            "FROM item_dates "
            "GROUP BY item_id)"
        )
        run_query(sql)

        # update shop_item_dates tables
        #     - revise expanding and rolling quantity stats columns to change
        #       values in rows for first shop-item-date to 0
        #     - same for first month and first week columns
        sql = (
            "UPDATE shop_item_dates "
            "SET sid_shop_item_first_month = 0, "
            "sid_shop_item_first_week = 0 "
            "WHERE (shop_id, item_id, sale_date) IN ("
            "SELECT shop_id, item_id, min(sale_date) "
            "FROM shop_item_dates "
            "GROUP BY shop_id, item_id)"
        )
        run_query(sql)
        sql = (
            "UPDATE sid_expand_qty_stats "
            "SET sid_shop_item_expand_qty_max = 0, "
            "sid_shop_item_expand_qty_mean = 0, "
            "sid_shop_item_expand_qty_median = 0, "
            "sid_shop_item_expand_qty_min = 0, "
            "sid_shop_item_expand_qty_mode = 0 "
            "WHERE (shop_id, item_id, sale_date) IN ("
            "SELECT shop_id, item_id, min(sale_date) "
            "FROM sid_expand_qty_stats "
            "GROUP BY shop_id, item_id)"
        )
        run_query(sql)
        sql = (
            "UPDATE sid_roll_qty_stats "
            "SET sid_shop_item_rolling_7d_avg_qty = 0, "
            "sid_shop_item_rolling_7d_max_qty = 0, "
            "sid_shop_item_rolling_7d_median_qty = 0, "
            "sid_shop_item_rolling_7d_min_qty = 0, "
            "sid_shop_item_rolling_7d_mode_qty = 0 "
            "WHERE (shop_id, item_id, sale_date) IN ("
            "SELECT shop_id, item_id, min(sale_date) "
            "FROM sid_roll_qty_stats "
            "GROUP BY shop_id, item_id)"
        )
        run_query(sql)

        # create new table for additional shop-item-dates, copying all column names,
        # their data types, and their not-null constraints, as well as indexes and
        # primary key constraints from existing shop_item_dates table.
        sql = (
            "CREATE TABLE addl_shop_item_dates (LIKE shop_item_dates INCLUDING INDEXES)"
        )
        run_query(sql)

        # insert shop-item-dates from the LATTER OF the first day an item was sold
        # anywhere AND the first day that the shop sold any item (inclusive) to the
        # day that’s currently in the data when the first observed sale happens for that shop-item
        sql = (
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM sales_cleaned "
            "GROUP BY item_id), "
            "sfd AS ("
            "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
            "FROM sales_cleaned "
            "GROUP BY shop_id), "
            "sifd AS ("
            "SELECT shop_id, item_id, min(sale_date) AS first_shop_item_sale_date "
            "FROM shop_item_dates "
            "GROUP BY shop_id, item_id), "
            "min_max_dts AS ("
            "SELECT sifd.shop_id, sifd.item_id, "
            "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS min_date, "
            "sifd.first_shop_item_sale_date::date - 1 AS max_date "
            "FROM sifd LEFT JOIN ifd "
            "ON sifd.item_id = ifd.item_id "
            "LEFT JOIN sfd "
            "ON sifd.shop_id = sfd.shop_id) "
            "INSERT INTO addl_shop_item_dates (shop_id, item_id, sale_date) "
            "SELECT shop_id, item_id, "
            "generate_series(min_date, max_date, '1 day')::date AS sale_date "
            "FROM min_max_dts"
        )
        run_query(sql)

        # for all shop-items in test data but not in train data, limited to items
        # in train data, insert shop-item-dates between the latter of the first day the
        # item was sold anywhere during train period and the first day that the shop
        # sold any item during the train period to the end of the train period
        sql = (
            # first sale date for each item in train data
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM sales_cleaned "
            "GROUP BY item_id), "
            # first sale date for each shop in train data
            "sfd AS ("
            "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
            "FROM sales_cleaned "
            "GROUP BY shop_id), "
            # "not new" items (items in test data that are also in train data)
            "not_new_items AS ("
            "SELECT DISTINCT item_id "
            "FROM test_data "
            "WHERE item_id IN ("
            "SELECT DISTINCT item_id FROM sales_cleaned)), "
            # shop-items among "not new" items that exist in test but not in train data
            "sist AS ("
            "SELECT td.shop_id, td.item_id "
            "FROM test_data td "
            "INNER JOIN not_new_items nni "
            "ON td.item_id = nni.item_id "
            "LEFT JOIN sales_cleaned sc "
            "ON td.shop_id = sc.shop_id AND td.item_id = sc.item_id "
            "WHERE sc.shop_id IS NULL AND sc.item_id IS NULL), "
            # min and max dates of the interval to be added for each shop-item
            "min_max_dts AS ("
            "SELECT sist.shop_id, sist.item_id, "
            "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS min_date, "
            "make_date(2015,10,31) AS max_date "
            "FROM sist LEFT JOIN ifd "
            "ON sist.item_id = ifd.item_id "
            "LEFT JOIN sfd "
            "ON sist.shop_id = sfd.shop_id) "
            # insert each shop-item's additional dates into addl_shop_item_dates table
            "INSERT INTO addl_shop_item_dates (shop_id, item_id, sale_date) "
            "SELECT shop_id, item_id, "
            "generate_series(min_date, max_date, '1 day')::date AS sale_date "
            "FROM min_max_dts"
        )
        run_query(sql)

        # repeat the above steps for the other sid_ tables
        sid_tables = ['sid_n_sale_dts', 'sid_expand_qty_cv_sqrd',
                'sid_expand_qty_stats', 'sid_roll_qty_stats', 'sid_expand_bw_sales_stats']
        for sid_table in sid_tables:
            sql_str = (
                "DROP TABLE IF EXISTS {0}; "
                "CREATE TABLE {0} (LIKE {1} INCLUDING INDEXES)"
            )
            sql = SQL(sql_str).format(
                Identifier('_'.join([sid_table[:3], 'addl', sid_table[4:]])),
                Identifier(sid_table)
            )
            run_query(sql)

            # insert shop-item-dates from the LATTER OF the first day an item was sold
            # anywhere AND the first day that the shop sold any item (inclusive) to the
            # day that’s currently in the data when the first observed sale happens for that shop-item
            sql_str = (
                "WITH ifd AS ("
                "SELECT item_id, min(sale_date) AS first_item_sale_date "
                "FROM sales_cleaned "
                "GROUP BY item_id), "
                "sfd AS ("
                "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
                "FROM sales_cleaned "
                "GROUP BY shop_id), "
                "sifd AS ("
                "SELECT shop_id, item_id, min(sale_date) AS first_shop_item_sale_date "
                "FROM shop_item_dates "
                "GROUP BY shop_id, item_id), "
                "min_max_dts AS ("
                "SELECT sifd.shop_id, sifd.item_id, "
                "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS min_date, "
                "sifd.first_shop_item_sale_date::date - 1 AS max_date "
                "FROM sifd LEFT JOIN ifd "
                "ON sifd.item_id = ifd.item_id "
                "LEFT JOIN sfd "
                "ON sifd.shop_id = sfd.shop_id) "
                "INSERT INTO {0} (shop_id, item_id, sale_date) "
                "SELECT shop_id, item_id, "
                "generate_series(min_date, max_date, '1 day')::date AS sale_date "
                "FROM min_max_dts"
            )
            sql = SQL(sql_str).format(Identifier('_'.join([sid_table[:3], 'addl', sid_table[4:]])))
            run_query(sql)

            # for all shop-items in test data but not in train data, limited to items
            # in train data, insert shop-item-dates between the latter of the first day the
            # item was sold anywhere during train period and the first day that the shop
            # sold any item during the train period to the end of the train period
            sql_str = (
                # first sale date for each item in train data
                "WITH ifd AS ("
                "SELECT item_id, min(sale_date) AS first_item_sale_date "
                "FROM sales_cleaned "
                "GROUP BY item_id), "
                # first sale date for each shop in train data
                "sfd AS ("
                "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
                "FROM sales_cleaned "
                "GROUP BY shop_id), "
                # "not new" items (items in test data that are also in train data)
                "not_new_items AS ("
                "SELECT DISTINCT item_id "
                "FROM test_data "
                "WHERE item_id IN ("
                "SELECT DISTINCT item_id FROM sales_cleaned)), "
                # shop-items among "not new" items that exist in test but not in train data
                "sist AS ("
                "SELECT td.shop_id, td.item_id "
                "FROM test_data td "
                "INNER JOIN not_new_items nni "
                "ON td.item_id = nni.item_id "
                "LEFT JOIN sales_cleaned sc "
                "ON td.shop_id = sc.shop_id AND td.item_id = sc.item_id "
                "WHERE sc.shop_id IS NULL AND sc.item_id IS NULL), "
                # min and max dates of the interval to be added for each shop-item
                "min_max_dts AS ("
                "SELECT sist.shop_id, sist.item_id, "
                "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS min_date, "
                "make_date(2015,10,31) AS max_date "
                "FROM sist LEFT JOIN ifd "
                "ON sist.item_id = ifd.item_id "
                "LEFT JOIN sfd "
                "ON sist.shop_id = sfd.shop_id) "
                # insert each shop-item's additional dates into addl_shop_item_dates table
                "INSERT INTO {0} (shop_id, item_id, sale_date) "
                "SELECT shop_id, item_id, "
                "generate_series(min_date, max_date, '1 day')::date AS sale_date "
                "FROM min_max_dts"
            )
            sql = SQL(sql_str).format(Identifier('_'.join([sid_table[:3], 'addl', sid_table[4:]])))
            run_query(sql)

        # change null values in newly created values
        # SKIPPING THE 5 QUERIES BELOW FOR NOW, AS THOSE COLUMNS CAN BE SET TO 0'S LATER
        # sql = (
        #     "UPDATE sid_addl_roll_qty_stats "
        #     "SET sid_shop_item_rolling_7d_max_qty = 0, "
        #     "sid_shop_item_rolling_7d_min_qty = 0, "
        #     "sid_shop_item_rolling_7d_avg_qty = 0, "
        #     "sid_shop_item_rolling_7d_mode_qty = 0, "
        #     "sid_shop_item_rolling_7d_median_qty = 0"
        # )
        # run_query(sql)
        # sql = (
        #     "UPDATE sid_addl_expand_qty_stats "
        #     "SET sid_shop_item_expand_qty_max = 0, "
        #     "sid_shop_item_expand_qty_mean = 0, "
        #     "sid_shop_item_expand_qty_min = 0, "
        #     "sid_shop_item_expand_qty_mode = 0, "
        #     "sid_shop_item_expand_qty_median = 0"
        # )
        # run_query(sql)
        # sql = (
        #     "UPDATE sid_addl_expand_bw_sales_stats "
        #     "SET sid_shop_item_date_avg_gap_bw_sales = 0, "
        #     "sid_shop_item_date_max_gap_bw_sales = 0, "
        #     "sid_shop_item_date_min_gap_bw_sales = 0, "
        #     "sid_shop_item_date_mode_gap_bw_sales = 0, "
        #     "sid_shop_item_date_median_gap_bw_sales = 0, "
        #     "sid_shop_item_date_std_gap_bw_sales = 0"
        # )
        # run_query(sql)
        # sql = (
        #     "UPDATE sid_addl_n_sale_dts "
        #     "SET sid_shop_item_cnt_sale_dts_last_7d = 0, "
        #     "sid_shop_item_cnt_sale_dts_last_30d = 0, "
        #     "sid_shop_item_cnt_sale_dts_before_day = 0"
        # )
        # run_query(sql)
        # sql = (
        #     "UPDATE sid_addl_expand_qty_cv_sqrd "
        #     "SET sid_expand_cv2_of_qty = 0"
        # )
        # run_query(sql)
        # SKIPPING THE QUERY BELOW FOR NOW, AS THOSE COLUMNS CAN BE SET TO 0'S LATER
        # sql = (
        #     "UPDATE addl_shop_item_dates "
        #     "SET sid_shop_item_days_since_first_sale = 0, "
        #     "sid_days_since_max_qty_sold = 0, "
        #     "sid_shop_item_qty_sold_day = 0, "
        #     "sid_shop_item_first_month = 0, "
        #     "sid_shop_item_last_qty_sold = 0, "
        #     "sid_shop_item_first_week = 0, "
        #     "sid_shop_item_expanding_adi = 0, "
        #     "sid_shop_item_date_diff_bw_last_and_prev_qty = 0, "
        #     "sid_shop_item_days_since_prev_sale = 0, "
        #     "sid_shop_item_qty_sold_7d_ago = 0, "
        #     "sid_qty_median_abs_dev = 0, "
        #     "sid_coef_var_price = 0, "
        #     "sid_shop_item_qty_sold_2d_ago = 0, "
        #     "sid_qty_mean_abs_dev = 0, "
        #     "sid_shop_item_qty_sold_1d_ago = 0, "
        #     "sid_shop_item_qty_sold_3d_ago = 0"
        # )
        # run_query(sql)
        # THIS ONE JUST BELOW IS DONE
        sql = (
            "UPDATE addl_shop_item_dates asid "
            "SET sid_item_category_id = it.i_item_category_id "
            "FROM ("
            "SELECT item_id, i_item_category_id "
            "FROM items) it "
            "WHERE asid.item_id = it.item_id"
        )
        run_query(sql)

        # add sid_shop_cat_qty_sold_last_7d to shop_cat_dates table,
        # which was inadvertently left out of the original dataset
        sql = (
            "ALTER TABLE shop_cat_dates "
            "DROP COLUMN IF EXISTS sid_shop_cat_qty_sold_last_7d; "
            "ALTER TABLE shop_cat_dates "
            "ADD COLUMN sid_shop_cat_qty_sold_last_7d smallint; "
            "WITH new_col AS ("
            "SELECT shop_id, sid_item_category_id, sale_date, "
            "sum(sid_shop_cat_qty_sold_day) OVER (PARTITION BY shop_id, "
            "sid_item_category_id ORDER BY sale_date "
            "ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS sid_shop_cat_qty_sold_last_7d "
            "FROM shop_cat_dates) "
            "UPDATE shop_cat_dates scd "
            "SET sid_shop_cat_qty_sold_last_7d = COALESCE(nc.sid_shop_cat_qty_sold_last_7d, 0) "
            "FROM new_col nc "
            "WHERE scd.shop_id = nc.shop_id AND scd.sid_item_category_id = nc.sid_item_category_id "
            "AND scd.sale_date = nc.sale_date;"
        )
        run_query(sql)

        # add feature counting days since “first day of availability of item at shop”
        # (days since the LATTER OF the first day the item was sold anywhere AND the
        # first day that the shop sold any item)
        sql = (
            # create counter column in addl_shop_item_dates table
            "ALTER TABLE addl_shop_item_dates "
            "DROP COLUMN IF EXISTS sid_days_since_available; "
            "ALTER TABLE addl_shop_item_dates "
            "ADD COLUMN sid_days_since_available smallint; "
            "WITH rn AS ("
            "SELECT shop_id, item_id, sale_date, "
            "row_number() OVER (PARTITION BY shop_id, item_id "
            "ORDER BY sale_date) - 1 AS sid_days_since_available "
            "FROM addl_shop_item_dates) "
            "UPDATE addl_shop_item_dates asid "
            "SET sid_days_since_available = rn.sid_days_since_available "
            "FROM rn "
            "WHERE asid.shop_id = rn.shop_id AND asid.item_id = rn.item_id AND "
            "asid.sale_date = rn.sale_date; "
            # create counter column in shop_item_dates table
            "ALTER TABLE shop_item_dates "
            "ADD COLUMN sid_days_since_available smallint; "
            "WITH rn AS ("
            "SELECT shop_id, item_id, sale_date, "
            "row_number() OVER (PARTITION BY shop_id, item_id "
            "ORDER BY sale_date) - 1 AS sid_days_since_available "
            "FROM shop_item_dates) "
            "UPDATE shop_item_dates sid "
            "SET sid_days_since_available = rn.sid_days_since_available "
            "FROM rn "
            "WHERE sid.shop_id = rn.shop_id AND sid.item_id = rn.item_id AND "
            "sid.sale_date = rn.sale_date; "
            # add the last counter value from addl_shop_item_dates table to every
            # value in the shop_item_dates table and add 1 to make the counter continuous
            "WITH ldav AS ("
            "SELECT shop_id, item_id, max(sid_days_since_available) AS last_days_avail_value "
            "FROM addl_shop_item_dates "
            "GROUP BY shop_id, item_id) "
            "UPDATE shop_item_dates sid "
            "SET sid_days_since_available = sid_days_since_available + last_days_avail_value + 1 "
            "FROM ldav "
            "WHERE sid.shop_id = ldav.shop_id AND sid.item_id = ldav.item_id;"
        )
        run_query(sql)

    if args.stop:
        stop_instance()

if __name__ == "__main__":
    main()
