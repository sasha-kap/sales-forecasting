"""
Functions:

run_query
- Run specified SQL queries to create empty table in PostgreSQL and to import
a CSV file from specified S3 bucket to that table

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    python3 s3_to_postgres.py

"""

import datetime
import logging
import os
from pathlib import Path
import sys

import psycopg2

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

        # execute the provided SQL query
        logging.debug(f"SQL query to be executed: {sql_query}")
        cur.execute(sql_query, params)

        # Make the changes to the database persistent
        conn.commit()
        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


def main():

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

    # create new table in PostgreSQL DB
    sql = (
        "DROP TABLE IF EXISTS test_data; "
        "CREATE TABLE test_data (id integer NOT NULL, shop_id smallint NOT NULL, "
        "item_id integer NOT NULL)"
    )
    run_query(sql)

    # import CSV data to created table
    sql = (
        "SELECT aws_s3.table_import_from_s3('test_data', '', '(format csv, header)', "
        "aws_commons.create_s3_uri('sales-demand-data', 'test.csv', 'us-west-2'))"
    )
    run_query(sql)


if __name__ == "__main__":
    main()
