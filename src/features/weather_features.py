import argparse
import datetime
import logging
import os
from pathlib import Path
import sys

import pandas as pd
import psycopg2  # import the postgres library

sys.path.insert(1, os.path.join(sys.path[0], ".."))
# Import the 'config' function from the config.py file
from utils.config import config
from utils.rds_instance_mgmt import start_instance, stop_instance
from utils.timer import Timer


def make_stations_csv():

    stations_url = (
        "https://www.ncei.noaa.gov/data/global-historical-"
        "climatology-network-daily/doc/ghcnd-stations.txt"
    )

    df = pd.read_fwf(
        stations_url,
        header=None,
        names=["id", "lat", "lon", "elev", "state", "name", "extra1", "extra2"],
        colspecs=[
            (0, 11),
            (12, 20),
            (21, 30),
            (31, 37),
            (38, 40),
            (41, 71),
            (72, 79),
            (80, 85),
        ],
    )

    logging.info(f"Stations info successfully retrieved from {stations_url}")

    rus_data = df[df.id.str.startswith("RSM")].reset_index(drop=True)

    rus_data.to_csv("../data/rus_weather_stations.csv", sep='\t', index=False)


def make_stations_db_table(stop_db=False):

    start_instance()

    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")

        # connect to the PostgreSQL server
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        # conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        # create a cursor to perform database operations
        # cur = conn.cursor()
        with conn.cursor() as cur, open("../data/rus_weather_stations.csv", "r") as f:
            # ['id', 'lat', 'lon', 'elev', 'state', 'name', 'extra1', 'extra2'],
            sql = (
                "DROP TABLE IF EXISTS weather_stations; "
                "CREATE TABLE weather_stations (station_id varchar(11) UNIQUE NOT NULL, "
                "latitude real NOT NULL, longitude real NOT NULL, elevation real NOT NULL, "
                "state varchar(2), station_name varchar(30), extra1 varchar(7), "
                "extra2 real)"
            )
            for q in sql.split("; "):
                logging.debug(f"SQL query to be executed: {q}")
                cur.execute(q)
                logging.debug(f"cur.statusmessage is {cur.statusmessage}")

            next(f)  # Skip the header row
            # for the command below to work, the single null value (empty string)
            # in the last column (extra2) had to be manually replaced with a
            # real value (just copied from the row just below)
            # otherwise, this exception was raised:
            # psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type real: ""
            cur.copy_from(f, "weather_stations", sep="\t")

        # Make the changes to the database persistent
        conn.commit()

        for line in conn.notices:
            logging.debug(line.strip("\n"))

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")

    if stop_db:
        stop_instance()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", metavar="<command>", help="'csv' or 'db'",
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after DB operations (if included) or not (if not included)",
    )

    args = parser.parse_args()

    if args.command not in ["csv", "db"]:
        print("'{}' is not recognized. " "Use 'csv' or 'db'".format(args.command))

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
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    if args.command == "csv":
        make_stations_csv()
    elif args.command == "db":
        make_stations_db_table(stop_db=args.stop)


if __name__ == "__main__":
    main()


# #create a cursor object
# #cursor object is used to interact with the database
# cur = conn.cursor()
#
# #create table with same headers as csv file
#
#
# cur.execute("CREATE TABLE IF NOT EXISTS test(**** text, **** float, **** float, ****
# text)")
#
# #open the csv file using python standard file I/O
# #copy file into the table just created
# with open('******.csv', 'r') as f:
# next(f) # Skip the header row.
#     #f , <database name>, Comma-Seperated
#     cur.copy_from(f, '****', sep=',')
#     #Commit Changes
#     conn.commit()
#     #Close connection
#     conn.close()