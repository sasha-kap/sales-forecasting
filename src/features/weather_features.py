import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import platform
import sys

import boto3
from botocore.exceptions import ClientError
from ec2_metadata import ec2_metadata
import pandas as pd
import psycopg2
from sqlalchemy.types import Float, INTEGER

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from prep_time_series_data import (
    _add_col_prefix,
    _downcast,
    _map_to_sql_dtypes,
    upload_file,
)

# Import the 'config' function from the config.py file
from utils.config import config
from utils.rds_instance_mgmt import start_instance, stop_instance
from utils.timer import Timer
from utils.write_df_to_sql_table import psql_insert_copy, write_df_to_sql


@Timer(logger=logging.info)
def make_stations_csv():
    """Create a CSV file of Russian weather stations and their GPS
    coordinates and other characteristics.
    """

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

    rus_data.to_csv("../data/rus_weather_stations.csv", sep="\t", index=False)


@Timer(logger=logging.info)
def make_stations_db_table(stop_db=False):
    """Load dataset with Russian weather stations and their characteristics
    to PostgreSQL database.
    """

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


@Timer(logger=logging.info)
def get_shop_to_station_distances(prelim_query=None, stop_db=False):
    """Connect to PostgreSQL database and execute queries to identify
    the nearest weather station to each shop.

    Parameters:
    -----------
    prelim_query : tuple
        Optional; 2-tuple containing SQL query string to delete weather stations that
        do not have complete weather data and a tuple of ID's of the weather
        stations that need to be deleted before the nearest neighbor search
        is performed.
    """

    start_instance()

    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")

        # connect to the PostgreSQL server
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True

        with conn.cursor() as cur:
            sql = (
                "ALTER TABLE shops DROP COLUMN IF EXISTS latlon; "
                "ALTER TABLE shops ADD latlon GEOMETRY; "
                "UPDATE shops SET latlon = ST_SetSRID(ST_MakePoint(s_geo_lon, s_geo_lat), 4326); "
                "ALTER TABLE weather_stations DROP COLUMN IF EXISTS latlon; "
                "ALTER TABLE weather_stations ADD latlon GEOMETRY; "
                "UPDATE weather_stations SET latlon = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326); "
                "SELECT t1.shop_id, t2.station_id, "
                "t2.latlon::geography <-> t1.latlon::geography as distance FROM "
                "(SELECT t1.shop_id as g1, (SELECT t.station_id FROM "
                "weather_stations AS t ORDER BY t.latlon <-> t1.latlon ASC LIMIT 1) "
                "AS g2 FROM shops AS t1) as q JOIN shops AS t1 ON q.g1 = t1.shop_id "
                "JOIN weather_stations AS t2 ON q.g2 = t2.station_id"
            )
            if prelim_query is not None:
                q, params = prelim_query
                logging.debug(f"SQL query to be executed: {cur.mogrify(q, (params,))}")
                cur.execute(q, (params,))
            for q in sql.split("; "):
                logging.debug(f"SQL query to be executed: {q}")
                cur.execute(q)
                logging.debug(f"cur.statusmessage is {cur.statusmessage}")
                if "SELECT" in q:
                    shop_station_distances = {s[0]: s[1:] for s in cur.fetchall()}

        for line in conn.notices:
            logging.debug(line.strip("\n"))

        with open("../data/shop_to_weather_station_map.json", "w") as f:
            json.dump(shop_station_distances, f)

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")

    if stop_db:
        stop_instance()


@Timer(logger=logging.info)
def get_data_summary():
    """Download daily weather data for each weather station for the desired
    period and produce a file summarizing completeness of each station's data.
    """

    with open("../data/shop_to_weather_station_map.json", "r") as f:
        shop_station_distances = json.load(f)

    first_day = datetime.date(2012, 12, 1)
    last_day = datetime.date(2015, 12, 31)

    summary_df = None
    for shop_id, station_info in shop_station_distances.items():
        station_url = (
            f"https://www.ncei.noaa.gov/data/global-historical-"
            f"climatology-network-daily/access/{station_info[0]}.csv"
        )

        station_data = (
            pd.read_csv(station_url, parse_dates=["DATE"])
            .query("@first_day <= DATE <= @last_day")
            .reset_index(drop=True)
        )

        if summary_df is None:
            summary_df = pd.DataFrame(
                station_data.isnull().sum().to_dict(), index=[shop_id]
            )
            summary_df.index.rename("shop_id")
            summary_df["num_rows"] = station_data["DATE"].count()
            summary_df["num_unique_dates"] = station_data["DATE"].nunique()
        else:
            summary_df = pd.concat(
                [
                    summary_df,
                    pd.DataFrame(
                        station_data.isnull().sum().to_dict(), index=[shop_id]
                    ),
                ],
                axis=0,
            )
            summary_df.loc[shop_id, "num_rows"] = station_data["DATE"].count()
            summary_df.loc[shop_id, "num_unique_dates"] = station_data["DATE"].nunique()

        summary_df.to_csv(
            "../data/rus_weather_stations_data_summary.csv", sep=",", index=True
        )


def list_of_stations_to_remove(df):
    """Produce list of weather stations that have incomplete daily
    weather data and that need to be removed from the database.

    Parameters:
    -----------
    df : DataFrame
        Dataframe created from CSV containing data completeness summary for
        each weather station

    Returns:
    --------
    Tuple of weather station ID's or None if no stations need to be removed
    """
    shops_w_zero_weather_data = df[df.num_rows == 0].index.to_list()
    shops_w_missing_prcp_vls = df[df.PRCP > 0].index.to_list()
    shops_w_missing_temp_vls = df[df.TMAX > 0].index.to_list()
    shops_needing_new_station = set(
        [
            *shops_w_zero_weather_data,
            *shops_w_missing_prcp_vls,
            *shops_w_missing_temp_vls,
        ]
    )

    if len(shops_needing_new_station) == 0:
        return None

    with open("../data/shop_to_weather_station_map.json", "r") as f:
        shop_station_distances = json.load(f)
    stations_to_remove = tuple(
        [
            v[0]
            for k, v in shop_station_distances.items()
            if int(k) in shops_needing_new_station
        ]
    )

    return stations_to_remove


@Timer(logger=logging.info)
def revise_nn_stations():
    """Identify stations that have incomplete weather data, delete them
    from weather_stations database table and identify new nearest neighbor
    weather stations, producing new summary file.  Repeat the process until
    every nearest neighbor weather station's data is complete.
    """
    incomplete_data = True
    while incomplete_data:
        # get ids of stations with incomplete data from summary csv
        summary_df = pd.read_csv(
            "../data/rus_weather_stations_data_summary.csv",
            sep=",",
            header=0,
            index_col=0,
        )
        stations_to_remove = list_of_stations_to_remove(summary_df)
        if stations_to_remove is not None:
            # pass query statement to delete those stations from weather_stations table
            # to the get_shop_to_station_distances() function
            # and get new nearest neighbors
            delete_query = "DELETE FROM weather_stations WHERE station_id IN %s"
            get_shop_to_station_distances(
                prelim_query=(delete_query, stations_to_remove)
            )

            # run get_data_summary() function to create updated summary CSV
            # check if all stations have complete data
            # set incomplete_data to False if all stations have complete data
            get_data_summary()
        else:
            incomplete_data = False


@Timer(logger=logging.info)
def get_weather_data_into_db(test_run=False, stop_db=False):
    """Download each weather station's daily weather data from the web and
    upload in consistent format to the PostgreSQL shop_dates_weather table.

    Parameters:
    -----------
    test_run : bool
        Skip actually writing data to DB (if True) and only produce a summary
        of SQL data types for each station's data, or do not skip (if False)
    """
    with open("./data/shop_to_weather_station_map.json", "r") as f:
        shop_station_distances = json.load(f)

    first_day = datetime.date(2012, 12, 1)
    last_day = datetime.date(2015, 12, 31)

    for counter, (shop_id, station_info) in enumerate(
        shop_station_distances.items(), 1
    ):
        station_url = (
            f"https://www.ncei.noaa.gov/data/global-historical-"
            f"climatology-network-daily/access/{station_info[0]}.csv"
        )

        station_data = (
            pd.read_csv(station_url, parse_dates=["DATE"])
            .query("@first_day <= DATE <= @last_day")
            .filter(
                items=[
                    "STATION",
                    "DATE",
                    "LATITUDE",
                    "LONGITUDE",
                    "ELEVATION",
                    "NAME",
                    "PRCP",
                    "PRCP_ATTRIBUTES",
                    "TMAX",
                    "TMAX_ATTRIBUTES",
                    "TMIN",
                    "TMIN_ATTRIBUTES",
                ]
            )
            .reset_index(drop=True)
        )
        station_data.columns = [col.lower() for col in station_data]

        station_data["shop_id"] = shop_id
        station_data["shop_id"] = station_data["shop_id"].astype("uint8")

        station_data["distance"] = station_info[1]

        station_data = _add_col_prefix(station_data, "sdw_")
        station_data.rename(columns={"date": "sale_date"}, inplace=True)

        # next: how to make sure that PRCP, TMAX and TMIN columns are numeric in the station_data DF
        # create actual features in pandas (figure out how to deal with online stores)
        # is creating features here the best idea, or is it better to do some data checks/cleaning first

        # upload table with correct features to PostgreSQL
        station_data = _downcast(station_data)
        dtypes_dict = _map_to_sql_dtypes(station_data)
        logging.debug(
            f"dtypes_dict of counter {counter} and shop_id {shop_id} is: {dtypes_dict}"
        )

        if not test_run:
            # based on results of test run that showed that a couple of columns
            # ended up having different SQL data types in different station data files
            # 'sdw_longitude': Float(precision=3, asdecimal=True) - 55 shops + 'sdw_longitude': SMALLINT() - 3 shops
            dtypes_dict['sdw_longitude'] = Float(precision=3, asdecimal=True)
            # 'sdw_elevation': SMALLINT() - 56 shops + 'sdw_elevation': INTEGER() - 2 shops
            dtypes_dict['sdw_elevation'] = INTEGER()

            start_instance()
            write_df_to_sql(
                station_data,
                "shop_dates_weather",
                dtypes_dict,
                if_exists="replace" if counter == 1 else "append",
            )
    if not test_run and stop_db:
        stop_instance()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'csv', 'db', 'map', 'summary', 'revise', 'to_sql'",
    )
    parser.add_argument(
        "--test_run",
        default=False,
        action="store_true",
        help="run write to SQL code skpping actually writing data to DB (if included) or not skipping (if not)",
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after DB operations (if included) or not (if not included)",
    )

    args = parser.parse_args()

    if args.command not in ("csv", "db", "map", "summary", "revise", "to_sql"):
        print(
            "'{}' is not recognized. "
            "Use 'csv', 'db', 'map', 'summary', 'revise' or 'to_sql'".format(args.command)
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
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # Check if code is being run on EC2 instance (vs locally)
    my_user = os.environ.get("USER")
    is_aws = True if "ec2" in my_user else False
    # Log EC2 instance name and type metadata
    if is_aws:
        instance_metadata = dict()
        instance_metadata['EC2 instance ID'] = ec2_metadata.instance_id
        instance_metadata['EC2 instance type'] = ec2_metadata.instance_type
        instance_metadata['EC2 instance public hostname'] = ec2_metadata.public_hostname

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

    if args.command == "csv":
        make_stations_csv()
    elif args.command == "db":
        make_stations_db_table(stop_db=args.stop)
    elif args.command == "map":
        get_shop_to_station_distances(stop_db=args.stop)
    elif args.command == "summary":
        get_data_summary()
    elif args.command == "revise":
        revise_nn_stations()
    elif args.command == "to_sql":
        get_weather_data_into_db(test_run=args.test_run, stop_db=args.stop)

    # copy log file to S3 bucket
    upload_file(f"./logs/{log_fname}", "my-ec2-logs", log_fname)


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
