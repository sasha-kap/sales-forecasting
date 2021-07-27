'''
Steps:
- create shop-item-date level dataset out of shop-item dataset
    - need to somehow select the first day of sale for new items
- write code to create features for one day's worth of shop-items at a time
    using data from before that day

UPDATE table2 t2
SET    val2 = t1.val1
FROM   table1 t1
WHERE  t2.table2_id = t1.table2_id

UPDATE summary s SET (sum_x, sum_y, avg_x, avg_y) =
    (SELECT sum(x), sum(y), avg(x), avg(y) FROM data d
     WHERE d.group_id = s.group_id);

UPDATE table t1 SET column1=sq.column1
FROM  (
   SELECT t2.column1, column2
   FROM   table t2
   JOIN   table t3 USING (column2)
   GROUP  BY column2
   ) AS sq
WHERE  t1.column2=sq.column2;

TO DO:
- update SQL data types in all queries - DONE
- expanding quantity sold stats: assign 0 default value (which will apply to first item-day, shop-day, shop-item-day) - DONE
- add script to create the sid_days_since_available feature in test data - DONE
    - for shop-items in train data, continue counting from last value
    - for new shop-items during the test period, assign 0's to Nov 1 shop-items and count up from there
- add sale date column to id_new_day, sd_new_day and sid_new_day tables - DONE
- convert make_date() function in queries to date parameter: datetime.date(2015,11,1) - DONE
    to be used a parameter to be passed to the execute() method
        - make_date(___) becomes %(curr_date)s - DONE
        - params becomes {'curr_date': datetime.date(2015,11,1)} - DONE
        - example (from https://www.psycopg.org/docs/usage.html#passing-parameters-to-sql-queries):
            >>> cur.execute("""
            ...     INSERT INTO some_table (an_int, a_date, another_date, a_string)
            ...     VALUES (%(int)s, %(date)s, %(date)s, %(str)s);
            ...     """,
            ...     {'int': 10, 'str': "O'Reilly", 'date': datetime.date(2005, 11, 18)})
- figure out some way to tell the script to print "done" after each individual query
    - probably should make a list/tuple of query strings and then loop over them and call
      cur.execute() on each one, committing each data manipulation step
        - SPLIT THE FULL query STRING INTO SEPARATE ONES, FIRST ON 'DROP TABLE', THEN ON 'ALTER':
            - example:
                 >>> s = "ALTER a b c; ALTER d e f"
                 >>> [("ALTER"+x).strip() for x in s.split("ALTER")][1:]
                 ['ALTER a b c;', 'ALTER d e f']
        - create separate connections for item-date, shop-date and shop-item-date queries
    - need to add a progress monitor to report percentage of queries done
    - need to produce some type of summary of predictions for each day before proceeding
        to the next day, with the ability to stop the process if one day's predictions
        look way off
            - can check if distribution of daily predictions has some kind of undesired trend
- check if any queries using shop_item_dates table can instead use the sales_cleaned table
    - i.e., when they use _qty_sold_day values that are <> 0 OR > 0
        - sd_num_unique_item_cats_prior_to_day
            - modified the code to use the sales_cleaned table, but this will require
                either updating that table with non-zero predicted shop-item quantities
                or creating a separate table with those non-zero predicted shop-item quantities
                (it will probably be a good idea to store predicted values in separate tables anyway, not
                to append them to existing tables)
        - same for sid_shop_item_last_qty_sold
        - same for sid_shop_item_date_diff_bw_last_and_prev_qty
        - same for sid_qty_mean_abs_dev
        - same for sid_qty_median_abs_dev
        - same for the three sid_shop_item_cnt_sale_dts_... columns
        - same for sid_shop_item_days_since_first_sale and sid_shop_item_days_since_prev_sale
        - same for sid_expand_cv2_of_qty
- write code to compute expanding coefficient of variation of price across dates with a sale for each shop-item
    for new shop-items in test data
    - investigate values of this feature by category-day during the first month
      that an item is on sale (i.e., from day 1 through day 30)
    - not sure if feasible to calculate this feature for new shop-items without having price
- look into parallelizing the SQL code via multiple connections/cursors
    - use autocommit (conn.autocommit = True)
    - threading
        - create one function for any query, then pass that function and the right set of queries to each thread
            - save each set of queries in its own .sql file
- write the script for the whole prediction pipeline
    - create separate tables of features (id_new_day, sd_new_day, etc.)
    - export the data out of Postgres as one joined table
        - write assert statement to make sure there are no null values in exported data
    - append individual features to the right existing tables (e.g., id_… features get appended to the item_dates table)
        - THOSE APPENDED ROWS NEED TO BE DELETED BEFORE NEXT MODEL
    - predict first day in test period
    - use prediction to generate features for second day in test period (i.e. data for every shop-item for second day)
        - once shop-item-day predictions are generated, they need to be aggregated to item-day, shop-day and day levels
            (to get id_item_qty_sold_day, sd_shop_qty_sold_day and d_day_total_qty_sold values) and appended to item_dates, shop_dates and dates tables
            so that next day's item-date, shop-date and date features can be created (such as id_item_days_since_prev_sale and
            id_item_qty_sold_7d_ago and d_day_total_qty_sold_1day_lag)
    - predict second day, etc.

- MAKE IT POSSIBLE TO RUN PREDICTIONS ON A FEW DAYS AT A TIME AND RESUME FROM STOP POINT

- make a diagram of how/when features and predictions are added to/deleted from existing tables
for first model and later models

- assign flag for first model or not first model
- loop over 30 days (Nov 1 thru Nov 30)
- create separate tables of features for the current day in the loop
- export the data out of Postgres as one joined table

PSYCOPG2 LOGGING CONNECTION

- how to run multiple queries in an efficient manner
    - if running different queries using different connections/cursors, what happens
    if one query fails
        - or one thread fails if using threading
    - what happens if there is a drop in network connection
    - add sys.exit() for when an exception occurs with a query
- how to check one day's worth of predicted values and stop the script
    - probably check the values before they are loaded back to Postgres
    - automatically, exit the script if some kind of flag is raised on those values

- ADD A CHECK FOR QUERY RETURNING CORRECT OBJECT, E.G.,:
        if curs.fetchone() is None:
            raise ValueError("account belonging to user not found")
- CREATE A FUNCTION PER SEPARATE SECTION OF QUERIES AND PASS IT THE CONNECTION AS ARGUMENT
    - ALSO PASS ANY ARGUMENTS TO THE QUERY AS ARGUMENTS TO THE FUNCTION, E.G.,
            def verify_account(conn, user, account):
                """
                Verify that the account is held by the user.
                """
                with conn.cursor() as curs:
                    sql = (
                        "SELECT 1 AS verified FROM accounts a "
                        "JOIN users u on u.id = a.owner_id "
                        "WHERE u.username=%s AND a.id=%s"
                    )
                    curs.execute(sql, (user, account))
- PUT SQL QUERIES INTO A .sql FILE AND READ IT WHEN NEED TO EXECUTE A QUERY, FOR EXAMPLE:
        def createdb(conn, schema="schema.sql"):
            with open(schema, 'r') as f:
                sql = f.read()
            try:
                with conn.cursor() as curs:
                    curs.execute(sql)
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
- USE CONTEXT MANAGER FOR QUERY EXECUTION, FOR EXAMPLE:
        def check_daily_deposit(conn, account):
            """
            Raise an exception if the deposit limit has been exceeded.
            """
            with conn.cursor() as curs:
                sql = (
                    "SELECT amount FROM ledger "
                    "WHERE date=now()::date AND type='credit' AND account_id=%s"
                )
                curs.execute(sql, (account,))
                total = sum(row[0] for row in curs.fetchall())
                if total > MAX_DEPOSIT_LIMIT:
                    raise Exception("daily deposit limit has been exceeded!")

ADD CONN.NOTICES OUTPUT TO QUERIES
        # execute the provided SQL query
        cur.execute(sql_query, params)
        # Make the changes to the database persistent
        conn.commit()
        # Send list of the session's messages sent to the client to log
        for line in conn.notices:
            logging.debug(line.strip("\n"))
        # close the communication with the PostgreSQL
        cur.close()
'''

import argparse
import csv
import datetime
import logging
import os
from pathlib import Path
import platform
import sys
import threading

import boto3
from botocore.exceptions import ClientError
from ec2_metadata import ec2_metadata
import psycopg2
from psycopg2.sql import SQL, Identifier

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils.config import config
from utils.rds_instance_mgmt import start_instance, stop_instance
from utils.timer import Timer
from queries import lag_query, all_queries_str

# create database connection in the __init__ method
# thread_function completes the SQL commands
# run function calls thread_function with individual parts of the list of SQL queries

class multi_thread_db_class(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
        """Create database connection for each thread separately.

        Parameters:
        -----------
        args : tuple
            SQL queries dedicated to the thread
        kwargs : dictionary
            query parameters to submit to cur.execute() method
        """
        super().__init__()
        self.args = args
        self.kwargs = kwargs

        self.stop_requested = threading.Event()
        self.exception = None

        try:
            # read connection parameters
            db_params = config(section="postgresql")

            # connect to the PostgreSQL server
            self.conn = psycopg2.connect(**db_params)
            self.conn.autocommit = True
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
            logging.info("Connected to the PostgreSQL database.")

        except (Exception, psycopg2.DatabaseError) as e:
            logging.exception("Exception occurred during database connection.")
            self.exception = e

    @Timer(logger=logging.info)
    def execute_query(self, sql):
        """ Execute individual SQL query.

        Parameters:
        -----------
        sql : str
            SQL query to be executed
        """
        try:
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                cur.execute(sql, self.kwargs)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as e:
            logging.exception(
                "Exception occurred in execute_query function. "
                f"Query that generated exception: {sql}"
            )
            self.close_db_conn()
            self.exception = e

    def close_db_conn(self):
        """ Close PostgreSQL database connection."""
        if self.conn is not None:
            self.conn.close()
            logging.info("Database connection closed.")

    def run(self):
        """ Method representing the thread’s activity, which is to loop over
        SQL queries assigned to the thread and call the execute_query() function
        for each query, closing the database connection after all queries are done.
        """
        try:
            i = 0
            # run execute_query on each query in list dedicated to thread
            while i < len(self.args):
                # do your thread thing here
                query = self.args[i]
                self.execute_query(query)
                i += 1

            global n_threads_finished
            n_threads_finished += 1

        except (Exception, psycopg2.DatabaseError) as e:
            logging.exception("Exception occurred in run() function.")
            self.exception = e

        finally:
            self.close_db_conn()

    def stop(self):
        """ Set the event to signal stop. """
        self.stop_requested.set()


def valid_day(s):
    """Convert command-line date argument to YY-MM-DD datetime value.

    Parameters:
    -----------
    s : str
        Command-line argument for first day for which to run predictions

    Returns:
    --------
    Datetime.date object

    Raises:
    -------
    ArgumentTypeError
        if input string cannot be converted into a valid date in November 2015
    """
    try:
        return datetime.date(2015, 11, int(s))
    except ValueError:
        msg = f"Not a valid day in November: {s}."
        raise argparse.ArgumentTypeError(msg)


class single_thread_db_class:
    def __init__(self, is_aws, log_fname):
        """Create database connection.

        Parameters:
        -----------
        is_aws : bool
            Indicator for whether script is running on a AWS EC2 instance,
            default: false
        log_fname : str
            Name of file where log will be sent
        """
        self.is_aws = is_aws
        self.log_fname = log_fname
        if self.is_aws:
            self.s3_client = boto3.client("s3")
        try:
            # read connection parameters
            db_params = config(section="postgresql")

            # connect to the PostgreSQL server
            self.conn = psycopg2.connect(**db_params)
            self.conn.autocommit = True
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
            logging.info("Connected to the PostgreSQL database.")

        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred during database connection.")
            sys.exit(1)

    def close_db_conn(self):
        """ Close PostgreSQL database connection."""
        if self.conn is not None:
            self.conn.close()
            logging.info("Database connection closed.")

    @Timer(logger=logging.info)
    def query_col_names(self, csv_path, csv_fname):
        """ Query info on columns in existing tables, saving output to CSV file.

        Parameters:
        -----------
        csv_path : str or pathlib.Path() object
            Filepath (directory and filename) where to save query results
        csv_fname : str
            Name of file (including .csv extension) to contain query output

        Returns:
        --------
        None
        """
        try:
            with self.conn.cursor() as cur:
                sql = (
                    "SELECT table_schema, table_name, "
                    "column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema "
                    "NOT IN ('pg_catalog', 'information_schema')"
                )
                outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(sql)

                with open(csv_path, "w") as f:
                    cur.copy_expert(outputquery, f)

        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred in query_col_names function.")
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

        else:
            # copy CSV file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./csv/{csv_fname}", "my-ec2-logs", csv_fname
                    )
                    logging.info(
                        "CSV file of column names in _new_day "
                        "tables successfully copied to S3."
                    )
                except ClientError as e:
                    logging.exception(
                        "CSV file of columnn names in _new_day "
                        "tables was not copied to S3."
                    )

    @Timer(logger=logging.info)
    def delete_from_sales_cleaned(self):
        """ Delete rows from previous model (deletion to only happen once per model)
        in the sales_cleaned table.
        """
        try:
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql = (
                    "DELETE FROM sales_cleaned WHERE sale_date >= make_date(2015,11,1)"
                )
                cur.execute(sql)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception(
                "Exception occurred in delete_from_sales_cleaned function."
            )
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def delete_test_prd(self):
        """ Delete rows for the test period from feature-containing tables.
        This step must precede insertion of features into those tables.
        """
        try:
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql = (
                    "DELETE FROM item_dates "
                    "WHERE sale_date >= make_date(2015,11,1); "
                    "DELETE FROM shop_dates "
                    "WHERE sale_date >= make_date(2015,11,1); "
                    "DELETE FROM shop_item_dates "
                    "WHERE sale_date >= make_date(2015,11,1);"
                )
                cur.execute(sql)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred in delete_test_prd function.")
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def export_features(self):
        """ Export features from PostgreSQL to CSV file. """
        try:
            # join sid_new_day with id_new_day on item,
            # with sd_new_day on shop,
            # with dates on date (after lags have been assigned),
            # with items on item, with shops on shop
            # then export the fully joined table to CSV and store in S3 bucket
            col_name_dict = {
                "sid_new_day": "sid",
                "id_new_day": "id",
                "sd_new_day": "sd",
                "dates": "d",
                "items": "i",
                "shops": "s",
            }

            sql_col_list = []
            with open(f"./{col_names_csv_path}", "r") as col_file:
                csv_reader = csv.reader(col_file, delimiter=",")
                next(csv_reader, None)  # skip header row
                for row in csv_reader:
                    if row[1] in [
                        "sid_new_day",
                        "id_new_day",
                        "sd_new_day",
                        "dates",
                        "items",
                        "shops",
                    ] and not (
                        row[1] != "sid_new_day"
                        and row[2] in ["shop_id", "item_id", "sale_date"]
                    ):
                        sql_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
            cols_to_select = ", ".join(sql_col_list)

            with self.conn.cursor() as cur:
                query = (
                    f"SELECT {cols_to_select} FROM sid_new_day sid "
                    "LEFT JOIN id_new_day id "
                    "ON sid.item_id = id.item_id "
                    "LEFT JOIN sd_new_day sd "
                    "ON sid.shop_id = id.shop_id "
                    "LEFT JOIN dates d "
                    "ON sid.sale_date = d.sale_date "
                    "LEFT JOIN items i "
                    "ON sid.item_id = i.item_id "
                    "LEFT JOIN shops s "
                    "ON sid.shop_id = s.shop_id"
                )
                sql = (
                    f"SELECT * from aws_s3.query_export_to_s3('{query}',"
                    "aws_commons.create_s3_uri('my-rds-exports', 'test_data_for_scoring.csv', 'us-west-2'),"
                    "options :='format csv, header');"
                )
                cur.execute(sql)
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred in export_features function.")
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def append_features(self):
        """ Append individual features to the appropriate existing tables
        (e.g., id_... features are appended to the item_dates table).
        """
        try:
            # insert into items_ver(item_id, item_group, name)
            # select item_id, item_group, name from items where item_id=2;
            # NEED TO GENERATE THE LIST OF COLUMNS IN EACH NEW TABLE (id_new_day, sd_new_day, sid_new_day)
            # AND THEN PLACE INTO THE QUERY BELOW IN BOTH PLACES
            id_col_list = []
            sd_col_list = []
            sid_new_col_list = []
            sid_main_col_list = []
            with open(f"./{col_names_csv_path}", "r") as col_file:
                csv_reader = csv.reader(col_file, delimiter=",")
                next(csv_reader, None)  # skip header row
                for row in csv_reader:
                    if row[1] == "id_new_day":
                        # id_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
                        id_col_list.append(row[2])
                    elif row[1] == "sd_new_day":
                        # sd_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
                        sd_col_list.append(row[2])
                    elif row[1] == "sid_new_day":
                        sid_new_col_list.append(row[2])
                        # sid_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
                    elif row[1] == "shop_item_dates":
                        sid_main_col_list.append(row[2])
            id_cols_to_select = ", ".join(id_col_list)
            sd_cols_to_select = ", ".join(sd_col_list)
            # some cols in sid_new_day do not belong in shop_item_dates and, possibly,
            # some shop_item_dates columns are not in sid_new_day, so need to create
            # a list with common elements:
            sid_col_list = list(set(sid_new_col_list) & set(sid_main_col_list))
            sid_cols_to_select = ", ".join(sid_col_list)

            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql = (
                    f"INSERT INTO item_dates ({id_cols_to_select}) "
                    f"SELECT {id_cols_to_select} FROM id_new_day; "
                    f"INSERT INTO shop_dates ({sd_cols_to_select}) "
                    f"SELECT {sd_cols_to_select} FROM sd_new_day; "
                    f"INSERT INTO shop_item_dates ({sid_cols_to_select}) "
                    f"SELECT {sid_cols_to_select} FROM sid_new_day; "
                    # SOME OF THE SID_ FEATURES NEED TO BE INSERTED INTO OTHER SID_ TABLES -
                    # for now, decided not to insert anything into those other sid_ tables
                )
                cur.execute(sql)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred in append_features function.")
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def import_preds_into_new_table(self, first_day):
        """ Import predictions from first model into daily_sid_predictions table
        in PostgreSQL database.

        Parameters:
        -----------
        first_day : bool
            Indicator for currently generating and storing predictions for the
            first day in the test period
        """
        try:
            if first_day:
                del self.conn.notices[
                    :
                ]  # clear the notices list before executing next query
                with self.conn.cursor() as cur:
                    sql = (
                        "DROP TABLE IF EXISTS daily_sid_predictions; "
                        "CREATE TABLE daily_sid_predictions (shop_id smallint NOT NULL, "
                        "item_id int NOT NULL, sale_date date NOT NULL, model1 int NOT NULL)"
                    )
                    cur.execute(sql)
                for line in self.conn.notices:
                    logging.debug(line.strip("\n"))

            # import CSV data to created table
            # predictions for different days are appended to existing table
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql = (
                    f"SELECT aws_s3.table_import_from_s3('daily_sid_predictions', '', '(format csv, header)', "
                    f"aws_commons.create_s3_uri('sales-demand-predictions', 'preds_model1.csv', 'us-west-2'))"
                    # if going to have separate CSVs for each day's predictions, need to update the preds_model1.csv parameter above to include date
                    # same for csv path below (in the import_preds_into_existing_table function)
                )
                cur.execute(sql)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception(
                "Exception occurred in import_preds_into_new_table function."
            )
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def import_preds_into_existing_table(self, first_day, model_col):
        """ Import predictions from any model after the first model into
        daily_sid_predictions table in PostgreSQL database.

        Parameters:
        -----------
        first_day : bool
            Indicator for currently generating and storing predictions for the
            first day in the test period
        model_col : str
            Name of column in daily predictions table containing predictions
            for current model
        """
        try:
            if first_day:
                del self.conn.notices[
                    :
                ]  # clear the notices list before executing next query
                with self.conn.cursor() as cur:
                    sql_str = (
                        "ALTER TABLE daily_sid_predictions "
                        "DROP COLUMN IF EXISTS {0}, "
                        "ADD COLUMN {0} smallint NOT NULL;"
                    )
                    sql = SQL(sql_str).format(Identifier(model_col))
                    cur.execute(sql)
                for line in self.conn.notices:
                    logging.debug(line.strip("\n"))

            # predictions are joined with existing rows on shop-item-date
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql_str = (
                    "CREATE TEMP TABLE new_model_preds (shop_id smallint NOT NULL, "
                    "item_id int NOT NULL, sale_date date NOT NULL, {0} smallint NOT NULL); "
                    "SELECT aws_s3.table_import_from_s3('new_model_preds', '', '(format csv, header)', "
                    f"aws_commons.create_s3_uri('sales-demand-predictions', 'preds_{model_col}.csv', 'us-west-2')); "
                    "UPDATE daily_sid_predictions dsp "
                    "SET {0} = {1} "
                    "FROM new_model_preds nmp "
                    "WHERE dsp.shop_id = nmp.shop_id AND dsp.item_id = nmp.item_id AND "
                    "dsp.sale_date = nmp.sale_date;"
                )
                sql = SQL(sql_str).format(
                    Identifier(model_col), Identifier("nmp", model_col),
                )
                cur.execute(sql)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception(
                "Exception occurred in import_preds_into_existing_table function."
            )
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)

    @Timer(logger=logging.info)
    def check_size_of_preds_table(self, model_cnt):
        """ Query size of daily_sid_predictions to make sure it has the right
        number of columns and rows after being populated with another day of data.

        Parameters:
        -----------
        model_cnt : int
            Model number (number of the model currently being worked on)
        """
        row_and_col_cts_can_be_used = True
        del self.conn.notices[:]  # clear the notices list before executing next query
        with self.conn.cursor() as cur:
            sql = (
                "WITH cols AS ("
                "SELECT column_name "
                "FROM information_schema.columns "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                "AND table_name = 'daily_sid_predictions') "
                "SELECT count(column_name) AS n_cols "
                "FROM cols"
            )
            try:
                n_cols = cur.execute(sql).fetchone()[0]
            except TypeError as e:
                row_and_col_cts_can_be_used = False
                logging.exception(
                    "Exception occurred when querying number of columns "
                    "in daily_sid_predictions table:"
                )

        with self.conn.cursor() as cur:
            sql = "SELECT count(*) FROM daily_sid_predictions AS n_rows"
            try:
                n_rows = cur.execute(sql).fetchone()[0]
            except TypeError as e:
                row_and_col_cts_can_be_used = False
                logging.exception(
                    "Exception occurred when querying number of rows "
                    "in daily_sid_predictions table:"
                )
        for line in self.conn.notices:
            logging.debug(line.strip("\n"))

        if row_and_col_cts_can_be_used:
            # there should be 214,200 rows per day
            # there should be 3 (shop, item, date) + 2 * model columns
            if (n_rows != 214_200 * i) or (n_cols != (3 + 1 * model_cnt)):
                logging.error(
                    f"Expected {214_200 * i} rows and {3 + 1 * model_cnt} "
                    "columns at this point in the predictions table; "
                    f"instead, the table has {n_rows} rows and {n_cols} columns."
                )
                sys.exit(1)

    @Timer(logger=logging.info)
    def agg_preds(self, model_col, params=None):
        """ Aggregate shop-item-date level predictions to appropriate level and
        update quantity sold values in dates (d_day_total_qty_sold), item_dates
        (id_item_qty_sold_day), shop_dates (sd_shop_qty_sold_day), shop_item_dates
        (sid_shop_item_qty_sold_day) and sales_cleaned (only non-zero predicted
        shop-item quantities) tables.

        Parameters:
        -----------
        model_col : str
            Name of column in daily predictions table containing predictions
            for current model
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method
        """
        try:
            del self.conn.notices[
                :
            ]  # clear the notices list before executing next query
            with self.conn.cursor() as cur:
                sql_str = (
                    "WITH day_total AS ("
                    "SELECT sale_date, sum({0}) AS d_day_total_qty_sold "
                    "FROM daily_sid_predictions "
                    "WHERE sale_date = %(curr_date)s "
                    "GROUP BY sale_date) "
                    "UPDATE dates d "
                    "SET d_day_total_qty_sold = dt.d_day_total_qty_sold "
                    "FROM day_total dt "
                    "WHERE d.sale_date = dt.sale_date; "
                )
                sql = SQL(sql_str).format(Identifier(model_col))
                cur.execute(sql, params)
            with self.conn.cursor() as cur:
                sql_str = (
                    "WITH item_total AS ("
                    "SELECT item_id, sum({0}) AS id_item_qty_sold_day "
                    "FROM daily_sid_predictions "
                    "WHERE sale_date = %(curr_date)s "
                    "GROUP BY item_id) "
                    "UPDATE item_dates id "
                    "SET id_item_qty_sold_day = it.id_item_qty_sold_day "
                    "FROM item_total it "
                    "WHERE id.sale_date = %(curr_date)s AND id.item_id = it.item_id; "
                )
                sql = SQL(sql_str).format(Identifier(model_col))
                cur.execute(sql, params)
            with self.conn.cursor() as cur:
                sql_str = (
                    "WITH shop_total AS ("
                    "SELECT shop_id, sum({0}) AS sd_shop_qty_sold_day "
                    "FROM daily_sid_predictions "
                    "WHERE sale_date = %(curr_date)s "
                    "GROUP BY shop_id) "
                    "UPDATE shop_dates sd "
                    "SET sd_shop_qty_sold_day = st.sd_shop_qty_sold_day "
                    "FROM shop_total st "
                    "WHERE sd.sale_date = %(curr_date)s AND sd.shop_id = st.shop_id; "
                )
                sql = SQL(sql_str).format(Identifier(model_col))
                cur.execute(sql, params)
            with self.conn.cursor() as cur:
                sql_str = (
                    "UPDATE shop_item_dates sid "
                    "SET sid_shop_item_qty_sold_day = {0} "
                    "FROM daily_sid_predictions dsp "
                    "WHERE sid.sale_date = %(curr_date)s AND sid.shop_id = dsp.shop_id AND "
                    "sid.item_id = dsp.item_id; "
                )
                sql = SQL(sql_str).format(Identifier("dsp", model_col))
                cur.execute(sql, params)
            with self.conn.cursor() as cur:
                # insert non-zero sid predicted quantity into sales_cleaned
                sql_str = (
                    "INSERT INTO sales_cleaned (shop_id, item_id, sale_date, item_cnt_day) "
                    "SELECT shop_id, item_id, sale_date, {0} AS item_cnt_day "
                    "FROM daily_sid_predictions "
                    "WHERE {0} <> 0 AND sale_date = %(curr_date)s"
                )
                sql = SQL(sql_str).format(Identifier(model_col))
                cur.execute(sql, params)
            for line in self.conn.notices:
                logging.debug(line.strip("\n"))
        except (Exception, psycopg2.DatabaseError) as error:
            logging.exception("Exception occurred in agg_preds function.")
            if self.conn is not None:
                self.conn.close()
                logging.info("Database connection closed.")
            # copy log file to S3 bucket if running script on a EC2 instance
            if self.is_aws:
                try:
                    response = self.s3_client.upload_file(
                        f"./logs/{self.log_fname}", "my-ec2-logs", self.log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    # need the following command-line arguments:
    # model counter
    # first date (date to start predictions for)
    # number of days to predict (check that first date + number of days - 1 does not surpass 11/30)
    parser.add_argument(
        "modelnum",
        metavar="<modelnum>",
        help="model counter (first model should be numbered with 1)",
        type=int,
    )
    parser.add_argument(
        "firstday",
        metavar="<firstday>",
        help="first (earliest) day for which to get predictions, format: DD",
        type=valid_day,
    )
    parser.add_argument(
        "numdays",
        metavar="<numdays>",
        help="number of days for which to run predictions, including the first day",
        type=int,
        choices=range(1, 30),
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after querying (if included) or not (if not included)",
    )

    args = parser.parse_args()
    # check that first day + number of days - 1 does not surpass 11/30
    if args.firstday + datetime.timedelta(days=(args.numdays - 1)) > datetime.date(
        2015, 11, 30
    ):
        parser.error(
            "Enter combination of first day and number of days that "
            "does not surpass Nov 30th."
        )

    fmt = "%(threadName)-9s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("logs")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    log_fname = (
        f"prep_test_data_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
    )
    log_path = log_dir.joinpath(log_fname)

    csv_dir = Path.cwd().joinpath("csv")
    path = Path(csv_dir)
    path.mkdir(exist_ok=True)
    cols_csv_fname = "columns_output_new_day_tables.csv"

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

    # start RDS instance
    start_instance()

    logging.info(f"Starting to run predictions for model {args.modelnum}...")

    # create database connection
    db = single_thread_db_class(is_aws, log_fname)

    for curr_date in [
        args.firstday + datetime.timedelta(days=x) for x in range(args.numdays)
    ]:

        curr_date_str = datetime.datetime.strftime(curr_date, format="%Y-%m-%d")
        logging.info(f"Starting to run predictions for {curr_date_str}...")

        first_day = curr_date == datetime.date(2015, 11, 1)
        first_model = args.modelnum == 1
        params = {"curr_date": curr_date}

        if not first_model and first_day:
            db.delete_from_sales_cleaned()
            db.delete_test_prd()

        # create separate tables of features for new day:
        # id_new_day, sd_new_day, sid_new_day
        # also, current date's rows are updated in dates table
        # that leaves shops and items tables, which remain constant/unchanged

        # QUERIES TO CREATE _NEW_DAY TABLES AND THE FEATURES IN THEM MOVED TO QUERIES.PY
        three_strings = [
            ("DROP TABLE" + x).strip() for x in all_queries_str.split("DROP TABLE")
        ][1:]
        three_lists = [
            [
                ("ALTER" + x).strip() if not x.startswith("DROP") else x.strip()
                for x in s.split("ALTER")
            ]
            for s in three_strings
        ]

        threads = [
            multi_thread_db_class(
                args=(three_lists[0].insert(0, lag_query)), kwargs=params
            ),
            multi_thread_db_class(args=(three_lists[1]), kwargs=params),
            multi_thread_db_class(args=(three_lists[2]), kwargs=params),
        ]
        for t in threads:
            t.start()

        # main thread looks at the status of all threads
        n_threads_finished = 0
        try:
            # while True:
            while n_threads_finished < 3:
                for t in threads:
                    if t.exception:
                        # there was an error in a thread - raise it in main thread too
                        # this will stop the loop
                        raise t.exception
                time.sleep(0.5)

        except Exception as e:
            # handle exceptions any way you like, or don't
            # This includes exceptions in main thread as well as those in other threads
            # (because of "raise t.exception" above)

            # so, as soon as an exception is raised in one of the threads, the try
            # clause is ended, and except and finally clauses are run, which means
            # any thread without an exception is stopped
            logging.exception("Exception encountered in main or another thread.")

            # copy log file to S3 bucket if running script on a EC2 instance
            if is_aws:
                try:
                    response = s3_client.upload_file(
                        f"./logs/{log_fname}", "my-ec2-logs", log_fname
                    )
                    logging.info("Log file was successfully copied to S3.")
                except ClientError as e:
                    logging.exception("Log file was not copied to S3.")

            sys.exit(1)  # finally clause will execute regardless of exception, so
            # threads will be stopped before exiting Python.

        finally:
            for t in threads:
                # threads will know how to clean up when stopped
                t.stop()

        col_names_csv_path = csv_dir.joinpath(cols_csv_fname)
        # RUN SUMMARY QUERY TO GENERATE COLUMN LIST FOR THE TABLES TO BE JOINED
        # AND THEN USE THAT LIST TO GENERATE LIST OF COLUMNS FOR SELECT CLAUSE BELOW
        # (THAT SUMMARY QUERY JUST NEEDS TO BE RUN ONCE IF OUTPUT CSV DOES NOT EXIST)
        if not Path(col_names_csv_path).is_file():
            db.query_col_names(col_names_csv_path, cols_csv_fname)

        db.export_features()

        db.append_features()

        # HERE IS WHERE PREDICTIONS ARE GENERATED

        # get shop-item-date predicted values back into PostgreSQL and update quantity sold values
        # in dates (d_day_total_qty_sold), item_dates (id_item_qty_sold_day),
        # shop_dates (sd_shop_qty_sold_day), shop_item_dates (sid_shop_item_qty_sold_day)
        # and sales_cleaned (only non-zero predicted shop-item quantities) tables
        # LOAD PREDICTIONS INTO A NEW TABLE IN SHOP-ITEM-DATE FORM, WITH THE PREDICTION COLUMN
        # INDICATING THE MODEL THAT THOSE PREDICTIONS CAME FROM, AND THEN AGGREGATE THE
        # VALUES AS PER ABOVE AND UPDATE THE OTHER TABLES
        # LOAD OTHER MODELS' PREDICTIONS AS SEPARATE COLUMNS IN THAT PREDICTIONS TABLE
        # > make a flag: if first model -> predictions for different days are appended to existing table;
        # > otherwise -> predictions are joined with existing rows on shop-item-date
        # FOR UPDATING THE OTHER TABLES:
        # if first model -> insert aggregated predictions as new rows
        #   except for dates -> update existing rows
        #   ACTUALLY, existing rows need to be updated in all tables except for sales_cleaned
        # otherwise -> update existing rows
        #   except for sales_cleaned -> need to delete rows from previous model (for all days: 11/1 thru 11/30) and insert rows anew
        #       so this deletion only needs to happen once at the beginning of running predictions for that model
        if first_model:
            db.import_preds_into_new_table(first_day)
        else:
            db.import_preds_into_existing_table(first_day, f"model{args.modelnum}")

        db.check_size_of_preds_table(args.modelnum)

        db.agg_preds(f"model{args.modelnum}", params)

    # close database connection
    db.close_db_conn()

    if args.stop == True:
        stop_instance()

    # copy log file to S3 bucket if running script on a EC2 instance
    if is_aws:
        s3_client = boto3.client("s3")
        try:
            response = s3_client.upload_file(
                f"./logs/{log_fname}", "my-ec2-logs", log_fname
            )
        except ClientError as e:
            logging.exception("Log file was not copied to S3.")


if __name__ == "__main__":
    main()


# SOME OF THE SID_ COLUMNS BELOW ARE NOT IN THE SHOP_ITEM_DATES TABLE, BUT IN ANOTHER SID_ TABLE
#  'sid_cat_sold_at_shop_before_day_flag': 'int16', *
#       - for each new shop-item-date, need to check if other items in same category were sold at the shop before that day
#       - create shop-category-date table with total quantity sold through that day
#           - sum(quantity sold) OVER (PARTITION BY shop_id, sid_item_category_id ORDER BY sale_date)
#       - increment each date value by one day (e.g., Jan 30 > Jan 31)
#       - create binary flag out of total_quantity_sold column (case __ when <> 0 then 1 else 0)
#       - update existing sid_cat_sold_at_shop_before_day_flag column by joining with the new table above on shop_id, sid_item_category_id, sale_date
#  'sid_coef_var_price': 'float32', FILL NEW SIDS WITH 0's
#  'sid_days_since_max_qty_sold': 'int32', * ZEROS BEFORE FIRST SALE DATE MAY NOT BE OKAY, NEED TO CHECK !!!
#  'sid_expand_cv2_of_qty': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_item_category_id': 'int16', x update by joining with items table on item_id column
#  'sid_qty_mean_abs_dev': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_qty_median_abs_dev': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_cat_qty_sold_day': 'int16', x
#  'sid_shop_item_cnt_sale_dts_before_day': 'int32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_cnt_sale_dts_last_30d': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_cnt_sale_dts_last_7d': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_avg_gap_bw_sales': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_diff_bw_last_and_prev_qty': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_max_gap_bw_sales': 'int32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_median_gap_bw_sales': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_min_gap_bw_sales': 'int32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_mode_gap_bw_sales': 'int32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_date_std_gap_bw_sales': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_days_since_first_sale': 'int32', * FILL NEW SIDS WITH 0'S
#  'sid_shop_item_days_since_prev_sale': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expand_qty_max': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expand_qty_mean': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expand_qty_median': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expand_qty_min': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expand_qty_mode': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_expanding_adi': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_first_month': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_first_week': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_last_qty_sold': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_qty_sold_1d_ago': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_qty_sold_2d_ago': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_qty_sold_3d_ago': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_qty_sold_7d_ago': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_qty_sold_day': 'int16', FILL NEW SIDS WITH 0's
#  'sid_shop_item_rolling_7d_avg_qty': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_rolling_7d_max_qty': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_rolling_7d_median_qty': 'float32', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_rolling_7d_min_qty': 'int16', * FILL NEW SIDS WITH 0's
#  'sid_shop_item_rolling_7d_mode_qty': 'int16' * FILL NEW SIDS WITH 0's


#  'sd_num_unique_item_cats_prior_to_day': 'int16', * EXCLUDES ZEROS
#  'sd_num_unique_items_prior_to_day': 'int32', * EXCLUDES ZEROS
#  'sd_shop_cnt_sale_dts_before_day': 'int32', * EXCLUDES NON-POSITIVE VALUES
#  'sd_shop_cnt_sale_dts_last_30d': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'sd_shop_cnt_sale_dts_last_7d': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'sd_shop_date_avg_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'sd_shop_date_diff_bw_last_and_prev_qty': 'int16', * EXCLUDES ZEROS
#  'sd_shop_date_max_gap_bw_sales': 'int32', * EXCLUDES ZEROS
#  'sd_shop_date_median_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'sd_shop_date_min_gap_bw_sales': 'int16', * EXCLUDES ZEROS
#  'sd_shop_date_mode_gap_bw_sales': 'int16', * EXCLUDES ZEROS
#  'sd_shop_date_std_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'sd_shop_days_since_first_sale': 'int32', * ZEROS ARE OKAY
#  'sd_shop_days_since_prev_sale': 'int32', * EXCLUDES NON-POSITIVE VALUES
#  'sd_shop_expand_qty_max': 'int32', * ZEROS ARE OKAY
#  'sd_shop_expand_qty_mean': 'float32', * ZEROS ARE OKAY
#  'sd_shop_expand_qty_median': 'float32', * ZEROS ARE OKAY
#  'sd_shop_expand_qty_min': 'int16', * ZEROS ARE OKAY
#  'sd_shop_expand_qty_mode': 'int32', * ZEROS ARE OKAY
#  'sd_shop_first_month': 'int16', * ZEROS ARE OKAY
#  'sd_shop_first_week': 'int16', * ZEROS ARE OKAY
#  'sd_shop_last_qty_sold': 'int16', * EXCLUDES ZEROS
#  'sd_shop_qty_sold_1d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'sd_shop_qty_sold_2d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'sd_shop_qty_sold_3d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'sd_shop_qty_sold_7d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'sd_shop_qty_sold_day': 'int16',
#  'sd_shop_rolling_7d_avg_qty': 'float32', * ZEROS ARE OKAY
#  'sd_shop_rolling_7d_max_qty': 'int32', * ZEROS ARE OKAY
#  'sd_shop_rolling_7d_median_qty': 'float32', * ZEROS ARE OKAY
#  'sd_shop_rolling_7d_min_qty': 'int16', * ZEROS ARE OKAY
#  'sd_shop_rolling_7d_mode_qty': 'int32', * ZEROS ARE OKAY


#  'id_item_days_since_first_sale': 'int32',
query = (
    "SELECT item_id, make_date(___) - first_sale_dt AS id_item_days_since_first_sale "
    "FROM ("
    "SELECT item_id, min(sale_date) AS first_sale_dt "
    "FROM item_dates "
    "GROUP BY item_id) t1"
)
#  'id_item_days_since_prev_sale': 'int16',
query = (
    "SELECT item_id, make_date(___) - last_sale_dt AS id_item_days_since_prev_sale "
    "FROM ("
    "SELECT item_id, max(sale_date) AS last_sale_dt "
    "FROM item_dates "
    "WHERE id_item_qty_sold_day > 0 "
    "GROUP BY item_id) t1"
)
# ALSO NEED TO CHECK MAX/MIN(SALE_DATE) NEEDS TO BE CAST TO DATE (::date) AND THE SUBTRACTION OPERATION TO INTEGER


# per https://dba.stackexchange.com/questions/2973/how-to-insert-values-into-a-table-from-a-select-query-in-postgresql
# insert into items_ver(item_id, item_group, name)
# select * from items where item_id=2;

# https://stackoverflow.com/questions/6256610/updating-table-rows-in-postgres-using-subquery
# UPDATE dummy
# SET customer=subquery.customer,
#     address=subquery.address,
#     partn=subquery.partn
# FROM (SELECT address_id, customer, address, partn
#       FROM  /* big hairy SQL */ ...) AS subquery
# WHERE dummy.address_id=subquery.address_id;
# OTHER ANSWERS THERE TOO


#  'id_cat_qty_sold_last_7d': 'int16', * ZEROS ARE OKAY
#  'id_cat_qty_sold_per_item_last_7d': 'float32', * ZEROS ARE OKAY
#  'id_cat_unique_items_sold_last_7d': 'int32', * EXCLUDES NON-POSITIVE VALUES
#  'id_days_since_max_qty_sold': 'int32', * ZEROS ARE OKAY
#  'id_expand_cv2_of_qty': 'float32', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_category_id': 'int16',
#  'id_item_cnt_sale_dts_before_day': 'int32', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_cnt_sale_dts_last_30d': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_cnt_sale_dts_last_7d': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_date_avg_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'id_item_date_diff_bw_last_and_prev_qty': 'int16', * EXCLUDES ZEROS
#  'id_item_date_max_gap_bw_sales': 'int16', * EXCLUDES ZEROS
#  'id_item_date_median_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'id_item_date_min_gap_bw_sales': 'int16', * EXCLUDES ZEROS
#  'id_item_date_mode_gap_bw_sales': 'int16', * EXCLUDES ZEROS
#  'id_item_date_std_gap_bw_sales': 'float32', * EXCLUDES ZEROS
#  'id_item_days_since_first_sale': 'int32', * ZEROS ARE OKAY
#  'id_item_days_since_prev_sale': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_expand_qty_max': 'int16', * ZEROS ARE OKAY
#  'id_item_expand_qty_mean': 'float32', * ZEROS ARE OKAY
#  'id_item_expand_qty_median': 'float32', * ZEROS ARE OKAY
#  'id_item_expand_qty_min': 'int16', * ZEROS ARE OKAY
#  'id_item_expand_qty_mode': 'int16', * ZEROS ARE OKAY
#  'id_item_expanding_adi': 'float32', * ZEROS ARE OKAY
#  'id_item_first_month': 'int16', * ZEROS ARE OKAY
#  'id_item_first_week': 'int16', * ZEROS ARE OKAY
#  'id_item_had_spike_before_day': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_last_qty_sold': 'int16', * EXCLUDES ZEROS
#  'id_item_n_spikes_before_day': 'int16', * EXCLUDES NON-POSITIVE VALUES
#  'id_item_qty_sold_1d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'id_item_qty_sold_2d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'id_item_qty_sold_3d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'id_item_qty_sold_7d_ago': 'int16', * ZEROS JUST DON'T AFFECT THE TOTAL
#  'id_item_qty_sold_day': 'int16',
#  'id_item_rolling_7d_avg_qty': 'float32', * ZEROS ARE OKAY
#  'id_item_rolling_7d_max_qty': 'int16', * ZEROS ARE OKAY
#  'id_item_rolling_7d_median_qty': 'float32', * ZEROS ARE OKAY
#  'id_item_rolling_7d_min_qty': 'int16', * ZEROS ARE OKAY
#  'id_item_rolling_7d_mode_qty': 'int16', * ZEROS ARE OKAY
#  'id_num_unique_shops_prior_to_day': 'float32', * EXCLUDES ZEROS


# {'d_apartment_fund_sqm': 'float32',
#  'd_average_life_exp': 'float32',
#  'd_average_provision_of_build_contract': 'float32',
#  'd_average_provision_of_build_contract_moscow': 'float32',
#  'd_balance_trade': 'float32',
#  'd_balance_trade_growth': 'float32',
#  'd_bandwidth_sports': 'int64',
#  'd_brent': 'float32',
#  'd_brent_1day_lag': 'float32',
#  'd_childbirth': 'float32',
#  'd_cpi': 'float32',
#  'd_date_block_num': 'int16',
#  'd_date_counter': 'int32',
#  'd_day_of_week': 'int16',
#  'd_day_total_qty_sold': 'int32',
#  'd_day_total_qty_sold_1day_lag': 'int32',
#  'd_day_total_qty_sold_6day_lag': 'int32',
#  'd_day_total_qty_sold_7day_lag': 'int32',
#  'd_days_after_holiday': 'int16',
#  'd_days_in_mon': 'int16',
#  'd_days_to_holiday': 'int16',
#  'd_deposits_growth': 'float32',
#  'd_deposits_rate': 'float32',
#  'd_deposits_value': 'int64',
#  'd_dow_cos': 'float32',
#  'd_dow_sin': 'float32',
#  'd_employment': 'float32',
#  'd_eurrub': 'float32',
#  'd_fixed_basket': 'float32',
#  'd_gdp_annual': 'float32',
#  'd_gdp_annual_growth': 'float32',
#  'd_gdp_deflator': 'float32',
#  'd_gdp_quart': 'float32',
#  'd_gdp_quart_growth': 'float32',
#  'd_holiday': 'int16',
#  'd_income_per_cap': 'float32',
#  'd_invest_fixed_assets': 'float32',
#  'd_invest_fixed_capital_per_cap': 'float32',
#  'd_is_weekend': 'int16',
#  'd_labor_force': 'float32',
#  'd_load_of_teachers_school_per_teacher': 'float32',
#  'd_load_on_doctors': 'float32',
#  'd_major_event': 'int16',
#  'd_micex': 'float32',
#  'd_micex_cbi_tr': 'float32',
#  'd_micex_rgbi_tr': 'float32',
#  'd_modern_education_share': 'object',
#  'd_month': 'int16',
#  'd_month_cos': 'float32',
#  'd_month_sin': 'float32',
#  'd_mortality': 'float32',
#  'd_mortgage_growth': 'float32',
#  'd_mortgage_rate': 'float32',
#  'd_mortgage_value': 'int64',
#  'd_net_capital_export': 'float32',
#  'd_oil_urals': 'float32',
#  'd_old_education_build_share': 'object',
#  'd_pop_natural_increase': 'float32',
#  'd_ppi': 'float32',
#  'd_provision_nurse': 'float32',
#  'd_ps4_game_release_dt': 'int16',
#  'd_ps4_game_release_dt_plus_2': 'int16',
#  'd_ps4_games_released_cnt': 'int16',
#  'd_qoy_cos': 'float32',
#  'd_qoy_sin': 'float32',
#  'd_quarter_counter': 'int16',
#  'd_quarter_of_year': 'int16',
#  'd_rent_price_1room_bus': 'float32',
#  'd_rent_price_1room_eco': 'float32',
#  'd_rent_price_2room_bus': 'float32',
#  'd_rent_price_2room_eco': 'float32',
#  'd_rent_price_3room_bus': 'float32',
#  'd_rent_price_3room_eco': 'float32',
#  'd_rent_price_4_room_bus': 'float32',
#  'd_retail_trade_turnover': 'float32',
#  'd_retail_trade_turnover_growth': 'float32',
#  'd_retail_trade_turnover_per_cap': 'float32',
#  'd_rts': 'float32',
#  'd_salary': 'float32',
#  'd_salary_growth': 'float32',
#  'd_seats_theather_rfmin_per_100000_cap': 'float32',
#  'd_students_state_oneshift': 'float32',
#  'd_turnover_catering_per_cap': 'int32',
#  'd_unemployment': 'float32',
#  'd_usdrub': 'float32',
#  'd_week_of_year': 'int64',
#  'd_woy_cos': 'float32',
#  'd_woy_sin': 'float32',
#  'd_year': 'category',
#  'i_digital_item': 'int16',
#  'i_item_cat_grouped_by_game_console': 'object',
#  'i_item_category_broad': 'object',
#  'i_item_category_id': 'int16',
#  'i_item_category_name': 'object',
#  'i_item_mon_of_first_sale': 'category',
#  'i_item_name': 'object',

#  'item_id': 'int32',
#  's_city': 'object',
#  's_geo_lat': 'float32',
#  's_geo_lon': 'float32',
#  's_n_other_stores_in_city': 'int16',
#  's_online_store': 'int16',
#  's_population': 'int64',
#  's_shop_name': 'object',
#  's_time_zone': 'int16',
#  'sale_date': 'datetime64[ns]',

#  'shop_id': 'int16',
