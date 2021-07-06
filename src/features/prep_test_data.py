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
- convert make_date() function in queries to date parameter: datetime.date(2015,11,1)
    to be used a parameter to be passed to the execute() method
        - make_date(___) becomes %(curr_date)s - DONE
        - params becomes {'curr_date': datetime.date(2015,11,1)}
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
                 ['ALTER a b c;', 'ALTER d e f;']
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

- IF GOING TO RUN ON EC2, NEED TO UPLOAD LOG, CSVs TO S3 (columns_output_new_day_tables.csv)

- make a diagram of how/when features and predictions are added to/deleted from existing tables
for first model and later models

- assign flag for first model or not first model
- loop over 30 days (Nov 1 thru Nov 30)
- create separate tables of features for the current day in the loop
- export the data out of Postgres as one joined table
'''

import argparse
import csv
import datetime
import logging
import os
from pathlib import Path
import platform
import sys

import boto3
from botocore.exceptions import ClientError
from ec2_metadata import ec2_metadata
import psycopg2

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils.config import config
from utils.rds_instance_mgmt import start_instance, stop_instance
from utils.timer import Timer


@Timer(logger=logging.info)
def query_col_names(csv_path):
    """ Connect to the PostgreSQL database server and query info on columns in
    existing tables.

    Parameters:
    -----------
    csv_path : str or pathlib.Path() object
        Filepath (directory and filename) where to save query results

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

        # Get information on columns in all tables from the information_schema.columns catalog
        # https://www.postgresql.org/docs/current/infoschema-columns.html
        query = """
            SELECT
               table_schema,
               table_name,
               column_name,
               data_type,
               is_nullable
            FROM
               information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        # csv_fname = "columns_output_new_day_tables.csv"
        # csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

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
            logging.debug(line.strip("\n"))
        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


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


def main():
    parser = argparse.ArgumentParser()
    # need the following command-line arguments:
    # model counter
    # first date (date to start predictions for)
    # number of days to predict (check that first date + number of days - 1 does not surpass 11/30)
    parser.add_argument(
        "modelnum",
        metavar="<modelnum>"
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

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("logs")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    log_fname = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
    log_path = log_dir.joinpath(log_fname)

    csv_dir = Path.cwd().joinpath("csv")
    path = Path(csv_dir)
    path.mkdir(exist_ok=True)

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

    start_instance()

    logging.info(f"Starting to run predictions for model {args.model_num}...")

    for i, curr_date in enumerate([
        args.firstday + datetime.timedelta(days=x) for x in range(args.numdays)
    ], 1):

        logging.info(
            f"Starting to run predictions for {datetime.datetime.strftime(curr_date, format='%Y-%m-%d')}..."
        )

        first_day = curr_date == datetime.date(2015, 11, 1)
        first_model = args.modelnum == 1
        params = {"curr_date": curr_date}

        if not first_model and first_day:
            # delete rows from previous model (deletion to only happen once) in the sales_cleaned table
            query = (
                "DELETE FROM sales_cleaned " "WHERE sale_date >= make_date(2015,11,1) "
            )
            # delete rows for the test period from feature-containing tables
            # THIS QUERY BELOW PRECEDES INSERTION OF FEATURES INTO THOSE TABLES
            query = (
                "DELETE FROM item_dates "
                "WHERE sale_date >= make_date(2015,11,1); "
                "DELETE FROM shop_dates "
                "WHERE sale_date >= make_date(2015,11,1); "
                "DELETE FROM shop_item_dates "
                "WHERE sale_date >= make_date(2015,11,1); "
            )

        # create separate tables of features for new day:
        # id_new_day, sd_new_day, sid_new_day
        # also, current date's rows are updated in dates table
        # that leaves shops and items tables, which remain constant/unchanged

        # QUERIES TO CREATE _NEW_DAY TABLES AND THE FEATURES IN THEM MOVED TO QUERIES.PY

        col_names_csv_path = csv_dir.joinpath("columns_output_new_day_tables.csv")
        # RUN SUMMARY QUERY TO GENERATE COLUMN LIST FOR THE TABLES TO BE JOINED
        # AND THEN USE THAT LIST TO GENERATE LIST OF COLUMNS FOR SELECT CLAUSE BELOW
        # (THAT SUMMARY QUERY JUST NEEDS TO BE RUN ONCE IF OUTPUT CSV DOES NOT EXIST)
        if not Path(col_names_csv_path).is_file():
            query_col_names(col_names_csv_path)

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
        run_query(sql)

        # append individual features to the right existing tables (e.g., id_… features get appended to the item_dates table)
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
        query = (
            f"INSERT INTO item_dates ({id_cols_to_select}) "
            f"SELECT {id_cols_to_select} FROM id_new_day; "
            f"INSERT INTO shop_dates ({sd_cols_to_select}) "
            f"SELECT {sd_cols_to_select} FROM sd_new_day; "
            f"INSERT INTO shop_item_dates ({sid_cols_to_select}) "
            f"SELECT {sid_cols_to_select} FROM sid_new_day; "
            # SOME OF THE SID_ FEATURES NEED TO BE INSERTED INTO OTHER SID_ TABLES -
            # for now, decided not to insert anything into those other sid_ tables
        )

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

            if first_day:

                sql = (
                    f"DROP TABLE IF EXISTS daily_sid_predictions; "
                    f"CREATE TABLE daily_sid_predictions (shop_id smallint NOT NULL, "
                    f"item_id int NOT NULL, sale_date date NOT NULL, model1 int NOT NULL)"
                )
                run_query(sql)

            # import CSV data to created table
            # predictions for different days are appended to existing table
            sql = (
                f"SELECT aws_s3.table_import_from_s3('daily_sid_predictions', '', '(format csv, header)', "
                f"aws_commons.create_s3_uri('sales-demand-predictions', 'preds_model1.csv', 'us-west-2'))"
                # if going to have separate CSVs for each day's predictions, need to update the preds_model1.csv parameter above to include date
                # same for csv path below
            )
            run_query(sql)

            # COMBINE THE TWO QUERIES ABOVE (TO RUN ON SAME CONNECTION)

        else:

            # predictions are joined with existing rows on shop-item-date
            sql = (
                "CREATE TEMP TABLE new_model_preds (shop_id smallint NOT NULL, "
                f"item_id int NOT NULL, sale_date date NOT NULL, model{args.modelnum} smallint NOT NULL); "
                "SELECT aws_s3.table_import_from_s3('new_model_preds', '', '(format csv, header)', "
                f"aws_commons.create_s3_uri('sales-demand-predictions', 'preds_model{args.modelnum}.csv', 'us-west-2')); "
                "ALTER TABLE daily_sid_predictions "
                f"ADD COLUMN model{args.modelnum} smallint NOT NULL; "
                "UPDATE daily_sid_predictions dsp "
                f"SET model{args.modelnum} = nmp.model{args.modelnum} "
                "FROM new_model_preds nmp "
                "WHERE dsp.shop_id = nmp.shop_id AND dsp.item_id = nmp.item_id AND "
                "dsp.sale_date = nmp.sale_date"
            )

        # query size of daily_sid_predictions to make sure it has the right
        # number of columns and rows after being populated with another day of data
        query = (
            "WITH cols AS ("
            "SELECT column_name "
            "FROM information_schema.columns "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "AND table_name = 'daily_sid_predictions'), "
            "SELECT count(column_name) AS n_cols "
            "FROM cols"
        )
        n_cols = conn.execute(query).fetchone()[0]
        query = (
            "SELECT count(*) FROM daily_sid_predictions AS n_rows "
        )
        n_rows = conn.execute(query).fetchone()[0]
        # there should be 214,200 rows per day
        # there should be 3 (shop, item, date) + 2 * model columns
        if (n_rows != 214_200 * i) | (n_cols != (3 + 1 * args.modelnum)):
            logging.error(
                f"Expected {214_200 * i} rows and {3 + 1 * args.modelnum} "
                "columns at this point in the predictions table; "
                f"instead, the table has {n_rows} rows and {n_cols} columns."
            )
            sys.exit(1)

        query = (
            "WITH day_total AS ("
            f"SELECT sale_date, sum(model{args.modelnum}) AS d_day_total_qty_sold "
            "FROM daily_sid_predictions "
            "WHERE sale_date = %(curr_date)s "
            "GROUP BY sale_date) "
            "UPDATE dates d "
            "SET d_day_total_qty_sold = dt.d_day_total_qty_sold "
            "FROM day_total dt "
            "WHERE d.sale_date = dt.sale_date; "
        )
        query = (
            "WITH item_total AS ("
            f"SELECT item_id, sum(model{args.modelnum}) AS id_item_qty_sold_day "
            "FROM daily_sid_predictions "
            "WHERE sale_date = %(curr_date)s "
            "GROUP BY item_id) "
            "UPDATE item_dates id "
            "SET id_item_qty_sold_day = it.id_item_qty_sold_day "
            "FROM item_total it "
            "WHERE id.sale_date = %(curr_date)s AND id.item_id = it.item_id; "
        )
        query = (
            "WITH shop_total AS ("
            f"SELECT shop_id, sum(model{args.modelnum}) AS sd_shop_qty_sold_day "
            "FROM daily_sid_predictions "
            "WHERE sale_date = %(curr_date)s "
            "GROUP BY shop_id) "
            "UPDATE shop_dates sd "
            "SET sd_shop_qty_sold_day = st.sd_shop_qty_sold_day "
            "FROM shop_total st "
            "WHERE sd.sale_date = %(curr_date)s AND sd.shop_id = st.shop_id; "
        )
        query = (
            "UPDATE shop_item_dates sid "
            f"SET sid_shop_item_qty_sold_day = dsp.model{args.modelnum} "
            "FROM daily_sid_predictions dsp "
            "WHERE sid.sale_date = %(curr_date)s AND sid.shop_id = dsp.shop_id AND "
            "sid.item_id = dsp.item_id; "
        )
        # insert non-zero sid predicted quantity into sales_cleaned
        query = (
            "INSERT INTO sales_cleaned (shop_id, item_id, sale_date, item_cnt_day) "
            f"SELECT shop_id, item_id, sale_date, model{args.modelnum} AS item_cnt_day "
            "FROM daily_sid_predictions "
            f"WHERE model{args.modelnum} <> 0 AND sale_date = %(curr_date)s"
        )

    if args.stop == True:
        stop_instance()

    # copy log file to S3 bucket
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
