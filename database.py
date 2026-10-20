import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """fetches env values and builds connection to snowflake"""
    conn = snowflake.connector.connect(  # opens connection to snowflake using env variables
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        session_parameters={
            "USE_CACHED_RESULTS": False
        },  # deactivate cache so Snowflake measures actual execution times
    )

    cursor = conn.cursor()
    cursor.close()

    return conn


def execute_sql(sql: str) -> dict:
    """executes SQL command and returns results as a dictionary"""
    try:
        conn = get_connection()  # uses function above to get connection
        cursor = conn.cursor()  # cursor enables execution of SQL commands

        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + " LIMIT 1000"

        cursor.execute(sql)  # send SQL command to snowflake for execution

        # cursor.description contains metadata about the columns
        columns = [col[0] for col in cursor.description]  # list of column names

        # fetchall() retrieves all rows of the query result
        # returns a list of tuples, where each tuple represents a row in the result set
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return {"success": True, "columns": columns, "rows": rows, "error": None}

    except Exception as e:
        # if any error occurs during connection -> error message
        return {"success": False, "columns": [], "rows": [], "error": str(e)}


# test connection
if __name__ == "__main__":
    result = execute_sql("SELECT CURRENT_DATE()")
    print(result)
