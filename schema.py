from database import get_connection


def get_schema() -> str:
    """fetches schema information from Snowflake and formats it as a string"""
    conn = get_connection()
    cursor = conn.cursor()

    # information_schema is a snowflake-metadata-schema that automatically exists
    # CURRENT_SCHEMA() is a snowflake function
    cursor.execute("""
            SELECT table_name, column_name, data_type, comment
            FROM information_schema.columns
            WHERE table_schema = CURRENT_SCHEMA()
            ORDER BY table_name, ordinal_position       
    """)

    # get all rows, list of tuples with 3 elements
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    # formatting
    schema_text = "Database schema:\n"
    # in case of multiple tables, keep track of current table
    current_table = ""

    # iterates through rows and formats them into a readable string
    for table, column, dtype, comment in rows:
        # adds table name if new table beginns
        if table != current_table:
            schema_text += f"\nTable: {table}\n"
            current_table = table
        schema_text += f"{column} ({dtype}), Comment:{comment}\n"

    return schema_text


# test
if __name__ == "__main__":
    print(get_schema())
