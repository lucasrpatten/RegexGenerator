"""
Contains the Database class, used for database management
"""

import sqlite3


class Database:
    """Class containing methods to manage the database

    Args:
        database_path (str, optional): Path of the database. Defaults to "./data.db".
        table (str, optional): Table containing data. Defaults to "patterns".
    """

    def __init__(self, database_path: str = "./data.db", table: str = "patterns"):

        self.db_path = database_path
        self.table = table
        self.conn, self.cursor = self.connect()
        self.length = self.get_length()

    def connect(self):
        """Connect to the database

        Returns:
            Connection: Connection to the database
            Cursor: Database cursor
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        return conn, cursor

    def commit(self):
        """Writes changes to the database
        """
        self.conn.commit()

    def close(self, commit: bool = True):
        """Closes the database connection

        Args:
            commit (bool, optional): Whether to write changes before closing. Defaults to True.
        """
        if commit:
            self.conn.commit()
        self.conn.close()
        self.conn, self.cursor = None, None

    def query(self, query: str, params: str | None = None):
        """Queries the database

        Args:
            query (str): Query to execute
            params (str, optional): Query parameters. Defaults to None.

        Returns:
            list[Any]: Query results
        """
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_length(self):
        """Gets the number of elements in the table

        Returns:
            int: table length
        """
        result = self.query(f"SELECT COUNT(*) FROM {self.table}")
        length = result[0][0]
        return length

    def add_row(
        self,
        accepted_texts: list[str],
        rejected_texts: list[str],
        output_pattern: list[str]
    ):
        """Adds a row to the table

        Args:
            accepted_patterns (list[str]): A list of accepted texts
            rejected_patterns (list[str]): A list of rejected texts
            output_pattern (list[str]): A list of output regex patterns
        """
        accepted_str = str(accepted_texts)
        rejected_str = str(rejected_texts)
        output_str = str(output_pattern)
        self.query(f"INSERT INTO {self.table} (pattern, match, reject) VALUES (?,?,?)", (
            output_str, accepted_str, rejected_str))
        self.commit()

    def delete_row(self, row_id: int):
        """Deletes a row by id

        Args:
            row_id (int): id of row to be deleted
        """
        self.query(f"DELETE FROM {self.table} WHERE id = {row_id}")
        self.query(f"UPDATE {self.table} SET id = id - 1 WHERE id > {row_id}")

    def get_table(self):
        """Returns the table

        Returns:
            list[Any]: Table as a list
        """
        table = self.query(f"SELECT * FROM {self.table}")
        return table
