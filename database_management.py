import sqlite3


class Database:
    def __init__(self, database_path="./data.db", table="patterns"):
        self.db_path = database_path
        self.table = table
        self.conn, self.cursor = self.connect()
        self.length = self.get_length()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        return conn, cursor

    def commit(self):
        self.conn.commit()

    def close(self, commit=True):
        if commit:
            self.conn.commit()
        self.conn.close()
        self.conn, self.cursor = None, None

    def query(self, query, params=""):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_length(self):
        result = self.query(f"SELECT COUNT(*) FROM {self.table}")
        length = result[0][0]
        return length

    def add_row(self, accepted_patterns, rejected_patterns, output_pattern):
        accepted_str = str(accepted_patterns)
        rejected_str = str(rejected_patterns)
        output_str = str(output_pattern)
        # How to add a row in sql using the sqlite3 module?
        self.query(f"INSERT INTO {self.table} (pattern, match, reject) VALUES (?,?,?)", (
            output_str, accepted_str, rejected_str))
        self.commit()

    def delete_row(self, row_id):
        row_id = int(row_id)
        self.query(f"DELETE FROM {self.table} WHERE id = {row_id}")
        self.query(f"UPDATE {self.table} SET id = id - 1 WHERE id > {row_id}")

    def get_table(self):
        table = self.query(f"SELECT * FROM {self.table}")
        return table

