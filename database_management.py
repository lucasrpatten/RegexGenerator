import sqlite3


class Database:
    def __init__(self, database_path="./data.db"):
        self.db_path = database_path
        self.table = "patterns"
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

    def query(self, query):
        conn, cursor = self.connect()
        cursor.execute(query)

    def get_length(self):
        self.query(f"SELECT COUNT(*) FROM {self.table}")
        result = self.cursor.fetchone()[0]
        self.length = len(result)
        return result
