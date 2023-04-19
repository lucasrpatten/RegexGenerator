from database_management import Database
import re
import ast


class DatabaseManager(Database):
    def __init__(self, database_path="./data.db", table="patterns"):
        super().__init__(database_path=database_path, table=table)

    def addData(self, accepted_patterns, rejected_patterns, output_pattern):
        pattern = re.compile(output_pattern)

        for i in accepted_patterns:
            if not isinstance(i, str):
                raise TypeError(
                    f"'{i}' is not a string")
            if not pattern.fullmatch(i):
                raise AssertionError(
                    f"'{i}' doesn't match pattern '{output_pattern}'")

        for i in rejected_patterns:
            if not isinstance(i, str):
                raise TypeError(
                    f"'{i}' is not a string")
            if pattern.fullmatch(i):
                raise AssertionError(
                    f"'{i}' should not match pattern '{output_pattern}'")

        self.add_row(accepted_patterns, rejected_patterns, output_pattern)
