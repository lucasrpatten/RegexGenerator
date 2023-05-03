from model.database_management import Database
import re

class DatabaseManager(Database):
    def __init__(self, database_path="./data.db", table="patterns"):
        super().__init__(database_path=database_path, table=table)

    def addData(self, accepted_texts: list[str], rejected_texts: list[str], output_pattern: str):
        pattern = re.compile(output_pattern)

        for i in accepted_texts:
            if not isinstance(i, str):
                raise TypeError(
                    f"'{i}' is not a string")
            if pattern.fullmatch(i) is None:
                raise AssertionError(
                    f"'{i}' doesn't match pattern '{output_pattern}'")

        for i in rejected_texts:
            if not isinstance(i, str):
                raise TypeError(
                    f"'{i}' is not a string")
            if pattern.fullmatch(i) is not None:
                raise AssertionError(
                    f"'{i}' should not match pattern '{output_pattern}'")

        self.add_row(accepted_texts, rejected_texts, output_pattern)
