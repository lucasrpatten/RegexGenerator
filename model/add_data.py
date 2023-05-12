from database_management import Database
import re
from bs4 import BeautifulSoup
import requests
import time

class WebScraper():
    @classmethod
    def collect(cls):
        d = DatabaseManager("./model/data.db", "patterns")
        for i in range(20, 5000):
            url = f"https://www.regexlib.com/REDetails.aspx?regexp_id={i}"
            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, "html.parser")
            expression = ""
            matches = []
            rejections = []
            try:
                expression = soup.find(id="ctl00_ContentPlaceHolder1_ExpressionLabel").getText()
                matches = soup.find(id="ctl00_ContentPlaceHolder1_MatchesLabel").getText().split(" | ")
                rejections = soup.find(id="ctl00_ContentPlaceHolder1_NonMatchesLabel").getText().split(" | ")
                print(i)
                print(expression, matches, rejections)
                if re.fullmatch(expression, matches) is None:
                    raise AttributeError()
                if re.fullmatch(expression, rejections) is not None:
                    raise AttributeError()
                d.addData(matches, rejections, expression)
            except (AttributeError, TypeError, re.error):
                pass
            time.sleep(0.1)

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

WebScraper.collect()