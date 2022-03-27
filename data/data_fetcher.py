import time
import datetime
import pandas as pd
import argparse
from io import StringIO
import os

from utils import ProgressBar


class DataGenerator:
    "Class to gather data for a given list of stocks."

    def __init__(self, stock_list, start_date, end_date, interval='1d',
                 file_loc='../data/', attempt_tol=3):
        """
        Parameters:
        ----------
        stock_list:     list
                        The list of stocks to gather data.

        start_date:     str, format 'YYYY-MM-DD'
                        The start date of the data required.

        end_date:       str, format 'YYYY-MM-DD'
                        The end date of the data required.

        interval:       str, default '1d'
            valid:      '1d', '1wk', '1mo'
                        The bar interval for data.

        file_loc:       str, default '../data/'
                        The folder to save the datasets.

        attempt_tol:    int, default 3
                        Number of retries to download the data.
        """
        self.stock_list = stock_list
        self.start_date = self.parse_date(start_date)
        self.end_date = self.parse_date(end_date)
        self.interval = interval
        self.file_loc = file_loc
        self.data = None
        self.progress = ProgressBar()
        self.attempt = 0
        self.attempt_tol = attempt_tol

    def parse_date(self, date):
        """
        Method to convert str into datetime object.

        Parameters:
        ----------
        date:   str
                Date in 'YYYY-MM-DD' format.

        Returns:
        -------
        Datetime:   A datetime object
        """
        return datetime.datetime.strptime(date, '%Y-%m-%d')

    def yahoo_date_format(self, date):
        """
        Method to convert datetime into yahoo finance accepted time format.

        Parameters:
        ----------
        date:   datetime
                A datetime object.

        Returns:
        -------
        Int:    Yahoo finance accepted time format.
        """
        return int(time.mktime(date.timetuple()))

    def _request_data(self):
        """
        Private method to request data for a list of stocks.
        """
        stocks_to_request = len(self.stock_list)
        stocks_not_done = []
        self.progress.start_progress(stocks_to_request)
        for stock in self.stock_list:
            period1 = self.yahoo_date_format(self.start_date)
            period2 = self.yahoo_date_format(self.end_date)
            query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}.NS?period1={period1}&period2={period2}&interval={self.interval}&events=history&includeAdjustedClose=true'
            try:
                df = pd.read_csv(query_string)
                df.to_csv(os.path.join(self.file_loc,
                          f'{stock}.csv'), index=False)
            except Exception as e:
                if 'HTTP Error 404:' not in str(e):
                    stocks_not_done.append(stock)

            self.progress.update_progress(stock)

        self.stock_list = stocks_not_done

    def request_data(self):
        """
        Method to request data for a list of stocks.
        """
        if self.attempt < self.attempt_tol:
            print(f"\nAttempt {self.attempt + 1}:\n")
            self._request_data()
            self.attempt += 1
            if self.stock_list:
                self.request_data()

        if self.stock_list:
            print(
                f"\nCould not fetch data for {len(self.stock_list)} stock(s):")
            print(*self.stock_list, sep='\t')


class DataUpdater(DataGenerator):
    """
    Class to intelligently update the dataset of a given list of 
    stocks from the last recorded date until a given date.
    """

    def __init__(self, stock_list, start_date="", end_date="",
                 interval='1d', file_loc='../data/', attempt_tol=3):
        """
        Parameters:
        ----------
        stock_list:     list
                        The list of stocks to gather data.

        start_date:     str, format 'YYYY-MM-DD'
                        The start date of the data required.

        end_date:       str, format 'YYYY-MM-DD'
                        The end date of the data required.

        interval:       str, default '1d'
            valid:      '1d', '1wk', '1mo'
                        The bar interval for data.

        file_loc:       str, default '../data/'
                        The folder to save the datasets.

        attempt_tol:    int, default 3
                        Number of retries to download the data.
        """

        if not end_date:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            start_date = "2000-01-01"
        super().__init__(stock_list, start_date, end_date,
                         interval=interval, file_loc=file_loc,
                         attempt_tol=attempt_tol)
        self.stock_list = [(stock, self.start_date)
                           for stock in self.stock_list]
        self._prepare_stock_list()

    def _prepare_stock_list(self):
        """
        Private method to prepare a list of tuples 
        with the stock symbol and start date.
        """
        for index, stock_tuple in enumerate(self.stock_list):
            stock, start_date = stock_tuple
            file_name = os.path.join(self.file_loc, f'{stock}.csv')
            if os.path.exists(file_name):
                with open(file_name) as f:
                    f = f.readlines()
                    header = f[0]
                    last_date = f[-1]
                    del f
                if header != last_date:
                    df = pd.read_csv(StringIO(header + last_date))
                    start_date = self.parse_date(
                        df.loc[0, 'Date']) + datetime.timedelta(days=1)
            self.stock_list[index] = (stock, start_date)

    def _request_data(self):
        """
        Private method to request data for a list of stocks.
        """
        stocks_to_request = len(self.stock_list)
        stocks_not_done = []
        self.progress.start_progress(stocks_to_request)
        for stock, start_date in self.stock_list:
            period1 = self.yahoo_date_format(start_date)
            period2 = self.yahoo_date_format(self.end_date)
            query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}.NS?period1={period1}&period2={period2}&interval={self.interval}&events=history&includeAdjustedClose=true'
            try:
                df = pd.read_csv(query_string)
                file_name = os.path.join(self.file_loc, f'{stock}.csv')
                if os.path.exists(file_name):
                    df.to_csv(file_name, mode='a', header=False, index=False)
                else:
                    df.to_csv(file_name, index=False)
            except Exception as e:
                if 'HTTP Error 404:' not in str(e):
                    stocks_not_done.append((stock, start_date))

            self.progress.update_progress(stock)

        self.stock_list = stocks_not_done

    def request_data(self):
        """
        Method to request data for a list of stocks.
        """
        if self.attempt < self.attempt_tol:
            print(f"\nAttempt {self.attempt + 1}:\n")
            self._request_data()
            self.attempt += 1
            if self.stock_list:
                print(
                    f"\nCould not fetch data for {len(self.stock_list)} stock(s):")
                stocks = [stock[0] for stock in self.stock_list]
                print(*stocks, sep='\t')
                self.request_data()

        if self.stock_list:
            print(
                f"\nCould not fetch data for {len(self.stock_list)} stock(s):")
            stocks = [stock[0] for stock in self.stock_list]
            print(*stocks, sep='\n')


def main():
    parser = argparse.ArgumentParser(
        description='Request data for a list of stocks for a given period.')
    parser.add_argument('stocks',
                        help='location to the csv file of stocks')
    parser.add_argument('data',
                        help='Location to store the csv files. If not mentioned, will save to the stock list file')
    parser.add_argument('-sd',
                        '--start_date',
                        default="",
                        help='The start date of the data')
    parser.add_argument('-ed',
                        '--end_date',
                        default="",
                        help='The end date of the data')
    parser.add_argument('-a',
                        '--attempt_tolerance',
                        type=int,
                        default=3,
                        help='Tolerance for the number of retries to request data.')
    parser.add_argument('-i',
                        '--interval',
                        default='1d',
                        choices=['1d', '1wk', '1mo'],
                        help='The interval for each bar of the data.')
    args = parser.parse_args()
    stock_list = pd.read_csv(args.stocks)['Symbol']
    du = DataUpdater(stock_list, start_date=args.start_date, end_date=args.end_date,
                     attempt_tol=args.attempt_tolerance, file_loc=args.data, interval=args.interval)
    du.request_data()


if __name__ == '__main__':
    main()
