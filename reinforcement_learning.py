import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()

from rl_models.evolution import Agent as evolution_agent
from rl_models.evolution import Model as evolution_model
from rl_models.turtle import Agent as turtle_agent
from rl_models.signal_rolling import Agent as signal_rolling_agent
from rl_models.moving_average import Agent as moving_average_agent

class ReinforcementLearner:
    def __init__(self, stock, start_date, end_date, window_size, initial_money, model_name, skip=1):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.start = self.format_date(start_date)
        self.end = self.format_date(end_date)

        self.stock_data = self.get_stock_data(self.stock)
        self.window_size = window_size
        self.skip = skip
        self.initial_money = initial_money
        self.model_name = model_name

        self.algorithm_directory = {
            "Evolution-strategy agent": self.evolution, 
            "Moving-average agent": self.moving, 
            "Signal rolling agent": self.signal, 
            "Turtle-trading agent": self.turtle
        }

        self.model = self.algorithm_directory[self.model_name]
        self.model_info = self.model()

    def format_date(self, date):
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        return date
    
    def get_stock_data(self, stock):
        path = f'data/stocks/{stock.upper()}.csv'
        df = pd.read_csv(path).dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= self.start) & (df['Date'] <= self.end)]
        
        return df

    def moving(self):
        short_window = int(0.025 * len(self.stock_data))
        long_window = int(0.05 * len(self.stock_data))

        signals = pd.DataFrame(index=self.stock_data.index)
        signals['signal'] = 0.0

        signals['short_ma'] = self.stock_data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_ma'] = self.stock_data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

        signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()

        states_buy, states_sell, total_gains, invest = moving_average_agent(
            self.stock_data.Close.tolist(), signals['signal'].tolist(), self.stock_data, self.initial_money
        )

        close = self.stock_data['Close']
        plt.figure(figsize = (15,5), dpi=1200)
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
        plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
        plt.title(f"Total Gains: {total_gains} ({invest}%) on an \ninitial capital of {self.initial_money}")
        plt.legend()

        name_of_graph = self.save_graph()

        return states_buy, states_sell, total_gains, invest, name_of_graph

    def signal(self):

        states_buy, states_sell, total_gains, invest = signal_rolling_agent(self.stock_data.Close.tolist(), initial_state = 1, delay = 4, initial_money = self.initial_money)

        close = self.stock_data['Close']
        plt.figure(figsize = (15,5), dpi=1200)
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
        plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
        plt.title(f"Total Gains: {total_gains} ({invest}%) on an \ninitial capital of {self.initial_money}")
        plt.legend()

        name_of_graph = self.save_graph()

        return states_buy, states_sell, total_gains, invest, name_of_graph

    def evolution(self):
        close = self.stock_data.Close.values.tolist()
        model = evolution_model(
            input_size=self.window_size, 
            layer_size=100, 
            output_size=3
        )
        agent = evolution_agent(
            model=model,
            window_size=self.window_size,
            trend=close,
            skip=self.skip,
            initial_money=self.initial_money
        )
        agent.fit(iterations=50, checkpoint=10)
        states_buy, states_sell, total_gains, invest = agent.buy()

        plt.figure(figsize = (15, 5), dpi=1200)
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
        plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
        plt.title(f"Total Gains: {total_gains} ({invest}%) on an \ninitial capital of {self.initial_money}")
        plt.legend()

        name_of_graph = self.save_graph()

        return states_buy, states_sell, total_gains, invest, name_of_graph

    def turtle(self):
        count = int(np.ceil(len(self.stock_data) * 0.1))
        signals = pd.DataFrame(index=self.stock_data.index)
        signals['signal'] = 0.0
        signals['trend'] = self.stock_data['Close']
        signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
        signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
        signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
        signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
        
        states_buy, states_sell, total_gains, invest = turtle_agent(
            self.stock_data.Close.tolist(), signals['signal'].tolist(), self.stock_data, self.initial_money
        )

        close = self.stock_data['Close']
        plt.figure(figsize = (15,5), dpi=1200)
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
        plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
        plt.title(f"Total Gains: {total_gains} ({invest}%) on an \ninitial capital of {self.initial_money}")
        plt.legend()

        name_of_graph = self.save_graph()

        return states_buy, states_sell, total_gains, invest, name_of_graph

    def save_graph(self):
        name = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H-%M-%S')
        location = f'static/graphs/{name}.jpg'
        plt.savefig(location, dpi=1200)
        plt.close()

        return location

    def get_graph_and_text(self):
        states_buy, states_sell, total_gains, invest, name_of_graph = self.model_info
        number_days = self.stock_data.Date.nunique() - 1
        if len(states_buy) > 0 and states_buy[-1] == number_days:
            action = 'Buy'
        elif len(states_sell) > 0 and states_sell[-1] == number_days:
            action = 'Sell'
        else:
            action = 'Hold'

        text = [
            f'You selected {self.model_name} for running on stock data for {self.stock} ranging from {self.start_date} to {self.end_date}.',
            f'The window size selected was: {self.window_size} and the inital money defined was: {self.initial_money}',
            f'The Performance of the model is below:',
            f'The Total Gains from {self.model_name} was: {total_gains}. The model currently suggests you to {action} the {self.stock} stock.'
        ]

        return text, name_of_graph
