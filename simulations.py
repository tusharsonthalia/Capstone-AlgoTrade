import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
sns.set()

def pct_change(x, period=1):
    x = np.array(x)
    return ((x[period:] - x[:-period]) / x[:-period])

def dynamic_volatility_monte_carlo(df, sim_name, stock_name, number_simulation=100, predict_day=30):
    results = pd.DataFrame()

    for i in tqdm(range(number_simulation)):
        prices = df.Close.values[-predict_day:].tolist()
        volatility = pct_change(prices[-predict_day:]).std()
        for d in range(predict_day):
            prices.append(prices[-1] * (1 + np.random.normal(0, volatility)))
            volatility = pct_change(prices[-predict_day:]).std()
        results[i] = pd.Series(prices[-predict_day:]).values

    name_of_graph = save_graph(results, df)
    text = get_text(sim_name, stock_name, number_simulation, predict_day)

    return name_of_graph, text

def simple_monte_carlo(df, sim_name, stock_name, number_simulation=100, predict_day=30):
    returns = df.Close.pct_change()
    volatility = returns.std()
    results = pd.DataFrame()

    for i in tqdm(range(number_simulation)):
        prices = []
        prices.append(df.Close.iloc[-1])
        for d in range(predict_day):
            prices.append(prices[d] * (1 + np.random.normal(0, volatility)))
        results[i] = pd.Series(prices).values
    
    name_of_graph = save_graph(results, df)
    text = get_text(sim_name, stock_name, number_simulation, predict_day)

    return name_of_graph, text

def drift_monte_carlo(df, sim_name, stock_name, number_simulation=100, predict_day=30):
    close = df['Close'].tolist()
    returns = pd.DataFrame(close).pct_change()
    last_price = close[-1]
    results = pd.DataFrame()
    avg_daily_ret = returns.mean()
    variance = returns.var()
    daily_vol = returns.std()
    daily_drift = avg_daily_ret - (variance / 2)
    drift = daily_drift - 0.5 * daily_vol ** 2

    results = pd.DataFrame()

    for i in tqdm(range(number_simulation)):
        prices = []
        prices.append(df.Close.iloc[-1])
        for d in range(predict_day):
            shock = [drift + daily_vol * np.random.normal()]
            shock = np.mean(shock)
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        results[i] = prices

    name_of_graph = save_graph(results, df)
    text = get_text(sim_name, stock_name, number_simulation, predict_day)

    return name_of_graph, text

def save_graph(results, df):
    raveled = results.values.ravel()
    raveled.sort()
    cp_raveled = raveled.copy()

    plt.figure(figsize=(17,5), dpi=1200)
    plt.subplot(1,3,1)
    plt.plot(results)
    plt.ylabel('Value')
    plt.xlabel('Simulated days')
    plt.title('Simulations')
    plt.subplot(1,3,2)
    sns.distplot(df.Close,norm_hist=True)
    plt.ylabel('')
    plt.title('$\mu$ = %.2f, $\sigma$ = %.2f'%(df.Close.mean(),df.Close.std()))
    plt.subplot(1,3,3)
    sns.distplot(raveled,norm_hist=True,label='monte carlo samples')
    sns.distplot(df.Close,norm_hist=True,label='real samples')
    plt.ylabel('')
    plt.title('simulation $\mu$ = %.2f, $\sigma$ = %.2f'%(raveled.mean(),raveled.std()))
    plt.legend()

    name = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H-%M-%S')
    location = f'static/graphs/{name}.jpg'
    plt.savefig(location, dpi=1200)
    plt.close()

    return location

def get_text(sim_name, stock_name, num_sim, num_predict):
    text = [
        f'You selected {sim_name} for running on stock data for {stock_name}.',
        f'The Number of Simulation selected was: {num_sim} and the Number of days to be predicted was: {num_predict}',
        f'The Results of the simulation are below:'
    ]

    return text

class Simulator:
    def __init__(self, stock, simulation_name, number_simulation, predict_days):
        self.number_simulation = number_simulation
        self.predict_days = predict_days
        self.stock = stock
        self.simulation_name = simulation_name

        self.stock_data = self.get_stock_data(self.stock)
        self.simulation_directory = {
            "Dynamic volatility Monte Carlo": dynamic_volatility_monte_carlo ,
            "Drift Monte Carlo": drift_monte_carlo ,
            "Simple Monte Carlo": simple_monte_carlo 
        }
        self.simulation = self.simulation_directory[self.simulation_name]

    def get_stock_data(self, stock):
        path = f'data/stocks/{stock.upper()}.csv'
        df = pd.read_csv(path).dropna()
        df.reset_index(drop=True)

        return df

    def perform_simulation(self):
        return self.simulation(self.stock_data, self.simulation_name, self.stock, self.number_simulation, self.predict_days)
