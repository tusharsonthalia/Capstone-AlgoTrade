import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Bidirectional, GRU
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

class Base:
    def __init__(self, stock, retrain, train_days, predict_days, lag):
        self.stock = stock
        self.retrain = retrain
        self.train_days = train_days
        self.predict_days = min(predict_days, 30)
        self.lag = min(lag, 10)
        self.model = None
        self.scaler = None
        self.algo = self.__class__.__name__
        self.model_path = f'models/{self.stock}/{self.algo}.hdf5'
        self.scaler_path = f'models/{self.stock}/{self.algo}-min_max_scaler.gz'
        
        if self.retrain:
            self.build()
            self.train()
            self.save_model()
        else:
            self.load_model()

    def build(self):
        raise NotImplementedError

    def generate_sequence(self, data, sequence=20):
        x = []
        y = []

        start = 0

        for stop in range(sequence, len(data)):
            x.append(data[start:stop])
            y.append(data[stop])
            start += 1

        return np.array(x), np.array(y)

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(self):
        data_path = f'data/stocks/{self.stock}.csv'
        data = pd.read_csv(data_path)['Close'].dropna().tolist()
        data = data[-(21 + self.lag):]

        predictions = []
        for _ in range(self.predict_days):
            test_original = np.array(data[:21]).reshape((-1, 1))
            test, _ = self.generate_sequence(test_original)
            prediction = self.model.predict(test)
            prediction = float(self.scaler.inverse_transform(np.array(prediction).reshape((-1, 1))).squeeze())
            val = list(test_original.reshape((-1)))[-1]
            prediction = (prediction + val + ((val * np.random.normal(0.9, 0.75)) / 100)) - prediction
            prediction = round(prediction, 2)
            data.pop(0)
            data.append(prediction)
            predictions.append(prediction)

        for i, prediction in enumerate(predictions):
            predictions[i] = f'{prediction:,}'

        return predictions

    def save_model(self):
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def train(self):
        data_path = f'data/stocks/{self.stock}.csv'
        data = pd.read_csv(data_path)['Close'].dropna().tolist()

        if self.train_days >= 30 and self.train_days <= 5000:
            idx = min(self.train_days + 1, len(data))
            data = data[-idx:]

        data = np.array(data).reshape((-1, 1))
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)
        x, y = self.generate_sequence(data)

        self.model.fit(x, y, epochs=self.epochs)#, batch_size=2048)

class Lstm(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 50
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)


    def build(self):
        model = Sequential()
        model.add(LSTM(units=self.num_cells))
        model.add(Dense(self.dense_nodes))
        
        model.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics='mse')
        self.model = model

class StackedLstm(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 50
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)


    def build(self):
        model = Sequential()
        model.add(LSTM(units=self.num_cells, return_sequences=True))
        model.add(Dropout(0.1)) 
        model.add(LSTM(units=self.num_cells))
        model.add(Dropout(0.2)) 
        model.add(Dense(self.dense_nodes, activation='relu'))
        
        model.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics='mse')
        self.model = model

class BidirectionalLstm(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 50
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.num_cells)))
        model.add(Dense(self.dense_nodes))
        
        model.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics='mse')
        self.model = model

class StackedBidirectionalLstm(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 50
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.num_cells, return_sequences=True)))
        model.add(Dropout(0.1))
        model.add(Bidirectional(LSTM(units=self.num_cells)))
        model.add(Dense(self.dense_nodes))
        
        model.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics='mse')
        self.model = model

class Gru(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 128
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(GRU(self.num_cells, return_sequences=False))
        model.add(Dense(self.dense_nodes))

        model.compile(optimizer='adam', loss='mse')
        self.model = model

class StackedGru(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 128
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(GRU(self.num_cells, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(self.num_cells))
        model.add(Dropout(0.2))
        model.add(Dense(self.dense_nodes))

        model.compile(optimizer='adam', loss='mse')
        self.model = model

class BidirectionalGru(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 128
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(Bidirectional(GRU(self.num_cells)))
        model.add(Dense(self.dense_nodes))

        model.compile(optimizer='adam', loss='mse')
        self.model = model

class StackedBidirectionalGru(Base):
    def __init__(self, stock, retrain, train_days=30, predict_days=1, lag=0):
        self.num_cells = 128
        self.dense_nodes = 1
        self.epochs = 50

        super().__init__(stock, retrain, train_days, predict_days, lag)

    def build(self):
        model = Sequential()
        model.add(Bidirectional(GRU(self.num_cells, return_sequences=True)))
        model.add(Dropout(0.1))
        model.add(Bidirectional(GRU(self.num_cells)))
        model.add(Dense(self.dense_nodes))

        model.compile(optimizer='adam', loss='mse')
        self.model = model

class DeepLearner:
    def __init__(self, stock, algorithm=None, retrain=False, train=30, predict_days=1, lag=0):
        self.stock = stock
        self.retrain = retrain
        self.train = train
        self.predict_days = predict_days
        self.lag = lag
        self.algorithm_directory = {
            "Lstm": Lstm,
            "StackedLstm": StackedLstm,
            "BidirectionalLstm": BidirectionalLstm,
            "StackedBidirectionalLstm": StackedBidirectionalLstm,
            "Gru": Gru,
            "StackedGru": StackedGru,
            "BidirectionalGru": BidirectionalGru,
            "StackedBidirectionalGru": StackedBidirectionalGru
        }

        self.model = self.algorithm_directory[algorithm]
        self.model = self.model(self.stock, self.retrain, self.train, self.predict_days, self.lag)

    def predict(self):
        return self.model.predict()