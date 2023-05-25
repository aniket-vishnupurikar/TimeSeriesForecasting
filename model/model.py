import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import io
import warnings
warnings.filterwarnings('ignore')


class Model():

    def __init__(self):
        pass

    def arima(self, data, frac, params, n_future, start):

        """
        data: univariate time series
        frac = fraction of data to be used for training
        params: a tuple representing (p,d,q)
        n_future: number of future forecasts

        """

        # Splitting
        f = int(len(data) * frac)
        train_data = data.iloc[:f]
        test_data = data.iloc[f:]
        # model fitting
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train_data, order=params)
        model_fit = model.fit()
        # test data orediction
        yhat_test = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        # train data prediction
        yhat_train = model_fit.predict(start=0, end=len(train_data) - 1)
        # getting train and test rmse
        from sklearn.metrics import mean_squared_error
        train_rmse = np.round(np.sqrt(mean_squared_error(test_data, yhat_test)), 2)
        test_rmse = np.round(np.sqrt(mean_squared_error(train_data[params[0]:], yhat_train[params[0]:])), 2)
        forecast = model_fit.predict(start=len(data), end=len(data) + n_future - 1)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(forecast)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def sarima(self, data, frac, params, s_params, trend, n_future, start):

        """
        data: univariate time series
        frac: fraction of the data to be used for training
        params: a tuple representing (p,d,q)
        s_params; a tuple of seasonal parameters
        trend: trend parameter
        n_future: number of future forecasts

        """

        # Splitting
        f = int(len(data) * frac)
        train_data = data.iloc[:f]
        test_data = data.iloc[f:]
        # model fitting
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(train_data, order=params, seasonal_order=s_params, trend=trend)
        model_fit = model.fit()
        # test data orediction
        yhat_test = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        # train data prediction
        yhat_train = model_fit.predict(start=0, end=len(train_data) - 1)
        # getting train and test rmse
        from sklearn.metrics import mean_squared_error
        test_rmse = np.round(np.sqrt(mean_squared_error(test_data, yhat_test)))
        train_rmse = np.round(np.sqrt(mean_squared_error(train_data[params[0]:], yhat_train[params[0]:])))
        forecast = model_fit.predict(start=len(data), end=len(data) + n_future - 1)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(forecast)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def exp_smoothing_1(self, data, frac, smoothing_level, n_future, start):

        """
        data: univariate timeseries
        frac: fraction of the data to be used for training
        smoothing_level: smoothing level for single exponential smoothing
        n_future: number of future forecasts
        """

        # Splitting
        f = int(len(data) * frac)
        train_data = data.iloc[:f]
        test_data = data.iloc[f:]

        # model fitting

        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        model = SimpleExpSmoothing(train_data)
        model_fit = model.fit(smoothing_level=smoothing_level)
        # test data orediction
        yhat_test = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        # train data prediction
        yhat_train = model_fit.predict(start=0, end=len(train_data) - 1)
        # getting train and test rmse
        from sklearn.metrics import mean_squared_error
        test_rmse = np.round(np.sqrt(mean_squared_error(test_data, yhat_test)), 2)
        train_rmse = np.round(np.sqrt(mean_squared_error(train_data, yhat_train)), 2)
        forecast = model_fit.predict(start=len(data), end=len(data) + n_future - 1)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(forecast)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def exp_smoothing_2(self, data, frac, params, n_future, start):

        """
        data: univariate timeseries
        frac: fraction of the d
        params: dict of holt-winters parameters=[trend, damped, seasonal, seasonal_periods]
        n_future: number of future forecasts

        """

        # Splitting
        f = int(len(data) * frac)
        train_data = data.iloc[:f]
        test_data = data.iloc[f:]
        # fitting the model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(train_data, trend=params['trend'], damped=params['damped'],
                                     seasonal=params['seasonal'],
                                     seasonal_periods=params['seasonal_period'])
        model_fit = model.fit(optimized=True)
        # test data orediction
        yhat_test = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        # train data prediction
        yhat_train = model_fit.predict(start=0, end=len(train_data) - 1)
        # getting train and test rmse
        from sklearn.metrics import mean_squared_error
        test_rmse = np.round(np.sqrt(mean_squared_error(test_data, yhat_test)), 2)
        train_rmse = np.round(np.sqrt(mean_squared_error(train_data, yhat_train)), 2)
        forecast = model_fit.predict(start=len(data), end=len(data) + n_future - 1)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(forecast)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def xgboost(self, data, frac, n_steps, n_estimators, n_future, start):

        """
        data: univariate timeseries
        n_steps: number of lags to considers
        n_estimators: number of estimators in xgboost
        n_future: number of future predictions
        """

        # preparing the data
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """

            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        n_steps = n_steps  # n_steps represents lag to create supervised learning dataset
        new_data = series_to_supervised(pd.DataFrame(data).values, n_steps)
        # splitting the data
        from sklearn.model_selection import train_test_split
        x = new_data.drop(["var1(t)"], axis=1)
        y = new_data["var1(t)"]
        train_size = int(new_data.shape[0] * frac)
        x_train = x.iloc[0:train_size]
        x_test = x.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        # model training
        from xgboost import XGBRegressor
        model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators)
        model.fit(x_train, y_train)
        # predictions
        yhat = model.predict(x_test)
        yhat = pd.Series(yhat, index=y_test.index)
        yhat_train = model.predict(x_train)
        yhat_train = pd.Series(yhat_train, index=y_train.index)
        from sklearn.metrics import mean_squared_error
        test_rmse = np.round(np.sqrt(mean_squared_error(y_test.values, yhat.values)), 2)
        train_rmse = np.round(np.sqrt(mean_squared_error(y_train.values, yhat_train.values)), 2)

        def future_prediction(model, n, prediction_data):
            """
            Makes n future predictions given a model and starting prediction data
            """
            predictions = list()
            for i in range(n):
                current_prediction = model.predict(prediction_data)
                predictions.append(current_prediction[0])
                prediction_data = np.append(prediction_data, current_prediction[0])
                prediction_data = np.delete(prediction_data, 0).reshape(1, -1)
            return np.array(predictions)

        prediction_data = x_test.iloc[-1].values.flatten().reshape(1, -1)
        future_predictions = future_prediction(model, n_future, prediction_data)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(future_predictions)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def rf(self, data, frac, n_steps, n_future, n_estimators, max_depth, min_samples_split, min_samples_leaf, start):

        """
        data: univariate timeseries
        frac: fraction of the data to be used to train
        n_steps: number of lags to considers
        n_future: number of future predictions
        rest if the parameters belong to random forest
        """

        # preparing the data
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """

            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        n_steps = n_steps  # n_steps represents lag to create supervised learning dataset
        new_data = series_to_supervised(pd.DataFrame(data).values, n_steps)
        # splitting the data
        from sklearn.model_selection import train_test_split
        x = new_data.drop(["var1(t)"], axis=1)
        y = new_data["var1(t)"]
        train_size = int(new_data.shape[0] * frac)
        x_train = x.iloc[0:train_size]
        x_test = x.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        # model training
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        model.fit(x_train, y_train)
        # predictions
        yhat = model.predict(x_test)
        yhat = pd.Series(yhat, index=y_test.index)
        yhat_train = model.predict(x_train)
        yhat_train = pd.Series(yhat_train, index=y_train.index)
        from sklearn.metrics import mean_squared_error
        test_rmse = np.round(np.sqrt(mean_squared_error(y_test.values, yhat.values)), 2)
        train_rmse = np.round(np.sqrt(mean_squared_error(y_train.values, yhat_train.values)), 2)

        def future_prediction(model, n, prediction_data):
            """
            Makes n future predictions given a model and starting prediction data
            """
            predictions = list()
            for i in range(n):
                current_prediction = model.predict(prediction_data)
                predictions.append(current_prediction[0])
                prediction_data = np.append(prediction_data, current_prediction[0])
                prediction_data = np.delete(prediction_data, 0).reshape(1, -1)
            return np.array(predictions)

        prediction_data = x_test.iloc[-1].values.flatten().reshape(1, -1)
        future_predictions = future_prediction(model, n_future, prediction_data)

        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = list(future_predictions)
        future.index = future['ds']

        return future['pred'], train_rmse, test_rmse

    def prophet(self, data, n_train, n_future, start):

        """
        data: univariate time series
        n_train: number of training data points
        n_future: number of future forecast points
        forecast_start: string representing starting date for forecast
        """
        import logging
        logging.getLogger('fbprophet').setLevel(logging.WARNING)

        prophet_df = pd.DataFrame({'ds': data.index, 'y': data.values}, index=data.index)
        # changing ds column to datetime
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        # train-test split
        train = prophet_df[:n_train]
        test = prophet_df[n_train:]
        # fitting the model
        from fbprophet import Prophet
        model = Prophet()
        # fit the model
        model.fit(train)
        # generating train and test predictions
        train_prediction = model.predict(pd.DataFrame(train['ds'], columns=['ds']))['yhat']
        test_prediction = model.predict(pd.DataFrame(test['ds'], columns=['ds']))['yhat']

        # creating train and test prediction dataframes
        train_prediction = pd.DataFrame(data=train_prediction.values, columns=['pred'], index=train.index)
        test_prediction = pd.DataFrame(data=test_prediction.values, columns=['pred'], index=test.index)
        # generating dates for future predictions
        import datetime
        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        # predicting for future
        future['pred'] = model.predict(future)['yhat']
        future.index = future['ds']
        # generating forecast from the start
        total_forecast = pd.concat(
            [train_prediction, test_prediction, pd.DataFrame(future['pred'], index=future.index)])
        # getting train and test rmse
        from sklearn.metrics import mean_squared_error
        test_rmse = np.sqrt(mean_squared_error(test['y'], test_prediction['pred']))
        train_rmse = np.sqrt(mean_squared_error(train['y'], train_prediction['pred']))

        return future['pred'], train_rmse, test_rmse

    def mlp(self, data, start, train_frac=0.8, n_future=24, lags=5):

        original_data = data.copy()

        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            """
            Frame a time series as a supervised learning dataset.
            Arguments:
                data: Sequence of observations as a list or NumPy array.
                n_in: Number of lag observations as input (X).
                n_out: Number of observations as output (y).
                dropnan: Boolean whether or not to drop rows with NaN values.
            Returns:
                Pandas DataFrame of series framed for supervised learning.
            """

            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        def future_prediction(model, n, prediction_data):

            """
            Makes n future predictions given a model and starting prediction data
            """
            predictions = list()
            for i in range(n):
                current_prediction = model.predict(prediction_data)
                predictions.append(current_prediction[0])
                prediction_data = np.append(prediction_data, current_prediction[0])
                prediction_data = np.delete(prediction_data, 0).reshape(1, -1)
            return np.array(predictions)

        sup_data = pd.DataFrame(data)
        n_steps = lags  # n_steps represents lag to create supervised learning dataset
        new_data = series_to_supervised(sup_data.values, n_steps)

        x = new_data.drop(["var1(t)"], axis=1)
        y = new_data["var1(t)"]

        train_size = int(new_data.shape[0] * train_frac)

        x_train = x.iloc[0:train_size]
        x_test = x.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        import tensorflow as tf

        ip = Input(shape=(lags,))
        dense1 = Dense(5, activation='relu')(ip)
        dense2 = Dense(3, activation='relu')(dense1)
        op = Dense(1, activation='linear')(dense2)
        model1 = Model(inputs=ip, outputs=op)
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        lr = ReduceLROnPlateau(monitor="val_root_mean_squared_error", factor=0.1, patience=10, verbose=0, mode="auto",
                               min_delta=10, min_lr=0.0001, )

        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        metric = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
        model1.compile(optimizer=opt, loss=loss, metrics=metric)

        model1.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=50, batch_size=2, verbose=0,
                   callbacks=[lr])

        # getting future prediction
        prediction_data = x_test.iloc[-1].values.flatten().reshape(1, -1)
        future_predictions = future_prediction(model1, n_future, prediction_data)
        future = pd.DataFrame(pd.date_range(start, periods=n_future, freq="M"), columns=['ds'])
        future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
        future['pred'] = future_predictions
        future.index = future['ds']

        # evaluating model
        train_loss, train_rmse = model1.evaluate(x_train, y_train, verbose=0)
        test_loss, test_rmse = model1.evaluate(x_test, y_test, verbose=0)

        return future['pred'], train_rmse, test_rmse
