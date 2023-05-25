import numpy as np
import pandas as pd
import datetime
import streamlit as st
from streamlit_option_menu import option_menu
from model import model
from model.model import Model
st.markdown("<h1 style='text-align: center; color: Red;'>Global Forecast Engine</h1>", unsafe_allow_html=True)
option = option_menu(None, ["Home","Automated Mode","Analysis Mode"],
                       default_index=0, orientation="horizontal")
if option == "Automated Mode":
    st.write("Click below to see the example format for input data")
    if st.button("Input data Format"):
        st.image("data.PNG")
    data = st.file_uploader('Upload file here')
    if data is not None:
        st.write("Data Uploaded.")
        df = pd.read_excel(data, index_col=0)
        df = df.T
        complete_data = df
        complete_data.index = pd.to_datetime(complete_data.index)
        complete_data.index = pd.DatetimeIndex(complete_data.index.values,
                                           freq=complete_data.index.inferred_freq)

        import warnings

        warnings.filterwarnings('ignore')

        import logging

        logging.getLogger('fbprophet').setLevel(logging.WARNING)

        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.simplefilter('ignore', ConvergenceWarning)

        n_past = complete_data.shape[0]
        cols = st.columns(4)
        with cols[0]:
            start = st.text_input("Start Date for Forecast")
            if start:
                start_ = datetime.datetime.strptime(start, "%Y-%m-%d")
        with cols[1]:
            n_future = int(st.number_input("Forecast Horizon"))
        with cols[2]:
            n_test = int(st.number_input("Validation Data Size"))
        with cols[3]:
            lags = int(st.number_input("Number of lags for supervised learning models"))

        if start and n_future and n_test and lags:
            forecasts0 = list()  # list to store forecast for arima
            forecasts1 = list()  # list to store forecast for sarima
            forecasts2 = list()  # list to store forecast for ses
            forecasts3 = list()  # list to store forecast for ses2
            forecasts4 = list()  # list to store forecast for xgboost
            forecasts5 = list()  # list to store forecast for rf
            forecasts6 = list()  # list to store forecast for prophet
            forecasts7 = list()  # list to store forecast for mlp

            rmse0 = list()  # list to store train and test rmse of arima
            rmse1 = list()  # list to store train and test rmse of sarima
            rmse2 = list()  # list to store train and test rmse of ses
            rmse3 = list()  # list to store train and test rmse of ses2
            rmse4 = list()  # list to store train and test rmse of xgboost
            rmse5 = list()  # list to store train and test rmse of rf
            rmse6 = list()  # list to store train and test rmse of prophet
            rmse7 = list()  # list to store train and test rmse of mlp

            model_dict = {0: "arima", 1: "sarima", 2: "single exp smoothing", 3: "double exp smoothing", 4: "xgboost",
                          5: "random forest",
                          6: "prophet", 7: "mlp"}

            from tqdm import tqdm

            # Training and forecast parameters(May chnage depending on data)
            n_train = n_past - n_test
            frac = n_train / n_past
            parts = list(complete_data.columns)
            i = 1
            final_result = list()
            for part in tqdm(parts):
                part_data = complete_data[part]
                diff_list = list()
                test_rmse_list = list()
                best_model = list()
                var_list = list()

                try:
                    # arima
                    model0 = Model()
                    forecast0, train_rmse0, test_rmse0 = model0.arima(part_data, frac, (5, 1, 3), n_future, start)
                    forecasts0.append(forecast0)
                    rmse0.append([train_rmse0, test_rmse0])
                    diff0 = abs(train_rmse0 - test_rmse0)
                    diff_list.append(diff0)
                    test_rmse_list.append(test_rmse0)
                    var0 = forecast0.var()
                    var_list.append(var0)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts0.append(future['pred'])
                    rmse0.append([np.nan, np.nan])
                    diff0 = np.inf
                    diff_list.append(diff0)
                    test_rmse_list.append(np.inf)
                    print("error in arima for {}:{}".format(part, e))

                try:
                    # sarima
                    model1 = Model()
                    forecast1, train_rmse1, test_rmse1 = model1.sarima(part_data, frac, (5, 1, 3), (1, 1, 1, 12), 'c',
                                                                       n_future, start)
                    forecasts1.append(forecast1)
                    rmse1.append([train_rmse1, test_rmse1])
                    diff1 = abs(train_rmse1 - test_rmse1)
                    diff_list.append(diff1)
                    test_rmse_list.append(test_rmse1)
                    var1 = forecast1.var()
                    var_list.append(var1)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts1.append(future['pred'])
                    rmse1.append([np.nan, np.nan])
                    diff1 = np.inf
                    diff_list.append(diff1)
                    test_rmse_list.append(np.inf)
                    print("error in sarima for {}:{}".format(part, e))

                try:

                    # ses
                    model2 = Model()
                    forecast2, train_rmse2, test_rmse2 = model2.exp_smoothing_1(part_data, frac, 0.1, n_future, start)
                    forecasts2.append(forecast2)
                    rmse2.append([train_rmse2, test_rmse2])
                    diff2 = abs(train_rmse2 - test_rmse2)
                    diff_list.append(diff2)
                    test_rmse_list.append(test_rmse2)
                    var2 = forecast2.var()
                    var_list.append(var2)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts2.append(future['pred'])
                    rmse2.append([np.nan, np.nan])
                    diff2 = np.inf
                    diff_list.append(diff2)
                    test_rmse_list.append(np.inf)
                    print("error in ses for {}:{}".format(part, e))

                try:

                    # ses2
                    model3 = Model()
                    params_dict = {'trend': None, 'damped': False, 'seasonal': 'add', 'seasonal_period': 12}
                    forecast3, train_rmse3, test_rmse3 = model3.exp_smoothing_2(part_data, frac, params_dict, n_future,
                                                                                start=start)
                    forecasts3.append(forecast3)
                    rmse3.append([train_rmse3, test_rmse3])
                    diff3 = abs(train_rmse3 - test_rmse3)
                    diff_list.append(diff3)
                    test_rmse_list.append(test_rmse3)
                    var3 = forecast3.var()
                    var_list.append(var3)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts3.append(future['pred'])
                    rmse3.append([np.nan, np.nan])
                    diff3 = np.inf
                    diff_list.append(diff3)
                    test_rmse_list.append(np.inf)
                    print("error in ses2 for {}:{}".format(part, e))

                try:

                    # xgboost
                    model4 = Model()
                    forecast4, train_rmse4, test_rmse4 = model4.xgboost(part_data, frac, lags, 100, n_future, start)
                    forecasts4.append(forecast4)
                    rmse4.append([train_rmse4, test_rmse4])
                    diff4 = abs(train_rmse4 - test_rmse4)
                    diff_list.append(diff4)
                    test_rmse_list.append(test_rmse4)
                    var4 = forecast4.var()
                    var_list.append(var4)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']
                    forecasts4.append(future['pred'])
                    rmse4.append([np.nan, np.nan])
                    diff4 = np.inf
                    diff_list.append(diff4)
                    test_rmse_list.append(np.inf)
                    print("error in xgboost for {}:{}".format(part, e))

                try:

                    # RF
                    model5 = Model()
                    forecast5, train_rmse5, test_rmse5 = model5.rf(part_data, frac, lags, n_future, 120, 3, 3, 3, start)
                    forecasts5.append(forecast5)
                    rmse5.append([train_rmse5, test_rmse5])
                    diff5 = abs(train_rmse5 - test_rmse5)
                    diff_list.append(diff5)
                    test_rmse_list.append(test_rmse5)
                    var5 = forecast5.var()
                    var_list.append(var5)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * 24
                    future.index = future['ds']

                    forecasts5.append(future['pred'])
                    rmse5.append([np.nan, np.nan])
                    diff5 = np.inf
                    diff_list.append(diff5)
                    test_rmse_list.append(np.inf)
                    print("error in rf for {}:{}".format(part, e))

                try:

                    # prophet
                    model6 = Model()
                    forecast6, train_rmse6, test_rmse6 = model6.prophet(part_data, n_train, n_future, start)
                    forecasts6.append(forecast6)
                    rmse6.append([train_rmse6, test_rmse6])
                    diff6 = abs(train_rmse6 - test_rmse6)
                    diff_list.append(diff6)
                    test_rmse_list.append(test_rmse6)
                    var6 = forecast0.var()
                    var_list.append(var6)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts6.append(future['pred'])
                    rmse6.append([np.nan, np.nan])
                    diff6 = np.inf
                    diff_list.append(diff6)
                    test_rmse_list.append(np.inf)
                    print("error in prophet for {}:{}".format(part, e))

                try:

                    # mlp
                    model7 = Model()
                    forecast7, train_rmse7, test_rmse7 = model7.mlp(data=part_data, start=start, train_frac=frac,
                                                                    n_future=n_future, lags=lags)
                    forecasts7.append(forecast7)
                    rmse7.append([train_rmse7, test_rmse7])
                    diff7 = abs(train_rmse7 - test_rmse7)
                    diff_list.append(diff7)
                    test_rmse_list.append(test_rmse7)
                    var7 = forecast7.var()
                    var_list.append(var7)

                except Exception as e:
                    future = pd.DataFrame(pd.date_range(start_, periods=n_future, freq="M"), columns=['ds'])
                    future['ds'] = future['ds'].apply(lambda dt: dt.replace(day=1))
                    future['pred'] = [np.nan] * n_future
                    future.index = future['ds']

                    forecasts7.append(future['pred'])
                    rmse7.append([np.nan, np.nan])
                    diff7 = np.inf
                    diff_list.append(diff7)
                    test_rmse_list.append(np.inf)
                    print("error in mlp for {}:{}".format(part, e))

                # test_rmse_rank = np.argsort(test_rmse_list)
                test_rmse_rank = np.argsort(var_list)[-3:]
                candidates = test_rmse_rank[:]
                s = candidates[0]
                min_diff = np.inf
                best_forecast = vars()['forecast' + str(s)]
                min_test_rmse = vars()['test_rmse' + str(s)]
                model_name = model_dict[s]
                for c in candidates:
                    diff = vars()["diff" + str(c)]
                    variance = vars()['var' + str(c)]
                    rmse = vars()['test_rmse' + str(c)]
                    if diff < min_diff and rmse < min_test_rmse:
                        min_diff = diff
                        model_name = model_dict[c]
                        min_test_rmse = vars()["test_rmse" + str(c)]
                        best_forecast = vars()["forecast" + str(c)]

                final_result.append((part, model_name, best_forecast, min_test_rmse))

        # Exporting results to an excel file. Output file will be named as: "Forecast [Current timestamp]"
        best_forecast = [k for i, j, k, l in final_result]  # part, model_name, best_forecast,min_test_rmse
        best_forecast = [series.round(0).astype(int) for series in best_forecast]
        best_model = [j for i, j, k, l in final_result]
        part_no = [i for i, j, k, l in final_result]
        test_rmse = [l for i, j, k, l in final_result]
        best_forecast_df = pd.concat(best_forecast, axis=1).T
        best_forecast_df.index = part_no
        best_models_df = pd.DataFrame(best_model, columns=['best model'], index=part_no)
        result = pd.merge(best_models_df, best_forecast_df, left_index=True, right_index=True)
        result['test rmse'] = test_rmse
        result.index.names = ['part']
        @st.cache_data
        def convert_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_to_csv(result)
        if st.button("See the output"):
            st.write(result)
        if st.button("Download the output"):
            download1 = st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv')
        # time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            # name = "Forecast_{}.xlsx".format(time_stamp)
            # buffer = io.BytesIO()
            # with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            #     result.to_excel(writer, sheet_name="Sheet1", index=False)
            #     writer.save()
            #     download2 = st.download_button(
            #         label="Download result",
            #         data=buffer,
            #         file_name=name,
            #         mime="application/vnd.ms-excel")
            #
            #
