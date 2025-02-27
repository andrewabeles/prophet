import streamlit as st 
import pandas as pd
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from utils import plot_forecast, plot_errors, plot_errors_dist

st.set_page_config(
    page_title="Prophet",
    page_icon=':chart_with_upwards_trend:',
    layout='wide'
)

st.title("Prophet")
with st.expander("About"):
    st.markdown("""
        This app aims to make time series forecasting more accessible. It uses Facebook's [Prophet forecasting procedure](https://facebook.github.io/prophet/) 
        to model time series data uploaded by the user. The model parameters can be adjusted in the sidebar. 
    """)

countries_df = pd.read_csv('countries.csv', sep='\t')

with st.sidebar:
    with st.expander("Data"):
        st.session_state.uploaded_file = st.file_uploader("Upload time series CSV file")
        if st.session_state.uploaded_file is not None:
            if 'df_raw' not in st.session_state:
                st.session_state.df_raw = pd.read_csv(st.session_state.uploaded_file)

            # select columns 
            ds_colname = st.selectbox(
                "Select datestamp (ds) column",
                st.session_state.df_raw.select_dtypes(include=[object, 'datetime', 'datetimetz']).columns
            )
            y_colname = st.selectbox(
                "Select values (y) column",
                st.session_state.df_raw.select_dtypes(include='number').columns
            )
            if 'df' not in st.session_state:
                df = st.session_state.df_raw[[ds_colname, y_colname]].copy()
            df.rename(columns={ds_colname: 'ds', y_colname: 'y'}, inplace=True)
            df['ds'] = pd.to_datetime(df['ds'], utc=True)
            df['ds'] = df['ds'].dt.tz_localize(None)

            # split data 
            val_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="""The proportion of observations that will be reserved for model evaluation."""
            )
            # split data
            df.sort_values('ds', ascending=True, inplace=True) 
            df_train, df_test = train_test_split(df, test_size=val_size, shuffle=False)
            df_train['set'] = 'train'
            df_test['set'] = 'test'
            df_split = pd.concat([df_train, df_test])
    with st.expander("Trend Parameters"):
        if st.session_state.uploaded_file is not None:
            growth = st.radio(
                "Growth", 
                ['linear', 'logistic'],
                help="""Select 'logistic' if you expect growth to saturate at a given maximum or minimum e.g. total addressable market."""
            )
            if growth == 'logistic':
                cap = st.number_input("Maximum", value=2 * df_train['y'].max())
                floor = st.number_input("Minimum", value=0)
                df['cap'] = cap 
                df['floor'] = floor 
                df_train['cap'] = cap 
                df_train['floor'] = floor 
                df_test['cap'] = cap 
                df_test['floor'] = floor
            changepoint_selection = st.radio("Changepoint Selection", ['automatic', 'manual'])
            if changepoint_selection == 'automatic':
                changepoint_range = st.slider("Range of Trend Changepoints", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
                changepoints=None
            elif changepoint_selection == 'manual':
                changepoint_range = 0.8
                changepoints = st.multiselect("Changepoints", df_train['ds'].unique())
            changepoint_prior_scale = st.selectbox("Changepoint Prior Strength", options=[0.005, 0.05, 0.5, 5, 50], index=1)

    with st.expander("Holiday Parameters"):
        if st.session_state.uploaded_file is not None:
            include_country_holidays = st.checkbox("Include Country Holidays")
            if include_country_holidays:
                country = st.selectbox("Select Country", countries_df['country'], index=149)
                country_code = countries_df[countries_df['country'] == country]['code'].values[0]
                holidays_prior_scale = st.selectbox("Holidays Prior Strength", options=[0.001, 0.01, 0.1, 1, 10, 100], index=4)

if st.session_state.uploaded_file is not None:
    split_fig = px.line(
        df_split,
        x='ds',
        y='y',
        color='set'
    )
    st.header("Data")
    st.plotly_chart(split_fig)   

    # define and fit Prophet model 
    m = Prophet(
        growth=growth,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoints=changepoints
    )
    if include_country_holidays:
        m.add_country_holidays(country_name=country_code)
        m.holidays_prior_scale = holidays_prior_scale
    m.fit(df_train)
    forecast = m.predict(df)

    # model components 
    st.header("Model")
    components_fig = plot_components_plotly(m, forecast)
    st.plotly_chart(components_fig)

    # forecast on test 
    df_merged = df_split.merge(
        forecast,
        on='ds',
        how='left'
    )
    show_changepoints = True
    forecast_fig = plot_forecast(m, df_merged, show_sets=True, show_changepoints=show_changepoints)
    st.plotly_chart(forecast_fig)

    # evaluation 
    df_merged['error'] = df_merged['y'] - df_merged['yhat']
    y_true_train = df_merged.query("set == 'train'")['y']
    y_pred_train = df_merged.query("set == 'train'")['yhat']
    y_true_test = df_merged.query("set == 'test'")['y']
    y_pred_test = df_merged.query("set == 'test'")['yhat']
    rmse_train = root_mean_squared_error(y_true_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_true_train, y_pred_train)
    rmse_test = root_mean_squared_error(y_true_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_true_test, y_pred_test)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train Metrics")
    with col2:
        st.subheader("Test Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Root Mean Squared Error', value=f'{round(rmse_train, 2)}')
    with col2:
        st.metric(label='Mean Absolute Percentage Error', value=f'{round(100 * mape_train, 2)}%')
    with col3:
        st.metric(label='Root Mean Squared Error', value=f'{round(rmse_test, 2)}')
    with col4:
        st.metric(label='Mean Absolute Percentage Error', value=f'{round(100 * mape_test, 2)}%')
    col1, col2 = st.columns(2)
    with col1:
        train_errors_fig = plot_errors(df_merged, set='train')
        train_errors_dist_fig = plot_errors_dist(df_merged, set='train')
        st.plotly_chart(train_errors_fig)
        st.plotly_chart(train_errors_dist_fig)
    with col2:
        test_errors_fig = plot_errors(df_merged, set='test')
        test_errors_dist_fig = plot_errors_dist(df_merged, set='test')
        st.plotly_chart(test_errors_fig)
        st.plotly_chart(test_errors_dist_fig)

    st.header("Forecast")
    forecast_periods = st.number_input("Forecast Periods", min_value=1, value=365)
    m_fcst = Prophet(
        growth=growth,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoints=changepoints
    )
    if include_country_holidays:
        m_fcst.add_country_holidays(country_name=country_code)
        m_fcst.holidays_prior_scale = holidays_prior_scale
    m_fcst.fit(df)
    future = m_fcst.make_future_dataframe(periods=forecast_periods)
    if growth == 'logistic':
        future['cap'] = cap 
        future['floor'] = floor
    future_forecast = m_fcst.predict(future)
    future_forecast['y'] = df['y'].copy()
    future_forecast['set'] = 'actual'
    future_forecast_fig = plot_forecast(m_fcst, future_forecast, show_sets=False, show_changepoints=False)
    st.plotly_chart(future_forecast_fig)
