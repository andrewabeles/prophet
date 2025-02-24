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

with st.sidebar:
    with st.expander("Data"):
        uploaded_file = st.file_uploader("Upload time series CSV file")
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)

            # select columns 
            ds_colname = st.selectbox(
                "Select datestamp (ds) column",
                df_raw.select_dtypes(include=[object, 'datetime', 'datetimetz']).columns
            )
            y_colname = st.selectbox(
                "Select values (y) column",
                df_raw.select_dtypes(include='number').columns
            )
            df = df_raw[[ds_colname, y_colname]].copy()
            df.rename(columns={ds_colname: 'ds', y_colname: 'y'}, inplace=True)
            df['ds'] = pd.to_datetime(df['ds'], utc=True)
            df['ds'] = df['ds'].dt.tz_localize(None)

            # split data 
            val_size = st.slider(
                "Select Validation Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="""The proportion of observations that will be used for model selection.
                Remember to retain a separate hidden test set for final model evaluation."""
            )
            # split data
            df.sort_values('ds', ascending=True, inplace=True) 
            df_train, df_val = train_test_split(df, test_size=val_size, shuffle=False)
            df_train['set'] = 'train'
            df_val['set'] = 'validation'
            df_split = pd.concat([df_train, df_val])
    with st.expander("Trend Parameters"):
        if uploaded_file is not None:
            growth = st.radio("Trend Type", ['linear', 'logistic'])
            if growth == 'logistic':
                cap = st.number_input("Maximum", value=2 * df_train['y'].max())
                floor = st.number_input("Minimum", value=0)
                df['cap'] = cap 
                df['floor'] = floor 
                df_train['cap'] = cap 
                df_train['floor'] = floor 
                df_val['cap'] = cap 
                df_val['floor'] = floor
            changepoint_selection = st.radio("Changepoint Selection", ['automatic', 'manual'])
            if changepoint_selection == 'automatic':
                n_changepoints = st.number_input("Number of Trend Changepoints", min_value=0, max_value=50, value=25)
                changepoint_range = st.slider("Range of Trend Changepoints", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
                changepoints=None
            elif changepoint_selection == 'manual':
                n_changepoints = None
                changepoint_range = 0.8
                changepoints = st.multiselect("Changepoints", df_train['ds'].unique())
            changepoint_prior_scale = st.slider("Changepoint Prior Strength", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

if uploaded_file is not None:
    split_fig = px.line(
        df_split,
        x='ds',
        y='y',
        color='set'
    )
    st.header("Split Data")
    st.plotly_chart(split_fig)   

    # define and fit Prophet model 
    m = Prophet(
        growth=growth,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoints=changepoints
    )
    m.fit(df_train)
    forecast = m.predict(df)

    # model components 
    st.header("Model Components")
    components_fig = plot_components_plotly(m, forecast)
    st.plotly_chart(components_fig)

    # forecast 
    df_merged = df_split.merge(
        forecast,
        on='ds',
        how='left'
    )
    st.header("Forecast")
    show_changepoints = True
    forecast_fig = plot_forecast(m, df_merged, show_changepoints=show_changepoints)
    st.plotly_chart(forecast_fig)

    # evaluation 
    df_merged['error'] = df_merged['y'] - df_merged['yhat']
    y_true = df_merged.query("set == 'validation'")['y']
    y_pred = df_merged.query("set == 'validation'")['yhat']
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='Root Mean Sqared Error', value=f'{round(rmse, 2)}')
        errors_fig = plot_errors(df_merged.query("set == 'validation'"))
        st.plotly_chart(errors_fig)
    with col2:
        st.metric(label='Mean Absolute Percentage Error', value=f'{round(100 * mape, 2)}%')
        errors_dist_fig = plot_errors_dist(df_merged.query("set == 'validation'"))
        st.plotly_chart(errors_dist_fig)
