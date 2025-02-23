import streamlit as st 
import pandas as pd
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from utils import plot_forecast

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

if uploaded_file is not None:
    # split data
    df.sort_values('ds', ascending=True, inplace=True) 
    df_train, df_val = train_test_split(df, test_size=val_size, shuffle=False)
    df_train['set'] = 'train'
    df_val['set'] = 'validation'
    df_split = pd.concat([df_train, df_val])
    split_fig = px.line(
        df_split,
        x='ds',
        y='y',
        color='set'
    )
    st.header("Split Data")
    st.plotly_chart(split_fig)   

    # define and fit Prophet model 
    m = Prophet()
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
    forecast_fig = plot_forecast(df_merged)
    st.header("Forecast")
    st.plotly_chart(forecast_fig)

    # evaluation 
    y_true = df_merged.query("set == 'validation'")['y']
    y_pred = df_merged.query("set == 'validation'")['yhat']
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='Root Mean Sqared Error', value=f'{round(rmse, 2)}')
    with col2:
        st.metric(label='Mean Absolute Percentage Error', value=f'{round(100 * mape, 2)}%')
