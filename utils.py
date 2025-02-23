import plotly.express as px 

def plot_forecast(forecast):
    fig = px.line(
        forecast,
        x='ds',
        y='y',
        color='set',
        hover_data={'ds': False, 'set': False}
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='forecast',
        mode='lines'
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        name='lower bound',
        mode='lines',
        line=dict(dash='dash'),
        marker=dict(color='gray')
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        name='upper bound',
        mode='lines',
        line=dict(dash='dash'),
        marker=dict(color='gray')
    )
    fig.update_layout(hovermode='x')
    return fig 
