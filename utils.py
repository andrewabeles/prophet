import plotly.express as px 

def plot_forecast(model, forecast, show_changepoints=False):
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
    if show_changepoints:
        for cp, delta in zip(model.changepoints, model.params['delta'][0]):
            if abs(delta) > 0.1:
                fig.add_vline(x=cp, line_dash="dash", line_color="red", opacity=0.6)
    fig.update_layout(hovermode='x')
    return fig 

def plot_errors(df, set='train'):
    fig = px.bar(
        df.query("set == @set"),
        x='ds',
        y='error',
        title=f'Forecast Errors ({set})'
    )
    return fig 

def plot_errors_dist(df, set='train'):
    fig = px.histogram(
        df.query("set == @set"),
        x='error',
        title=f'Error Distribution ({set})'
    )
    return fig 
