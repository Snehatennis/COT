import pandas as pd
from sodapy import Socrata
import dash
from dash import html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pandas import Timedelta
from dash import dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq

from flask import Flask
from dash import Dash
import os

server = Flask(__name__)

import sodapy

client = sodapy.Socrata("publicreporting.cftc.gov", None)
results = client.get("6dca-aqww", limit = 20000)

#offset = 0
#while True:
#    results = client.get("6dca-aqww", limit=10000, offset=offset)
#    if len(results) == 0:
#        break

#    offset += 10000
    
#client = Socrata("publicreporting.cftc.gov", None)

#results = client.get("6dca-aqww", limit = 1000000)

df = pd.DataFrame.from_records(results)

options = ["CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
"BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
"USD INDEX - ICE FUTURES U.S.", "EURO FX - CHICAGO MERCANTILE EXCHANGE",
"BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE", "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
"RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE", "EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE", "U.S. DOLLAR INDEX - ICE FUTURES U.S."]

df = df.sort_values('report_date_as_yyyy_mm_dd', ascending=False)

df1 = df.loc[df['market_and_exchange_names'].isin(options)]

df1 = df1.reset_index(drop = True)

df2 = df1[['market_and_exchange_names', 'report_date_as_yyyy_mm_dd', 'open_interest_all', 'noncomm_positions_long_all',
           'noncomm_positions_short_all', 'change_in_noncomm_long_all', 'change_in_noncomm_short_all',
          'pct_of_oi_noncomm_long_all', 'pct_of_oi_noncomm_short_all', 'comm_positions_long_all',
          'comm_positions_short_all', 'change_in_comm_long_all', 'change_in_comm_short_all',
          'pct_of_oi_comm_long_all', 'pct_of_oi_comm_short_all', 'nonrept_positions_long_all',
          'nonrept_positions_short_all', 'change_in_nonrept_long_all', 'change_in_nonrept_short_all',
          'pct_of_oi_nonrept_long_all', 'pct_of_oi_nonrept_short_all']]

df2 = pd.DataFrame(df2)

for col in  df2.columns[2:]:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
#Non Commercial
df2['Non Commercial Long %'] = (df2['noncomm_positions_long_all']/ (df2['noncomm_positions_long_all']+
                                                                           df2['noncomm_positions_short_all']))

df2['Non Commercial Short %'] = (df2['noncomm_positions_short_all']/ (df2['noncomm_positions_long_all']+
                                                                           df2['noncomm_positions_short_all']))

df2['Non Commercial Net Position'] = (df2['noncomm_positions_long_all'] - df2['noncomm_positions_short_all'])


#Commercial
df2['Commercial Long %'] = (df2['comm_positions_long_all']/ (df2['comm_positions_long_all']+
                                                                           df2['comm_positions_short_all']))

df2['Commercial Short %'] = (df2['comm_positions_short_all']/ (df2['comm_positions_long_all']+
                                                                           df2['comm_positions_short_all']))
df2['Commercial Net Position'] = (df2['comm_positions_long_all'] - df2['comm_positions_short_all'])


#Non Reportable
df2['Non Reportable Long %'] = (df2['nonrept_positions_long_all']/ (df2['nonrept_positions_long_all']+
                                                                           df2['nonrept_positions_short_all']))

df2['Non Reportable Short %'] = (df2['nonrept_positions_short_all']/ (df2['nonrept_positions_long_all']+
                                                                           df2['nonrept_positions_short_all']))
df2['Non Reportable Net Position'] = (df2['nonrept_positions_long_all'] - df2['nonrept_positions_short_all'])

df3 = df2.sort_values(by='report_date_as_yyyy_mm_dd', ascending=False).reset_index(drop=True)

def convert_dates(column):
    
    """ convert the df's date column into a datetime format 
            for sorting and concatentation purposes"""
    
    column = column.apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))
    column = pd.to_datetime(column, format='%d/%m/%Y')
    
    return column

df2['report_date_as_yyyy_mm_dd'] = convert_dates(df2['report_date_as_yyyy_mm_dd'])  
df3['Date'] = convert_dates(df3['report_date_as_yyyy_mm_dd'])

#Sort both dfs by date for chronological purposes
df3 = df3.sort_values(by='Date').reset_index(drop=True)
df2 = df2.sort_values(by='report_date_as_yyyy_mm_dd').reset_index(drop=True)

#rename to 'Date' for concatenation
df2.rename(columns = {'report_date_as_yyyy_mm_dd':'Date'}, inplace = True)

df = pd.DataFrame(df2)

df['market_and_exchange_names'] = df['market_and_exchange_names'].replace({
    'USD INDEX - ICE FUTURES U.S.': 'USD INDEX',
    'U.S. DOLLAR INDEX - ICE FUTURES U.S.': 'USD INDEX',
    'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE' : 'SWISS FRANC',
    'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE' : 'JAPANESE YEN',
    'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'CANADIAN DOLLAR',
    'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'AUSTRALIAN DOLLAR',
    'EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE' : 'EURO FX/BRITISH POUND XRATE',
    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE' : 'E-MINI S&P 500',
    'RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE' : 'RUSSEL E-MINI',
    'NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'NZ DOLLAR',
    'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE' : 'BRITISH POUND',
    'EURO FX - CHICAGO MERCANTILE EXCHANGE' : 'EURO',
    'BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE' : 'BRAZILLIAN REAL'
})

#Central Bank Rates
import requests

api_url = 'https://api.api-ninjas.com/v1/interestrate'
response = requests.get(api_url, headers={'X-Api-Key': 'WxqvE2/mfm9rPx67/g0MIQ==GUnyWlARArflJMjn'}).json()
data=pd.DataFrame(response['central_bank_rates'])

#Inflation Rates
import requests
import pandas as pd

# List of countries
countries = ['United States', 'Australia', 'Chile', 'South Korea', 'Brazil',
       'Great Britain', 'Canada', 'China', 'Czech Republic', 'Denmark',
       'Europe', 'Hungary', 'India', 'Indonesia', 'Israel', 'Mexico',
       'New Zealand', 'Norway', 'Poland', 'Russia', 'Saudi Arabia',
       'South Africa', 'Sweden', 'Switzerland', 'Turkey']

# Initialize an empty list to store data for each country
data_list = []

# Loop through the list of countries and make API requests
for country in countries:
    api_url = 'https://api.api-ninjas.com/v1/inflation?country={}'.format(country)
    response = requests.get(api_url, headers={'X-Api-Key': 'WxqvE2/mfm9rPx67/g0MIQ==GUnyWlARArflJMjn'})
    
    if response.status_code == 200:
        # Convert the JSON response to a Python dictionary
        data_dict = response.json()
        
        # Append the data for the current country to the data_list
        data_list.extend(data_dict)
    else:
        print("API request for", country, "failed with status code:", response.status_code)

# Create a DataFrame from the combined data
inflation_df = pd.DataFrame(data_list)

# Print the DataFrame
print(inflation_df)

#GDP data:
import requests
import pandas as pd

# List of countries
countries = ['United States', 'Australia', 'Chile', 'South Korea', 'Brazil',
             'Great Britain', 'Canada', 'China', 'Czech Republic', 'Denmark',
             'Europe', 'Hungary', 'India', 'Indonesia', 'Israel', 'Mexico',
             'New Zealand', 'Norway', 'Poland', 'Russia', 'Saudi Arabia',
             'South Africa', 'Sweden', 'Switzerland', 'Turkey']

# Initialize an empty list to store data frames
data_frames = []

# API key
api_key = 'WxqvE2/mfm9rPx67/g0MIQ==GUnyWlARArflJMjn'

# Loop through the list of countries and make API requests
for country in countries:
    api_url = f'https://api.api-ninjas.com/v1/country?name={country}'
    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    
    if response.status_code == requests.codes.ok:
        data_dict = response.json()
        
        data_frame = pd.DataFrame(data_dict)
        data_frames.append(data_frame)
    
    else:
        print("API request for", country, "failed with status code:", response.status_code)

# Create a DataFrame from the combined data
country_df = pd.concat(data_frames)

# Print the DataFrame
#print(country_df)

country_df = country_df[['name', 'gdp', 'unemployment']]
country_df.rename(columns={'name': 'country'}, inplace=True)

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
#from pandas.tseries.offsets import Timedelta
import numpy as np
import json

# Create a dark theme provider
theme = {
    'color': {
        'primary': '#000000',
        'secondary': '#ffffff',
        'text': '#333333',
        'background': '#000000'
    }
}

# Create the app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUX])

# Sidebar style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 50,
    "left": 0,
    "bottom": 0,
    "width": "0.5rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
#        html.H2("Market Indicators", style={'font-size': '16px'}),  # Adjust font-size here
        html.Hr(),
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="COT Individual", value="tab-1", style={'font-size': '10px'}),  # Adjust font-size here
                dcc.Tab(label="Combined COT chart", value="tab-2", style={'font-size': '10px'}),  # Adjust font-size here
                dcc.Tab(label="Market Data Table", value="tab-3", style={'font-size': '10px'}),  # Adjust font-size here
                dcc.Tab(label="Central Bank Rates", value='tab-4',style={'font-size': '10px'}),
                dcc.Tab(label="Inflation Rates", value='tab-5', style={'font-size': '10px'}),
                dcc.Tab(label="GDP data", value='tab-6', style={'font-size': '10px'})
            ],
            vertical=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

app.layout = dbc.Container([
    html.H1('COT Dashboard', style={'color': theme['color']['primary']}),
    dbc.Row([
        dbc.Col(sidebar, width=2, style={'padding-right': '0'}),
        dbc.Col(id='tabs-content', width=10),
    ],
    style={'margin-left': '0.2cm', 'margin-right': '0.2cm'}),  # Add margin to both sides
],
fluid=True)  # Set fluid to True to make the container fill the entire width of the viewport

# Add some additional styling to the tabs content
app.clientside_callback(
    """
    function resizeContent() {
        var sidebarWidth = document.querySelector('#sidebar').clientWidth;
        var content = document.querySelector('#tabs-content');
        content.style.marginLeft = sidebarWidth + 'px';
    }
    resizeContent();
    window.addEventListener('resize', resizeContent);
    """,
    Output('tabs-content', 'style'),
    Input('tabs', 'value'),
)



@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.DatePickerRange(id='date-range-picker', start_date=df['Date'].min(), end_date=df['Date'].max(), style={'color': theme['color']['text']}),
            dcc.Dropdown(id='category-dropdown', options=[
                {'label': 'Non Commercial', 'value': 'Non Commercial'}, # 'style': {'color': theme['color']['text']}},
                {'label': 'Commercial', 'value': 'Commercial'}, # 'style': {'color': theme['color']['text']}},
                {'label': 'Non Reportable', 'value': 'Non Reportable'}, # 'style': {'color': theme['color']['text']}}
            ], style={'color': theme['color']['text']}),
            dcc.Dropdown(id='market-dropdown', 
            options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()],
            multi=True, style={'color': theme['color']['text']}),
            html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dcc.Graph(id='graph-1', style={'color': theme['color']['text']})
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.DatePickerRange(id='date-range-picker-2', start_date=df['Date'].min(), end_date=df['Date'].max(), style={'color': theme['color']['text']}),
            dcc.Dropdown(id='market-dropdown-2', options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()]
            , multi=True, style={'color': theme['color']['text']}),
            html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dcc.Graph(id='graph-2', style={'color': theme['color']['text']})
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.DatePickerSingle(id='date-picker', date=df['Date'].max(), style={'color': theme['color']['text']}),
            dcc.Dropdown(id='category-dropdown', options=[
                {'label': 'Non Commercial', 'value': 'Non Commercial'}, # 'style': {'color': theme['color']['text']}},
                {'label': 'Commercial', 'value': 'Commercial'}, # 'style': {'color': theme['color']['text']}},
                {'label': 'Non Reportable', 'value': 'Non Reportable'},# 'style': {'color': theme['color']['text']}}
            ], style={'color': theme['color']['text']}),
            dcc.Dropdown(id='market-dropdown', 
            options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()],
            multi=True, style={'color': theme['color']['text']}),
            html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dash_table.DataTable(id='table-1', columns=[
                {'name': 'Market', 'id': 'Market'},
                {'name': 'Long Position', 'id': 'Long Position'},
                {'name': 'Short Position', 'id': 'Short Position'}
            ])
        ])
    elif tab == 'tab-4':
        return html.Div([
            dcc.Dropdown(id = 'country-dropdown', 
                         options = [{'label': country, 'value': country} for country in data['country'].unique()],
            multi=True, style={'color': theme['color']['text']}),
             html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dash_table.DataTable(id='table-2', columns=[
                {'name': 'Country', 'id': 'country'},
                {'name': 'Rate', 'id': 'rate_pct'},
                {'name': 'Date', 'id': 'last_updated'}],
                data=json.loads(data.to_json(orient='records', date_format='iso')))
            
        ])
    elif tab == 'tab-5':
        return html.Div([
            dcc.Dropdown(id = 'country-dropdown', 
                         options = [{'label': country, 'value': country} for country in inflation_df['country'].unique()],
            multi=True, style={'color': theme['color']['text']}),
             html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dash_table.DataTable(id='table-3', columns=[
                {'name': 'Country', 'id': 'country'},
                {'name': 'Type', 'id': 'type'},
                {'name': 'Month', 'id': 'period'},
                {'name': 'Monthly Rate', 'id': 'monthly_rate_pct'},
                {'name': 'Yearly Rate', 'id': 'yearly_rate_pct'}],
                data=json.loads(data.to_json(orient='records', date_format='iso')))
            
        ])
    elif tab == 'tab-6':
        return html.Div([
            dcc.Dropdown(id = 'country-dropdown', 
                         options = [{'label': country, 'value': country} for country in data['country'].unique()],
            multi=True, style={'color': theme['color']['text']}),
             html.Button(
                "Go",
                id="filter-button",
                className="btn btn-primary",
                style={'background-color': 'green', 'color': theme['color']['secondary']}
            ),
            dash_table.DataTable(id='table-4', columns=[
                {'name': 'Country', 'id': 'country'},
                {'name': 'GDP', 'id': 'gdp'},
                {'name': 'Unemployment', 'id': 'unemployment'}],
                data=json.loads(data.to_json(orient='records', date_format='iso')))
            
        ])

    # Add more tabs for other graphs as needed
@app.callback(
    Output('graph-1', 'figure'),
    [Input('filter-button', 'n_clicks')],
    [State('date-range-picker', 'start_date'),
     State('date-range-picker', 'end_date'),
     State('category-dropdown', 'value'),
     State('market-dropdown', 'value')]
)

def update_graph_1(n_clicks, start_date, end_date, category, markets):
    if n_clicks is None:
        # No button click yet, return an empty figure
        return {}

    fig = go.Figure()

    colors = ['blue', 'red', 'green']  # Define colors for each market value

    for i, market in enumerate(markets):
        # Filter the DataFrame for the current market
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'] == market)]

        if category == 'Non Commercial':
            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Commercial Net Position'], name=f'{market} - Non Commercial', marker_color=colors[i]))
        elif category == 'Commercial':
            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Commercial Net Position'], name=f'{market} - Commercial', marker_color=colors[i]))
        elif category == 'Non Reportable':
            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Reportable Net Position'], name=f'{market} - Non Reportable', marker_color=colors[i]))

    fig.update_layout(
        yaxis=dict(title='Net Position'),
        xaxis=dict(title='Date'),
        barmode='group',  # Use 'group' to place the bars side by side
        bargap=0.1,  # Adjust this value to reduce the space between bars
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig
@app.callback(
    Output('graph-2', 'figure'),
    [Input('filter-button', 'n_clicks')],
    [State('date-range-picker-2', 'start_date'),
     State('date-range-picker-2', 'end_date'),
     State('market-dropdown-2', 'value')]
)

def update_graph_2(n_clicks, start_date, end_date, markets):
    if n_clicks is None:
        # No button click yet, return an empty figure
        return {}

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'].isin(markets))]

    fig = go.Figure()

    # Calculate the offset for each market bar to make them side by side
    n_markets = len(markets)
    bar_width = 0.2  # Adjust this value to change the width of the bars
    bar_offsets = np.linspace(-bar_width * (n_markets - 1) / 2, bar_width * (n_markets - 1) / 2, n_markets)

    colors = ['blue', 'red', 'green']

    for i, market in enumerate(markets):
        x_offset = Timedelta(bar_offsets[i], unit='D')
        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Commercial Net Position'], name=f'{market} - Non Commercial', marker_color=colors[0]))
        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Commercial Net Position'], name=f'{market} - Commercial', marker_color=colors[1]))
        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Reportable Net Position'], name=f'{market} - Non Reportable', marker_color=colors[2]))

    fig.update_layout(
        yaxis=dict(title='Net Position'),
        xaxis=dict(title='Date'),
        barmode='group',  # Use 'group' to place the bars side by side
        bargap=0.1,  # Adjust this value to reduce the space between bars
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig
@app.callback(
    Output('table-1', 'data'),
    [Input('filter-button', 'n_clicks')],
    [State('date-picker', 'date'),
     State('category-dropdown', 'value'),
     State('market-dropdown', 'value')]
)

def update_table(n_clicks, selected_date, category, markets):
    if n_clicks is None or not all([selected_date, category, markets]):
        # No button click or missing inputs, return an empty table
        return []

    # Filter the data based on the selected date and markets
    filtered_df = df[(df['Date'] == selected_date) & (df['market_and_exchange_names'].isin(markets))]

    # Create a new list to hold the table data for each market
    table_data = []

    # Loop through each market and append its data to the table_data list
    for market in markets:
        # Filter the data for the specific market
        market_data = filtered_df[filtered_df['market_and_exchange_names'] == market]

        # Check if the filtered DataFrame for the market is empty
        if market_data.empty:
            # Handle the case when the filtered DataFrame is empty
            long_position = 0
            short_position = 0
        else:
            # Get the long and short position values based on the selected category
            if category == 'Non Commercial':
                long_position = round((float(market_data['Non Commercial Long %'].values[0])*100),2)
                short_position = round((float(market_data['Non Commercial Short %'].values[0])*100),2)
            elif category == 'Commercial':
                long_position = round((float(market_data['Commercial Long %'].values[0])*100),2)
                short_position = round((float(market_data['Commercial Short %'].values[0])*100),2)
            elif category == 'Non Reportable':
                long_position = round((float(market_data['Non Reportable Long %'].values[0])*100),2)
                short_position = round((float(market_data['Non Reportable Short %'].values[0])*100),2)
            else:
                # Handle the case when no category is selected
                long_position = 0
                short_position = 0

        # Convert the long and short position values to strings
        long_position_str = str(long_position)
        short_position_str = str(short_position)

        # Convert the market value to a string
        market_str = str(market)  # Ensure market_str is a string

        # Create a dictionary for the table row and append it to table_data list
        table_data.append({'Market': market_str, 'Long Position': long_position_str, 'Short Position': short_position_str})

    return table_data

@app.callback(
    Output('table-2', 'data'),
    [Input('filter-button', 'n_clicks')],
    [State('country-dropdown', 'value')]
)
def update_table_2(n_clicks, country):
    if n_clicks is None or not country:
        return []  # Return an empty data set when no country is selected or on initial load
    
    filtered_data = data[data['country'].isin(country)].to_dict('records')

    return filtered_data

@app.callback(
    Output('table-3', 'data'),
    [Input('filter-button', 'n_clicks')],
    [State('country-dropdown', 'value')]
)
def update_table_3(n_clicks, country):
    if n_clicks is None or not country:
        return []  # Return an empty data set when no country is selected or on initial load
    
    filtered_data1 = inflation_df[inflation_df['country'].isin(country)].to_dict('records')

    return filtered_data1

@app.callback(
    Output('table-4', 'data'),
    [Input('filter-button', 'n_clicks')],
    [State('country-dropdown', 'value')]
)
def update_table_4(n_clicks, country):
    if n_clicks is None or not country:
        return []  # Return an empty data set when no country is selected or on initial load
    
    filtered_data2 = country_df[country_df['country'].isin(country)].to_dict('records')

    return filtered_data2


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 4545)))

    