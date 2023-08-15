{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "34d8c879-93ef-4a9c-a383-81ff1051a315",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/2678637838.py:4: UserWarning: \n",
      "The dash_html_components package is deprecated. Please replace\n",
      "`import dash_html_components as html` with `from dash import html`\n",
      "  import dash_html_components as html\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/2678637838.py:5: UserWarning: \n",
      "The dash_core_components package is deprecated. Please replace\n",
      "`import dash_core_components as dcc` with `from dash import dcc`\n",
      "  import dash_core_components as dcc\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sodapy import Socrata\n",
    "import dash\n",
    "import dash_html_components as html\n",
    "import dash_core_components as dcc\n",
    "from dash.dependencies import Input, Output, State\n",
    "import plotly.graph_objects as go\n",
    "import numpy as np\n",
    "from pandas import Timedelta\n",
    "from dash import dash_table\n",
    "#from app import server\n",
    "#from app import app\n",
    "#from layouts import layout_birst_category, layout_ga_category, layout_paid_search, noPage, layout_display, layout_publishing, layout_metasearch\n",
    "#import callbacks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c28e0ac0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:root:Requests made without an app_token will be subject to strict throttling limits.\n"
     ]
    }
   ],
   "source": [
    "# Unauthenticated client only works with public data sets. Note 'None'\n",
    "# in place of application token, and no username or password:\n",
    "client = Socrata(\"publicreporting.cftc.gov\", None)\n",
    "\n",
    "# Example authenticated client (needed for non-public datasets):\n",
    "# client = Socrata(publicreporting.cftc.gov,\n",
    "#                  MyAppToken,\n",
    "#                  username=\"user@example.com\",\n",
    "#                  password=\"AFakePassword\")\n",
    "\n",
    "# First 2000 results, returned as JSON from API / converted to Python list of\n",
    "# dictionaries by sodapy.\n",
    "results = client.get(\"6dca-aqww\", limit = 1000000)\n",
    "\n",
    "# Convert to pandas DataFrame\n",
    "df = pd.DataFrame.from_records(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5c7fdee9",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "31eaa674",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "options = [\"CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE\", \"SWISS FRANC - CHICAGO MERCANTILE EXCHANGE\",\n",
    "\"BRITISH POUND - CHICAGO MERCANTILE EXCHANGE\", \"JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE\",\n",
    "\"USD INDEX - ICE FUTURES U.S.\", \"EURO FX - CHICAGO MERCANTILE EXCHANGE\",\n",
    "\"BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE\", \"NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE\",\n",
    "\"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE\", \"AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE\",\n",
    "\"RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE\", \"EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE\", \"U.S. DOLLAR INDEX - ICE FUTURES U.S.\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2afad25-0e60-4952-81b8-828a161de620",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b178d55c-8495-4c60-8208-af3fd81ff4cc",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3b63f0a5-ba7d-42ff-89f2-48c3733de258",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = df.sort_values('report_date_as_yyyy_mm_dd', ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c97b68b8-a933-414a-9649-f5d8a4550421",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ef66e13d",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df1 = df.loc[df['market_and_exchange_names'].isin(options)]\n",
    "\n",
    "df1 = df1.reset_index(drop = True)\n",
    "\n",
    "#df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4d0495f6-736a-42f7-9908-bb898c61e802",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#df1['market_and_exchange_names'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f8ec4212",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#for col in df1:\n",
    "#    print(col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1369d5f1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df2 = df1[['market_and_exchange_names', 'report_date_as_yyyy_mm_dd', 'open_interest_all', 'noncomm_positions_long_all',\n",
    "           'noncomm_positions_short_all', 'change_in_noncomm_long_all', 'change_in_noncomm_short_all',\n",
    "          'pct_of_oi_noncomm_long_all', 'pct_of_oi_noncomm_short_all', 'comm_positions_long_all',\n",
    "          'comm_positions_short_all', 'change_in_comm_long_all', 'change_in_comm_short_all',\n",
    "          'pct_of_oi_comm_long_all', 'pct_of_oi_comm_short_all', 'nonrept_positions_long_all',\n",
    "          'nonrept_positions_short_all', 'change_in_nonrept_long_all', 'change_in_nonrept_short_all',\n",
    "          'pct_of_oi_nonrept_long_all', 'pct_of_oi_nonrept_short_all']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "40348619-02b5-4c6f-a341-6d4d2cf5e7b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/304265892.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2[col] = pd.to_numeric(df2[col], errors='coerce')\n"
     ]
    }
   ],
   "source": [
    "for col in  df2.columns[2:]:\n",
    "    df2[col] = pd.to_numeric(df2[col], errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2cb5a571",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Commercial Long %'] = (df2['noncomm_positions_long_all']/ (df2['noncomm_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:5: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Commercial Short %'] = (df2['noncomm_positions_short_all']/ (df2['noncomm_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:8: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Commercial Net Position'] = (df2['noncomm_positions_long_all'] - df2['noncomm_positions_short_all'])\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:12: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Commercial Long %'] = (df2['comm_positions_long_all']/ (df2['comm_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:15: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Commercial Short %'] = (df2['comm_positions_short_all']/ (df2['comm_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:17: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Commercial Net Position'] = (df2['comm_positions_long_all'] - df2['comm_positions_short_all'])\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:21: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Reportable Long %'] = (df2['nonrept_positions_long_all']/ (df2['nonrept_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:24: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Reportable Short %'] = (df2['nonrept_positions_short_all']/ (df2['nonrept_positions_long_all']+\n",
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/191335731.py:26: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Non Reportable Net Position'] = (df2['nonrept_positions_long_all'] - df2['nonrept_positions_short_all'])\n"
     ]
    }
   ],
   "source": [
    "#Non Commercial\n",
    "df2['Non Commercial Long %'] = (df2['noncomm_positions_long_all']/ (df2['noncomm_positions_long_all']+\n",
    "                                                                           df2['noncomm_positions_short_all']))\n",
    "\n",
    "df2['Non Commercial Short %'] = (df2['noncomm_positions_short_all']/ (df2['noncomm_positions_long_all']+\n",
    "                                                                           df2['noncomm_positions_short_all']))\n",
    "\n",
    "df2['Non Commercial Net Position'] = (df2['noncomm_positions_long_all'] - df2['noncomm_positions_short_all'])\n",
    "\n",
    "\n",
    "#Commercial\n",
    "df2['Commercial Long %'] = (df2['comm_positions_long_all']/ (df2['comm_positions_long_all']+\n",
    "                                                                           df2['comm_positions_short_all']))\n",
    "\n",
    "df2['Commercial Short %'] = (df2['comm_positions_short_all']/ (df2['comm_positions_long_all']+\n",
    "                                                                           df2['comm_positions_short_all']))\n",
    "df2['Commercial Net Position'] = (df2['comm_positions_long_all'] - df2['comm_positions_short_all'])\n",
    "\n",
    "\n",
    "#Non Reportable\n",
    "df2['Non Reportable Long %'] = (df2['nonrept_positions_long_all']/ (df2['nonrept_positions_long_all']+\n",
    "                                                                           df2['nonrept_positions_short_all']))\n",
    "\n",
    "df2['Non Reportable Short %'] = (df2['nonrept_positions_short_all']/ (df2['nonrept_positions_long_all']+\n",
    "                                                                           df2['nonrept_positions_short_all']))\n",
    "df2['Non Reportable Net Position'] = (df2['nonrept_positions_long_all'] - df2['nonrept_positions_short_all'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "26a0d361",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df3 = df2.sort_values(by='report_date_as_yyyy_mm_dd', ascending=False).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1f157b72",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#df3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fffaca54",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/_t/dvn8f3w52vn45l2jjvw6b6_c0000gn/T/ipykernel_7435/475635395.py:11: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['report_date_as_yyyy_mm_dd'] = convert_dates(df2['report_date_as_yyyy_mm_dd'])\n"
     ]
    }
   ],
   "source": [
    "def convert_dates(column):\n",
    "    \n",
    "    \"\"\" convert the df's date column into a datetime format \n",
    "            for sorting and concatentation purposes\"\"\"\n",
    "    \n",
    "    column = column.apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))\n",
    "    column = pd.to_datetime(column, format='%d/%m/%Y')\n",
    "    \n",
    "    return column\n",
    "\n",
    "df2['report_date_as_yyyy_mm_dd'] = convert_dates(df2['report_date_as_yyyy_mm_dd'])  \n",
    "df3['Date'] = convert_dates(df3['report_date_as_yyyy_mm_dd'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ea189c1f",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Sort both dfs by date for chronological purposes\n",
    "df3 = df3.sort_values(by='Date').reset_index(drop=True)\n",
    "df2 = df2.sort_values(by='report_date_as_yyyy_mm_dd').reset_index(drop=True)\n",
    "\n",
    "#rename to 'Date' for concatenation\n",
    "df2.rename(columns = {'report_date_as_yyyy_mm_dd':'Date'}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "7a732b39-f13f-4f0e-88e6-df7da37d99d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(df2)\n",
    "#df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "0c64e606-a6f7-48f4-9184-37ad135254c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df['market_and_exchange_names'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6fcbdad-50ea-4466-a588-803b92ce60f3",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "63e147e0-641a-4de2-a777-996b109d11c0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['market_and_exchange_names'] = df['market_and_exchange_names'].replace({\n",
    "    'USD INDEX - ICE FUTURES U.S.': 'USD INDEX',\n",
    "    'U.S. DOLLAR INDEX - ICE FUTURES U.S.': 'USD INDEX',\n",
    "    'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE' : 'SWISS FRANC',\n",
    "    'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE' : 'JAPANESE YEN',\n",
    "    'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'CANADIAN DOLLAR',\n",
    "    'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'AUSTRALIAN DOLLAR',\n",
    "    'EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE' : 'EURO FX/BRITISH POUND XRATE',\n",
    "    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE' : 'E-MINI S&P 500',\n",
    "    'RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE' : 'RUSSEL E-MINI',\n",
    "    'NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE' : 'NZ DOLLAR',\n",
    "    'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE' : 'BRITISH POUND',\n",
    "    'EURO FX - CHICAGO MERCANTILE EXCHANGE' : 'EURO',\n",
    "    'BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE' : 'BRAZILLIAN REAL'\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5731c594-6774-49b1-a694-7eb3e4bc51e8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#df['market_and_exchange_names'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebb24857-4321-4713-9c83-0385d729bd47",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "9ec48345-8c43-4e1d-9b60-2b6dfe04a98a",
   "metadata": {},
   "source": [
    "### Final Dashboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "61a773f2-7515-4412-ae3a-b64f90b79d2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask\n",
    "from dash import Dash"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "9c2af816-138e-432a-804a-2d996c2ec7ed",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "server = Flask(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "bdd7f98e-03bb-4e98-a6a6-051c1fcf8bc1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import dash\n",
    "import dash_html_components as html\n",
    "import dash_core_components as dcc\n",
    "from dash.dependencies import Input, Output, State\n",
    "import plotly.graph_objects as go"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "538d2ade-ca10-4801-97cb-a8bb3a600e13",
   "metadata": {
    "tags": []
   },
   "source": [
    "colors = {\n",
    "    'background': '#000',  # Black background\n",
    "    'text': '#FFF',        # Light text color\n",
    "    'primary': '#007BFF',\n",
    "    'secondary': '#6C757D',\n",
    "    'accent': '#FF6F61'\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aff26873-3165-40f8-af08-1a6c9acab36f",
   "metadata": {
    "tags": []
   },
   "source": [
    "app = dash.Dash(__name__ , suppress_callback_exceptions=True)#, url_base_pathname='/cot-dashboard/test1/')\n",
    "#app = Dash(__name__, server=server, suppress_callback_exceptions=True)\n",
    "\n",
    "# Create a button that will filter the dataframe.\n",
    "button = html.Button(\n",
    "    \"Go\",\n",
    "    id=\"filter-button\",\n",
    "    className=\"btn btn-primary\",\n",
    "    style={\"width\": \"100px\", \"height\": \"80px\"}\n",
    ")\n",
    "\n",
    "app.layout = html.Div([\n",
    "    html.H1('Interactive Dashboard'),\n",
    "    dcc.Tabs(id=\"tabs\", value='tab-1', children=[\n",
    "        dcc.Tab(label='COT Individual', value='tab-1'),\n",
    "        dcc.Tab(label='Combined COT chart', value='tab-2'),\n",
    "        dcc.Tab(label='Market Data Table', value = 'tab-3'),\n",
    "        # Add more tabs for other graphs as needed\n",
    "    ]),\n",
    "    html.Div(id='tabs-content'),\n",
    "])\n",
    "\n",
    "@app.callback(Output('tabs-content', 'children'),\n",
    "              Input('tabs', 'value'))\n",
    "def render_content(tab):\n",
    "    if tab == 'tab-1':\n",
    "        return html.Div([\n",
    "            dcc.DatePickerRange(id='date-range-picker', start_date=df['Date'].min(), end_date=df['Date'].max()),\n",
    "            dcc.Dropdown(id='category-dropdown', options=[\n",
    "                {'label': 'Non Commercial', 'value': 'Non Commercial'},\n",
    "                {'label': 'Commercial', 'value': 'Commercial'},\n",
    "                {'label': 'Non Reportable', 'value': 'Non Reportable'}\n",
    "            ]),\n",
    "            dcc.Dropdown(id='market-dropdown', options=[\n",
    "                {'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()\n",
    "            ], multi=True),\n",
    "            button,\n",
    "            dcc.Graph(id='graph-1')\n",
    "        ])\n",
    "    elif tab == 'tab-2':\n",
    "        return html.Div([\n",
    "            dcc.DatePickerRange(id='date-range-picker-2', start_date=df['Date'].min(), end_date=df['Date'].max()),\n",
    "            dcc.Dropdown(id='market-dropdown-2', options=[\n",
    "                {'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()\n",
    "            ], multi=True),\n",
    "            button,\n",
    "            dcc.Graph(id='graph-2')\n",
    "        ])\n",
    "    elif tab == 'tab-3':  # New tab for the data table\n",
    "        return html.Div([\n",
    "            dcc.DatePickerSingle(id='date-picker',date=df['Date'].min(),),  # Set the initial date here\n",
    "            dcc.Dropdown(id='category-dropdown', options=[\n",
    "                {'label': 'Non Commercial', 'value': 'Non Commercial'},\n",
    "                {'label': 'Commercial', 'value': 'Commercial'},\n",
    "                {'label': 'Non Reportable', 'value': 'Non Reportable'}\n",
    "            ]),\n",
    "            dcc.Dropdown(id='market-dropdown', options=[\n",
    "                {'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()\n",
    "            ], multi=True),\n",
    "            button,\n",
    "            dash_table.DataTable(id='table-1', columns=[\n",
    "                {'name': 'Market', 'id': 'Market'},\n",
    "                {'name': 'Long Position', 'id': 'Long Position'},\n",
    "                {'name': 'Short Position', 'id': 'Short Position'}\n",
    "            ]),\n",
    "        ])\n",
    "    # Add more conditions for additional tabs with different graphs\n",
    "\n",
    "@app.callback(\n",
    "    Output('graph-1', 'figure'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-range-picker', 'start_date'),\n",
    "     State('date-range-picker', 'end_date'),\n",
    "     State('category-dropdown', 'value'),\n",
    "     State('market-dropdown', 'value')]\n",
    ")\n",
    "\n",
    "#Bars are not correct ***\n",
    "\n",
    "def update_graph_1(n_clicks, start_date, end_date, category, markets):\n",
    "    if n_clicks is None:\n",
    "        # No button click yet, return an empty figure\n",
    "        return {}\n",
    "\n",
    "    fig = go.Figure()\n",
    "\n",
    "    colors = ['blue', 'red', 'green']  # Define colors for each market value\n",
    "\n",
    "    for i, market in enumerate(markets):\n",
    "        # Filter the DataFrame for the current market\n",
    "        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'] == market)]\n",
    "\n",
    "        if category == 'Non Commercial':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Commercial Net Position'], name=f'{market} - Non Commercial', marker_color=colors[i]))\n",
    "        elif category == 'Commercial':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Commercial Net Position'], name=f'{market} - Commercial', marker_color=colors[i]))\n",
    "        elif category == 'Non Reportable':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Reportable Net Position'], name=f'{market} - Non Reportable', marker_color=colors[i]))\n",
    "\n",
    "    fig.update_layout(\n",
    "        yaxis=dict(title='Net Position'),\n",
    "        xaxis=dict(title='Date'),\n",
    "        barmode='group',  # Use 'group' to place the bars side by side\n",
    "        bargap=0.1,  # Adjust this value to reduce the space between bars\n",
    "        legend=dict(yanchor=\"top\", y=0.99, xanchor=\"left\", x=0.01)\n",
    "    )\n",
    "\n",
    "    return fig\n",
    "\n",
    "\n",
    "@app.callback(\n",
    "    Output('graph-2', 'figure'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-range-picker-2', 'start_date'),\n",
    "     State('date-range-picker-2', 'end_date'),\n",
    "     State('market-dropdown-2', 'value')]\n",
    ")\n",
    "\n",
    "def update_graph_2(n_clicks, start_date, end_date, markets):\n",
    "    if n_clicks is None:\n",
    "        # No button click yet, return an empty figure\n",
    "        return {}\n",
    "\n",
    "    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'].isin(markets))]\n",
    "\n",
    "    fig = go.Figure()\n",
    "\n",
    "    # Calculate the offset for each market bar to make them side by side\n",
    "    n_markets = len(markets)\n",
    "    bar_width = 0.2  # Adjust this value to change the width of the bars\n",
    "    bar_offsets = np.linspace(-bar_width * (n_markets - 1) / 2, bar_width * (n_markets - 1) / 2, n_markets)\n",
    "\n",
    "    colors = ['blue', 'red', 'green']\n",
    "\n",
    "    for i, market in enumerate(markets):\n",
    "        x_offset = Timedelta(bar_offsets[i], unit='D')\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Commercial Net Position'], name=f'Non Commercial - {market}', marker_color=colors[0]))\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Commercial Net Position'], name=f'Commercial - {market}', marker_color=colors[1]))\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Reportable Net Position'], name=f'Non Reportable - {market}', marker_color=colors[2]))\n",
    "\n",
    "    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['open_interest_all'], name='Open Interest', yaxis='y2'))\n",
    "\n",
    "    fig.update_layout(\n",
    "        yaxis=dict(title='Net Position'),\n",
    "        yaxis2=dict(title='Open Interest (All)', side='right', overlaying='y'),\n",
    "        xaxis=dict(title='Date'),\n",
    "        barmode='group',  # Use 'group' to place the bars side by side\n",
    "        legend=dict(yanchor=\"top\", y=0.99, xanchor=\"left\", x=0.01)\n",
    "    )\n",
    "\n",
    "    return fig\n",
    "  \n",
    "@app.callback(\n",
    "    Output('table-1', 'data'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-picker', 'date'),\n",
    "     State('category-dropdown', 'value'),\n",
    "     State('market-dropdown', 'value')]\n",
    ")\n",
    "\n",
    "def update_table(n_clicks, selected_date, category, markets):\n",
    "    if n_clicks is None or not all([selected_date, category, markets]):\n",
    "        # No button click or missing inputs, return an empty table\n",
    "        return []\n",
    "\n",
    "    # Filter the data based on the selected date and markets\n",
    "    filtered_df = df[(df['Date'] == selected_date) & (df['market_and_exchange_names'].isin(markets))]\n",
    "\n",
    "    # Create a new list to hold the table data for each market\n",
    "    table_data = []\n",
    "\n",
    "    # Loop through each market and append its data to the table_data list\n",
    "    for market in markets:\n",
    "        # Filter the data for the specific market\n",
    "        market_data = filtered_df[filtered_df['market_and_exchange_names'] == market]\n",
    "\n",
    "        # Check if the filtered DataFrame for the market is empty\n",
    "        if market_data.empty:\n",
    "            # Handle the case when the filtered DataFrame is empty\n",
    "            long_position = 0\n",
    "            short_position = 0\n",
    "        else:\n",
    "            # Get the long and short position values based on the selected category\n",
    "            if category == 'Non Commercial':\n",
    "                long_position = market_data['Non Commercial Long %'].values[0]\n",
    "                short_position = market_data['Non Commercial Short %'].values[0]\n",
    "            elif category == 'Commercial':\n",
    "                long_position = market_data['Commercial Long %'].values[0]\n",
    "                short_position = market_data['Commercial Short %'].values[0]\n",
    "            elif category == 'Non Reportable':\n",
    "                long_position = market_data['Non Reportable Long %'].values[0]\n",
    "                short_position = market_data['Non Reportable Short %'].values[0]\n",
    "            else:\n",
    "                # Handle the case when no category is selected\n",
    "                long_position = 0\n",
    "                short_position = 0\n",
    "\n",
    "        # Convert the long and short position values to strings\n",
    "        long_position_str = str(long_position)\n",
    "        short_position_str = str(short_position)\n",
    "\n",
    "        # Convert the market value to a string\n",
    "        market_str = str(market)  # Ensure market_str is a string\n",
    "\n",
    "        # Create a dictionary for the table row and append it to table_data list\n",
    "        table_data.append({'Market': market_str, 'Long Position': long_position_str, 'Short Position': short_position_str})\n",
    "\n",
    "    return table_data\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e9bb425-63d4-49f4-a9d3-161442c31509",
   "metadata": {
    "tags": []
   },
   "source": [
    "options = [\n",
    "    {'label': 'EURO', 'value': 'EURO'},\n",
    "    {'label': 'CANADIAN DOLLAR', 'value': 'CANADIAN DOLLAR'},\n",
    "    {'label': 'SWISS FRANC', 'value': 'SWISS FRANC'},\n",
    "    {'label': 'JAPANESE YEN', 'value': 'JAPANESE YEN'},\n",
    "    {'label': 'AUSTRALIAN DOLLAR', 'value': 'AUSTRALIAN DOLLAR'},\n",
    "    {'label': 'USD INDEX', 'value': 'USD INDEX'},\n",
    "    {'label': 'BRAZILLIAN REAL', 'value': 'BRAZILLIAN REAL'},\n",
    "    {'label': 'EURO FX/BRITISH POUND XRATE', 'value': 'EURO FX/BRITISH POUND XRATE'},\n",
    "    {'label': 'E-MINI S&P 500', 'value': 'E-MINI S&P 500'},\n",
    "    {'label': 'RUSSEL E-MINI', 'value': 'RUSSEL E-MINI'},\n",
    "    {'label': 'BRITISH POUND', 'value': 'BRITISH POUND'},\n",
    "    {'label': 'NZ DOLLAR', 'value': 'NZ DOLLAR'}\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "0f7fe178-e366-4946-8e6c-5586fbab704a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import dash\n",
    "import dash_html_components as html\n",
    "import dash_core_components as dcc\n",
    "import plotly.graph_objects as go\n",
    "import pandas as pd\n",
    "\n",
    "# Load the COT data\n",
    "#df = pd.read_csv('cot.csv')\n",
    "\n",
    "# Create a dark theme provider\n",
    "theme = {\n",
    "    'color': {\n",
    "        'primary': '#000000',\n",
    "        'secondary': '#ffffff',\n",
    "        'text': '#333333',\n",
    "        'background': '#000000'\n",
    "    }\n",
    "}\n",
    "\n",
    "app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[\n",
    "        {'href': 'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css', 'rel': 'stylesheet'},\n",
    "        'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'\n",
    "    ])\n",
    "\n",
    "app.layout = html.Div([\n",
    "    html.H1('COT Dashboard', style={'color': theme['color']['primary']}),\n",
    "    dcc.Tabs(id=\"tabs\", value='tab-1', children=[\n",
    "        dcc.Tab(label='COT Individual', value='tab-1'),\n",
    "        dcc.Tab(label='Combined COT chart', value='tab-2'),\n",
    "        dcc.Tab(label='Market Data Table', value = 'tab-3'),\n",
    "        # Add more tabs for other graphs as needed\n",
    "    ]),\n",
    "    html.Div(id='tabs-content'),\n",
    "])\n",
    "\n",
    "@app.callback(Output('tabs-content', 'children'),\n",
    "              Input('tabs', 'value'))\n",
    "def render_content(tab):\n",
    "    if tab == 'tab-1':\n",
    "        return html.Div([\n",
    "            dcc.DatePickerRange(id='date-range-picker', start_date=df['Date'].min(), end_date=df['Date'].max(), style={'color': theme['color']['text']}),\n",
    "            dcc.Dropdown(id='category-dropdown', options=[\n",
    "                {'label': 'Non Commercial', 'value': 'Non Commercial'}, # 'style': {'color': theme['color']['text']}},\n",
    "                {'label': 'Commercial', 'value': 'Commercial'}, # 'style': {'color': theme['color']['text']}},\n",
    "                {'label': 'Non Reportable', 'value': 'Non Reportable'}, # 'style': {'color': theme['color']['text']}}\n",
    "            ], style={'color': theme['color']['text']}),\n",
    "            dcc.Dropdown(id='market-dropdown', \n",
    "            options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()],\n",
    "            multi=True, style={'color': theme['color']['text']}),\n",
    "            html.Button(\n",
    "                \"Go\",\n",
    "                id=\"filter-button\",\n",
    "                className=\"btn btn-primary\",\n",
    "                style={'color': theme['color']['primary']}\n",
    "            ),\n",
    "            dcc.Graph(id='graph-1', style={'color': theme['color']['text']})\n",
    "        ])\n",
    "    elif tab == 'tab-2':\n",
    "        return html.Div([\n",
    "            dcc.DatePickerRange(id='date-range-picker-2', start_date=df['Date'].min(), end_date=df['Date'].max(), style={'color': theme['color']['text']}),\n",
    "            dcc.Dropdown(id='market-dropdown-2', options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()]\n",
    "            , multi=True, style={'color': theme['color']['text']}),\n",
    "            html.Button(\n",
    "                \"Go\",\n",
    "                id=\"filter-button\",\n",
    "                className=\"btn btn-primary\",\n",
    "                style={'color': theme['color']['primary']}\n",
    "            ),\n",
    "            dcc.Graph(id='graph-2', style={'color': theme['color']['text']})\n",
    "        ])\n",
    "    elif tab == 'tab-3':\n",
    "        return html.Div([\n",
    "            dcc.DatePickerSingle(id='date-picker', date=df['Date'].min(), style={'color': theme['color']['text']}),\n",
    "            dcc.Dropdown(id='category-dropdown', options=[\n",
    "                {'label': 'Non Commercial', 'value': 'Non Commercial'}, # 'style': {'color': theme['color']['text']}},\n",
    "                {'label': 'Commercial', 'value': 'Commercial'}, # 'style': {'color': theme['color']['text']}},\n",
    "                {'label': 'Non Reportable', 'value': 'Non Reportable'},# 'style': {'color': theme['color']['text']}}\n",
    "            ], style={'color': theme['color']['text']}),\n",
    "            dcc.Dropdown(id='market-dropdown', \n",
    "            options = [{'label': market, 'value': market} for market in df['market_and_exchange_names'].unique()],\n",
    "            multi=True, style={'color': theme['color']['text']}),\n",
    "            html.Button(\n",
    "                \"Go\",\n",
    "                id=\"filter-button\",\n",
    "                className=\"btn btn-primary\",\n",
    "                style={'color': theme['color']['primary']}\n",
    "            ),\n",
    "            dash_table.DataTable(id='table-1', columns=[\n",
    "                {'name': 'Market', 'id': 'Market'},\n",
    "                {'name': 'Long Position', 'id': 'Long Position'},\n",
    "                {'name': 'Short Position', 'id': 'Short Position'}\n",
    "            ])\n",
    "        ])\n",
    "\n",
    "    # Add more tabs for other graphs as needed\n",
    "@app.callback(\n",
    "    Output('graph-1', 'figure'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-range-picker', 'start_date'),\n",
    "     State('date-range-picker', 'end_date'),\n",
    "     State('category-dropdown', 'value'),\n",
    "     State('market-dropdown', 'value')]\n",
    ")\n",
    "\n",
    "def update_graph_1(n_clicks, start_date, end_date, category, markets):\n",
    "    if n_clicks is None:\n",
    "        # No button click yet, return an empty figure\n",
    "        return {}\n",
    "\n",
    "    fig = go.Figure()\n",
    "\n",
    "    colors = ['blue', 'red', 'green']  # Define colors for each market value\n",
    "\n",
    "    for i, market in enumerate(markets):\n",
    "        # Filter the DataFrame for the current market\n",
    "        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'] == market)]\n",
    "\n",
    "        if category == 'Non Commercial':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Commercial Net Position'], name=f'{market} - Non Commercial', marker_color=colors[i]))\n",
    "        elif category == 'Commercial':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Commercial Net Position'], name=f'{market} - Commercial', marker_color=colors[i]))\n",
    "        elif category == 'Non Reportable':\n",
    "            fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Non Reportable Net Position'], name=f'{market} - Non Reportable', marker_color=colors[i]))\n",
    "\n",
    "    fig.update_layout(\n",
    "        yaxis=dict(title='Net Position'),\n",
    "        xaxis=dict(title='Date'),\n",
    "        barmode='group',  # Use 'group' to place the bars side by side\n",
    "        bargap=0.1,  # Adjust this value to reduce the space between bars\n",
    "        legend=dict(yanchor=\"top\", y=0.99, xanchor=\"left\", x=0.01)\n",
    "    )\n",
    "\n",
    "    return fig\n",
    "@app.callback(\n",
    "    Output('graph-2', 'figure'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-range-picker-2', 'start_date'),\n",
    "     State('date-range-picker-2', 'end_date'),\n",
    "     State('market-dropdown-2', 'value')]\n",
    ")\n",
    "\n",
    "def update_graph_2(n_clicks, start_date, end_date, markets):\n",
    "    if n_clicks is None:\n",
    "        # No button click yet, return an empty figure\n",
    "        return {}\n",
    "\n",
    "    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['market_and_exchange_names'].isin(markets))]\n",
    "\n",
    "    fig = go.Figure()\n",
    "\n",
    "    # Calculate the offset for each market bar to make them side by side\n",
    "    n_markets = len(markets)\n",
    "    bar_width = 0.2  # Adjust this value to change the width of the bars\n",
    "    bar_offsets = np.linspace(-bar_width * (n_markets - 1) / 2, bar_width * (n_markets - 1) / 2, n_markets)\n",
    "\n",
    "    colors = ['blue', 'red', 'green']\n",
    "\n",
    "    for i, market in enumerate(markets):\n",
    "        x_offset = Timedelta(bar_offsets[i], unit='D')\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Commercial Net Position'], name=f'{market} - Non Commercial', marker_color=colors[0]))\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Commercial Net Position'], name=f'{market} - Commercial', marker_color=colors[1]))\n",
    "        fig.add_trace(go.Bar(x=filtered_df['Date'] + x_offset, y=filtered_df['Non Reportable Net Position'], name=f'{market} - Non Reportable', marker_color=colors[2]))\n",
    "\n",
    "    fig.update_layout(\n",
    "        yaxis=dict(title='Net Position'),\n",
    "        xaxis=dict(title='Date'),\n",
    "        barmode='group',  # Use 'group' to place the bars side by side\n",
    "        bargap=0.1,  # Adjust this value to reduce the space between bars\n",
    "        legend=dict(yanchor=\"top\", y=0.99, xanchor=\"left\", x=0.01)\n",
    "    )\n",
    "\n",
    "    return fig\n",
    "@app.callback(\n",
    "    Output('table-1', 'data'),\n",
    "    [Input('filter-button', 'n_clicks')],\n",
    "    [State('date-picker', 'date'),\n",
    "     State('category-dropdown', 'value'),\n",
    "     State('market-dropdown', 'value')]\n",
    ")\n",
    "\n",
    "def update_table(n_clicks, selected_date, category, markets):\n",
    "    if n_clicks is None or not all([selected_date, category, markets]):\n",
    "        # No button click or missing inputs, return an empty table\n",
    "        return []\n",
    "\n",
    "    # Filter the data based on the selected date and markets\n",
    "    filtered_df = df[(df['Date'] == selected_date) & (df['market_and_exchange_names'].isin(markets))]\n",
    "\n",
    "    # Create a new list to hold the table data for each market\n",
    "    table_data = []\n",
    "\n",
    "    # Loop through each market and append its data to the table_data list\n",
    "    for market in markets:\n",
    "        # Filter the data for the specific market\n",
    "        market_data = filtered_df[filtered_df['market_and_exchange_names'] == market]\n",
    "\n",
    "        # Check if the filtered DataFrame for the market is empty\n",
    "        if market_data.empty:\n",
    "            # Handle the case when the filtered DataFrame is empty\n",
    "            long_position = 0\n",
    "            short_position = 0\n",
    "        else:\n",
    "            # Get the long and short position values based on the selected category\n",
    "            if category == 'Non Commercial':\n",
    "                long_position = market_data['Non Commercial Long %'].values[0]\n",
    "                short_position = market_data['Non Commercial Short %'].values[0]\n",
    "            elif category == 'Commercial':\n",
    "                long_position = market_data['Commercial Long %'].values[0]\n",
    "                short_position = market_data['Commercial Short %'].values[0]\n",
    "            elif category == 'Non Reportable':\n",
    "                long_position = market_data['Non Reportable Long %'].values[0]\n",
    "                short_position = market_data['Non Reportable Short %'].values[0]\n",
    "            else:\n",
    "                # Handle the case when no category is selected\n",
    "                long_position = 0\n",
    "                short_position = 0\n",
    "\n",
    "        # Convert the long and short position values to strings\n",
    "        long_position_str = str(long_position)\n",
    "        short_position_str = str(short_position)\n",
    "\n",
    "        # Convert the market value to a string\n",
    "        market_str = str(market)  # Ensure market_str is a string\n",
    "\n",
    "        # Create a dictionary for the table row and append it to table_data list\n",
    "        table_data.append({'Market': market_str, 'Long Position': long_position_str, 'Short Position': short_position_str})\n",
    "\n",
    "    return table_data\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "548cd8c9-eba6-4faa-9183-0764bea33620",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#if __name__ == '__main__':\n",
    "#    app.run_server(host='0.0.0.0', port=8025, debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "7a4f1d96-9708-4e24-86bc-65ce71efec4b",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"100%\"\n",
       "            height=\"650\"\n",
       "            src=\"http://127.0.0.1:5050/\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "            \n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.IFrame at 0x12230f0d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "#    server.run(app)\n",
    "    app.run_server(debug=False, port = 5050)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef986dcf-71ba-4458-8138-c610d307b13c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5722c27-b8f9-4d78-b9d4-1380d5348226",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0d1c1f3-7b21-4f7e-94dc-c396847f770e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "164f9e28-640f-4ec1-880d-57bbac364035",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06de047c-4548-4db6-b245-a2435bdc1831",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
