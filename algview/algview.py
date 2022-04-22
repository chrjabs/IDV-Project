# Main file of the algview interactive visualization app

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_tabulator as dt

# === Load Data ===

# Run data
mlic_data = pd.read_pickle('data/mlic_data.pkl.gz')
scep_data = pd.read_pickle('data/scep_data.pkl.gz')
scsc_data = pd.read_pickle('data/scsc_data.pkl.gz')

# Cactus Data
cactus_include = ['time', 'instance', 'alg', 'timeout', 'memout']
run_data = pd.concat([mlic_data[cactus_include],
                       scep_data[cactus_include], scsc_data[cactus_include]])

# Instance data
mlic_inst_data = pd.read_pickle('data/mlic_instances.pkl.gz')
scep_inst_data = pd.read_pickle('data/scep_instances.pkl.gz')
scsc_inst_data = pd.read_pickle('data/scsc_instances.pkl.gz')

# Summaries
algs = pd.concat([mlic_data['alg'], scep_data['alg'],
                 scsc_data['alg']]).unique()
mlic_insts = mlic_data['instance'].unique()
scep_insts = scep_data['instance'].unique()
scsc_insts = scsc_data['instance'].unique()
insts = pd.concat(
    [mlic_data['instance'], scep_data['instance'], scsc_data['instance']]).unique()

# Table Data
table_data = pd.DataFrame()
for alg in run_data['alg'].unique():
    for inst in run_data['instance'].unique():
        try:
            table_data.loc[inst, alg] = run_data.loc[(
                run_data['alg'] == alg) & (run_data['instance'] == inst), 'time'].iloc[0]
        except IndexError:
            table_data.loc[inst, alg] = np.nan
table_data['instance'] = table_data.index

# === Colour Scale ===

colour_scale = [
    # Tsitsul normal 12 colours colour scale
    # http://tsitsul.in/blog/coloropt/
    '#ebac23',
    '#b80058',
    '#008cf9',
    '#006e00',
    '#00bbad',
    '#d163e6',
    '#b24502',
    '#ff9287',
    '#5954d6',
    '#00c6f8',
    '#878500',
    '#00a75c',
    '#bdbdbd',
]
dash_scale = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']


def alg_colour_dash_scale(alg):
    toks = alg.split('-')
    if toks[0] == 'bioptsat':
        if toks[1] == 'msh':
            colour = colour_scale[0]
        elif toks[1] == 'su':
            colour = colour_scale[1]
        elif toks[1] == 'msu3':
            colour = colour_scale[2]
        elif toks[1] == 'us':
            colour = colour_scale[3]
        elif toks[1] == 'oll':
            colour = colour_scale[4]
        else:
            colour = colour_scale[hash(toks[1]) % len(colour_scale)]

        if len(toks) <= 2:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[2:])) %
                               (len(dash_scale) - 1)) + 1]
    elif toks[0] == 'pminimal':
        colour = colour_scale[5]
        dash = dash_scale[0]
    elif toks[0] == 'seesaw':
        colour = colour_scale[6]
        if len(toks) <= 1:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[1:])) %
                               (len(dash_scale) - 1)) + 1]
    elif toks[0] == 'paretomcs':
        colour = colour_scale[7]
        if len(toks) <= 1:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[1:])) %
                               (len(dash_scale) - 1)) + 1]

    return colour, dash

# === Application Layout ===

# --- Reusable Styles ---


box_style = {'padding': '10px', 'border': '2px solid black'}

# --- Reusable Elements ---

startup_algs = ['bioptsat-msh', 'seesaw', 'pminimal', 'paretomcs']
alg_selector = html.Div([
    html.H5('Algorithms'),
    html.Div(id='div-alg-sel', children=[dcc.Checklist(options=algs, value=startup_algs,
             id='checklist-alg')], style={'height': '400px', 'maxHeight': '400px', 'overflow': 'scroll'}),
], style=box_style)

inst_selector_check = dcc.Checklist(
    options=insts, value=insts, id='checklist-inst')
inst_selector_radio = dcc.RadioItems(
    options=insts, value=None, id='radio-inst')

inst_selector = html.Div([
    html.H5('Instances'),
    html.Div(id='div-inst-sel', children=[inst_selector_check],
             style={'height': '400px', 'maxHeight': '400px', 'overflow': 'scroll'}),
], style=box_style)

alg_right = [
    dcc.Tabs(id='tabs-view', value='cactus-view', children=[
        dcc.Tab(label='Overview - Cactus and Histogram', value='cactus-view'),
        dcc.Tab(label='Pairwise Comparison', value='scatter-view'),
    ]),
    html.Div(id='div-view')
]

cactus_graphs = [
    dcc.Graph(id='cactus-plot'),
    dcc.Graph(id='hist-plot'),
]

run_table = dt.DashTabulator(id='run-table')

# --- Main Layout ---

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP,
                                 'https://codepen.io/chriddyp/pen/bWLwgP.css'],
           suppress_callback_exceptions=True)

server = app.server

app.layout = html.Div(
    children=[
        html.H1('AlgView - View Algorithm Runtime Data'),
        dcc.Tabs(id='tabs-pages', value='algs-page', children=[
            dcc.Tab(label='Algorithms Page', value='algs-page'),
            dcc.Tab(label='Instance Page', value='inst-page'),
        ]),
        html.Div([dbc.Row([
            dbc.Col(html.Div([
                dbc.Row([
                    dbc.Col(alg_selector),
                    dbc.Col(inst_selector),
                ]),
                html.Div(id='div-table', style=dict(maxHeight='600px',
                         height='600px', overflow='scroll', **box_style)),
            ]), width=4),
            dbc.Col(html.Div(id='div-right', style=box_style))
        ])]),
    ], style={'margin': '20px'}
)

html.Div(id='test')

# === Application Callbacks ===


@ app.callback(
    Output('div-inst-sel', 'children'),
    Output('div-right', 'children'),
    Output('div-table', 'children'),
    Input('tabs-pages', 'value'),
)
def render_page(page):
    if page == 'algs-page':
        return inst_selector_check, alg_right, run_table
    elif page == 'inst-page':
        return inst_selector_radio, None, None


@ app.callback(
    Output('div-view', 'children'),
    Input('tabs-view', 'value'),
)
def render_view(view):
    if view == 'cactus-view':
        return cactus_graphs
    elif view == 'scatter-view':
        return None


@app.callback(
    Output('cactus-plot', 'figure'),
    Input('checklist-alg', 'value'),
    Input('checklist-inst', 'value'),
)
def update_cactus(algs, insts):
    data = run_data[run_data['alg'].isin(algs) & run_data['instance'].isin(insts)
                     & ~run_data['timeout'] & ~run_data['memout']].sort_values('time')

    for a in algs:
        data.loc[data['alg'] == a, 'rank'] = range(
            1, (data['alg'] == a).sum()+1)

    fig = go.Figure()
    for i, a in enumerate(data['alg'].unique()):
        colour, dash = alg_colour_dash_scale(a)
        fig.add_trace(go.Scatter(x=data.loc[data['alg'] == a, 'rank'], y=data.loc[data['alg'] == a, 'time'],
                                 hovertemplate='<b>algorithm: ' + a +
                                 '</b><br><i># instances: %{x}<br>cpu time: %{y}s</i><br>instance: %{text}',
                                 text=data.loc[data['alg'] == a,
                                               'instance'], name=a, mode='lines+markers',
                                 line=dict(color=colour, dash=dash), marker=dict(size=5)))

    fig.update_xaxes(title_text='# instances')
    fig.update_yaxes(title_text='cpu time')

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


@app.callback(
    Output('hist-plot', 'figure'),
    Input('checklist-alg', 'value'),
    Input('checklist-inst', 'value'),
)
def update_hist(algs, insts):
    data = run_data[run_data['alg'].isin(
        algs) & run_data['instance'].isin(insts)]

    fig = go.Figure()
    for i, a in enumerate(data['alg'].unique()):
        colour, dash = alg_colour_dash_scale(a)
        fig.add_trace(go.Histogram(
            x=data.loc[data['alg'] == a, 'time'], name=a, marker=dict(color=colour)))

    fig.update_xaxes(title_text='cpu time')
    fig.update_yaxes(title_text='# runs')

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), barmode='stack')

    return fig


@app.callback(
    Output('run-table', 'columns'),
    Output('run-table', 'data'),
    Input('checklist-alg', 'value'),
    Input('checklist-inst', 'value'),
)
def update_hist(algs, insts):
    data = table_data.filter(items=insts, axis=0)[algs + ['instance']]

    columns = [{'title': 'Instance', 'field': 'instance'}]
    for a in algs:
        columns.append({'title': a, 'field': a, 'formatter': 'progress', 'formatterParams': {'min': 0, 'max': 5400}, 'hozAlign': 'left'})

    return columns, data.to_dict(orient='records')


# === Main ===

if __name__ == '__main__':
    app.run_server(debug=True)
