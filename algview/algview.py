# Main file of the algview interactive visualization app

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import dash_tabulator as dt

# === Load Data ===

# Run data
run_data = pd.read_pickle('data/run_data.pkl.gz')

# Instance data
mlic_inst_data = pd.read_pickle('data/mlic_instances.pkl.gz')
scep_inst_data = pd.read_pickle('data/scep_instances.pkl.gz')
scsc_inst_data = pd.read_pickle('data/scsc_instances.pkl.gz')

# Summaries
algs = run_data['alg'].unique()
mlic_insts = mlic_inst_data.index.to_series()
scep_insts = scep_inst_data.index.to_series()
scsc_insts = scsc_inst_data.index.to_series()
insts = pd.concat([mlic_insts, scep_insts, scsc_insts]).unique()

# Table Data
table_data = pd.read_pickle('data/table_data.pkl.gz')

# === Colour Scale ===

long_colour_scale = [
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
short_colour_scale = [
    # Tsitsul normal 8 colours colour scale
    # http://tsitsul.in/blog/coloropt/
    '#4053d3',
    '#ddb310',
    '#b51d14',
    '#00beff',
    '#fb49b0',
    '#00b25d',
    '#cacaca',
]
dash_scale = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']


def alg_colour_dash_scale(alg):
    toks = alg.split('-')
    if toks[0] == 'bioptsat':
        if toks[1] == 'msh':
            colour = long_colour_scale[0]
        elif toks[1] == 'su':
            colour = long_colour_scale[1]
        elif toks[1] == 'msu3':
            colour = long_colour_scale[2]
        elif toks[1] == 'us':
            colour = long_colour_scale[3]
        elif toks[1] == 'oll':
            colour = long_colour_scale[4]
        else:
            colour = long_colour_scale[hash(toks[1]) % len(long_colour_scale)]

        if len(toks) <= 2:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[2:])) %
                               (len(dash_scale) - 1)) + 1]
    elif toks[0] == 'pminimal':
        colour = long_colour_scale[5]
        dash = dash_scale[0]
    elif toks[0] == 'seesaw':
        colour = long_colour_scale[6]
        if len(toks) <= 1:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[1:])) %
                               (len(dash_scale) - 1)) + 1]
    elif toks[0] == 'paretomcs':
        colour = long_colour_scale[7]
        if len(toks) <= 1:
            dash = dash_scale[0]
        else:
            dash = dash_scale[(hash('-'.join(toks[1:])) %
                               (len(dash_scale) - 1)) + 1]

    return colour, dash


def instance_type_colour(inst_type):
    if inst_type == 'Decision Rule Learning':
        return short_colour_scale[0]
    elif inst_type == 'SetCovering-EP':
        return short_colour_scale[1]
    elif inst_type == 'SetCovering-SC':
        return short_colour_scale[2]
    else:
        return short_colour_scale[3]

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

splom_graph = dcc.Graph(id='splom-plot')

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
        return splom_graph


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
        columns.append({'title': a, 'field': a, 'formatter': 'progress', 'formatterParams': {
                       'min': 0, 'max': 5400}, 'hozAlign': 'left'})

    return columns, data.to_dict(orient='records')


@app.callback(
    Output('splom-plot', 'figure'),
    Input('checklist-alg', 'value'),
    Input('checklist-inst', 'value'),
)
def update_splom(algs, insts):
    data = table_data.filter(items=insts, axis=0)[
        algs + ['instance', 'instance type']]

    fig = make_subplots(rows=len(algs), cols=len(
        algs), shared_xaxes=True, shared_yaxes=True)

    for i in range(len(algs)):
        for j in range(len(algs)):
            for it in data['instance type'].unique():
                fig.add_trace(go.Scatter(x=data.loc[data['instance type'] == it, algs[i]], y=data.loc[data['instance type'] == it, algs[j]],
                                         hovertemplate='<b>instance: %{text}</b><br>' +
                                         algs[i] + ': %{x}<br>' +
                                         algs[j] + ': %{y}',
                                         text=data.loc[data['instance type']
                                                       == it, 'instance'],
                                         legendgroup=it, showlegend=(i == 0 and j == 0), mode='markers', marker=dict(color=instance_type_colour(it),
                                         line_color='white', line_width=0.5), name=it), row=j+1, col=i+1)

    diagonal_lines = []
    for i in range(1, len(algs)**2+1):
        xax = 'x{}'.format(i if i > 1 else '')
        yax = 'y{}'.format(i if i > 1 else '')
        diagonal_lines.extend([dict(layer='below', type='line', xref=xax, yref=yax, y0=0.01, y1=10000, x0=0.01, x1=10000,
                                    line=dict(color=short_colour_scale[6], dash='dash')),
                               dict(layer='below', type='line', xref=xax, yref=yax, y0=0.02, y1=10000,
                                    x0=0.01, x1=5000, line=dict(color=short_colour_scale[6], dash='dot')),
                               dict(layer='below', type='line', xref=xax, yref=yax, y0=0.01, y1=5000,
                                    x0=0.02, x1=10000, line=dict(color=short_colour_scale[6], dash='dot'))])

    fig.update_layout(shapes=diagonal_lines)
    fig.update_xaxes(type='log', range=[-2, 4],
                     matches='x', constrain='domain')
    fig.update_yaxes(type='log', matches='x', scaleanchor='x', scaleratio=1)

    for i in range(len(algs)):
        fig.update_xaxes(title_text=algs[i], row=len(algs), col=i+1)
        fig.update_yaxes(title_text=algs[i], row=i+1, col=1)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), dragmode='select', height=1000, width=1000,
                      hovermode='closest', grid_xaxes=['x{}'.format(i) for i in range(1, len(algs)+1)])

    return fig


# === Main ===

if __name__ == '__main__':
    app.run_server(debug=True)
