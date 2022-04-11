# Main file of the algview interactive visualization app

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# === Load Data ===

# Run data
mlic_data = pd.read_pickle('data/mlic_data.pkl.gz')
scep_data = pd.read_pickle('data/scep_data.pkl.gz')
scsc_data = pd.read_pickle('data/scsc_data.pkl.gz')

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

# === State ===

init_filter = {
    'algs': algs,
    'insts': insts,
    'single_inst': None,
}

# === Application Layout ===

# --- Reusable Styles ---

box_style = {'padding': '10px', 'border': '2px solid black'}

# --- Reusable Elements ---

alg_selector = html.Div([
    html.H5('Algorithms'),
    html.Div(id='div-alg-sel', children=[dcc.Checklist(options=algs, value=init_filter['algs'],
             id='checklist-alg')], style={'maxHeight': '400px', 'overflow': 'scroll'}),
], style=box_style)

inst_selector_check = dcc.Checklist(
    options=insts, value=init_filter['insts'], id='checklist-inst')
inst_selector_radio = dcc.RadioItems(
    options=insts, value=init_filter['single_inst'], id='radio-inst')

inst_selector = html.Div([
    html.H5('Instances'),
    html.Div(id='div-inst-sel', children=[inst_selector_check],
             style={'maxHeight': '400px', 'overflow': 'scroll'}),
], style=box_style)

alg_right = [
    dcc.Tabs(id='tabs-view', value='cactus-view', children=[
        dcc.Tab(label='Overview', value='cactus-view'),
        dcc.Tab(label='Pairwise Comparison', value='scatter-view'),
    ]),
    html.Div(id='div-view')
]

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
                html.Div(id='div-table', style=box_style),
            ]), width=4),
            dbc.Col(html.Div(id='div-right', style=box_style))
        ])]),
        dcc.Store(id='current-filter', data=init_filter),
    ], style={'margin': '20px'}
)

html.Div(id='test')

# === Application Callbacks ===


@ app.callback(
    Output('div-inst-sel', 'children'),
    Output('div-right', 'children'),
    Input('tabs-pages', 'value'),
)
def render_page(page):
    if page == 'algs-page':
        return inst_selector_check, alg_right
    elif page == 'inst-page':
        return inst_selector_radio, None


@ app.callback(
    Output('div-view', 'children'),
    Input('tabs-views', 'value'),
)
def render_view(view):
    if view == 'cactus-view':
        return None
    elif view == 'scatter-view':
        return None


@ app.callback(
    Output('current-filter', 'data'),
    Input('checklist-alg', 'value'),
    Input('checklist-inst', 'value'),
    Input('radio-inst', 'value'),
    State('current-state', 'data'),
)
def update_state(algs, insts, inst, state):
    state['algs'] = algs
    state['insts'] = insts
    state['single_inst'] = inst


# === Main ===

if __name__ == '__main__':
    app.run_server(debug=True)
