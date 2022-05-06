# Main file of the algview interactive visualization app

import numpy as np
import pandas as pd
import itertools
from dash import Dash, dcc, html, Input, Output, State, ALL, callback_context
from dash.exceptions import PreventUpdate
from dash.dash_table import DataTable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import dash_tabulator as dt
from .all_none_checklist import AllNoneChecklist, setup_anc

# === Load Data ===

# Run data
run_data = pd.read_pickle('data/run_data.pkl.gz')

# Instance data
mlic_inst_data = pd.read_pickle('data/mlic_instances.pkl.gz')
scep_inst_data = pd.read_pickle('data/scep_instances.pkl.gz')
scsc_inst_data = pd.read_pickle('data/scsc_instances.pkl.gz')

# Summaries
algs = np.sort(run_data['alg'].unique())
bioptsat_algs = [a for a in algs if 'bioptsat' in a]
pminimal_algs = [a for a in algs if 'pminimal' in a]
seesaw_algs = [a for a in algs if 'seesaw' in a]
paretomcs_algs = [a for a in algs if 'paretomcs' in a]
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


app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP,
                                 'https://codepen.io/chriddyp/pen/bWLwgP.css'],
           suppress_callback_exceptions=True)

server = app.server

# --- Reusable Styles ---

box_style = {'padding': '10px', 'border': '2px solid black'}

# --- Reusable Elements ---

startup_algs = ['bioptsat-msh', 'seesaw', 'pminimal', 'paretomcs']
alg_selector = html.Div([
    html.H5('Algorithms'),
    html.Div(id='div-alg-sel', children=dbc.Accordion([
        dbc.AccordionItem(AllNoneChecklist(bioptsat_algs, ['bioptsat-msh'],
                                           group='alg-filter', max_height='190px'),
                          title='BiOptSat'),
        dbc.AccordionItem(AllNoneChecklist(pminimal_algs, ['pminimal'],
                                           group='alg-filter', max_height='190px'),
                          title='P-minimal'),
        dbc.AccordionItem(AllNoneChecklist(seesaw_algs, ['seesaw-satexmin'],
                                           group='alg-filter', max_height='190px'),
                          title='Seesaw'),
        dbc.AccordionItem(AllNoneChecklist(paretomcs_algs, ['paretomcs'],
                                           group='alg-filter', max_height='190px'),
                          title='ParetoMCS'),
    ], flush=True), style={'height': '400px', 'maxHeight': '400px'}),
], style=box_style)

inst_selector_check = dbc.Accordion([
    dbc.AccordionItem(AllNoneChecklist(mlic_insts, mlic_insts,
                                       group='inst-filter', max_height='220px'),
                      title='Decision Rule Learning'),
    dbc.AccordionItem(AllNoneChecklist(scep_insts, scep_insts,
                                       group='inst-filter', max_height='220px'),
                      title='SetCovering-EP'),
    dbc.AccordionItem(AllNoneChecklist(scsc_insts, scsc_insts,
                                       group='inst-filter', max_height='220px'),
                      title='SetCovering-SC'),
], flush=True)
inst_selector_radio = dbc.Accordion([
    dbc.AccordionItem(html.Div(dcc.RadioItems(options=mlic_insts, value=mlic_insts[0],
                               id='radio-mlic-inst'), style={'maxHeight': '250px', 'overflow': 'scroll'}),
                      title='Decision Rule Learning'),
    dbc.AccordionItem(html.Div(dcc.RadioItems(options=scep_insts, value=None,
                               id='radio-scep-inst'), style={'maxHeight': '250px', 'overflow': 'scroll'}),
                      title='SetCovering-EP'),
    dbc.AccordionItem(html.Div(dcc.RadioItems(options=scsc_insts, value=None,
                               id='radio-scsc-inst'), style={'maxHeight': '250px', 'overflow': 'scroll'}),
                      title='SetCovering-SC'),
], flush=True)

inst_selector = html.Div([
    html.H5('Instances'),
    html.Div(id='div-inst-sel', children=[inst_selector_check],
             style={'height': '400px', 'maxHeight': '400px'}),
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

run_table = dt.DashTabulator(
    id='run-table', options={'height': '600px', 'selectable': True})

inst_right = [
    dcc.Graph(id='paretofront-plot'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='runtime-plot'), width=6),
        dbc.Col(dcc.Graph(id='progress-plot'), width=6),
    ]),
]

inst_tables = [
    html.H5('Instance Data'),
    dt.DashTabulator(id='inst-data-table'),
    html.H5('Run Data'),
    dt.DashTabulator(id='run-data-table', options={'height': '200px'}),
]

# --- Main Layout ---

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
            dbc.Col(html.Div(id='div-right', style=box_style), width=8)
        ])]),
        dcc.Store(id='filtered-alg'),
        dcc.Store(id='filtered-inst'),
        dcc.Store(id='selected-inst'),
        dcc.Store(id='single-inst'),
    ], style={'margin': '20px'}
)

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
        return inst_selector_radio, inst_right, inst_tables


@ app.callback(
    Output('div-view', 'children'),
    Input('tabs-view', 'value'),
)
def render_view(view):
    if view == 'cactus-view':
        return cactus_graphs
    elif view == 'scatter-view':
        return splom_graph


@ app.callback(
    Output('selected-inst', 'data'),
    Input('splom-plot', 'selectedData'),
    Input('run-table', 'multiRowsClicked'),
    State('filtered-inst', 'data'),
)
def update_selected_inst(splom_select, table_select, filtered_inst):
    ctx = callback_context
    trigger = None
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_inst = filtered_inst

    if trigger == 'splom-plot':
        if splom_select:
            selected_inst = [p['id'] for p in splom_select['points']]
    elif trigger == 'run-table':
        if table_select:
            selected_inst = [r['instance'] for r in table_select]

    return selected_inst


@ app.callback(
    Output('cactus-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('filtered-inst', 'data'),
)
def update_cactus(algs, insts):
    data = run_data[run_data['alg'].isin(algs) & run_data['instance'].isin(insts)
                    & ~run_data['timeout'] & ~run_data['memout']].sort_values('time')

    for a in algs:
        data.loc[data['alg'] == a, 'rank'] = range(
            1, (data['alg'] == a).sum()+1)

    fig = go.Figure()
    for a in algs:
        colour, dash = alg_colour_dash_scale(a)
        fig.add_trace(go.Scatter(x=data.loc[data['alg'] == a, 'rank'], y=data.loc[data['alg'] == a, 'time'],
                                 hovertemplate='<b>algorithm: ' + a +
                                 '</b><br><i># instances: %{x}<br>cpu time: %{y}s</i><br>instance: %{text}',
                                 text=data.loc[data['alg'] == a,
                                               'instance'], name=a, mode='lines+markers',
                                 line=dict(color=colour, dash=dash), marker=dict(size=5)))

    fig.update_xaxes(title_text='# instances')
    fig.update_yaxes(title_text='cpu time')

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20))

    return fig


@app.callback(
    Output('hist-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('filtered-inst', 'data'),
)
def update_hist(algs, insts):
    data = run_data[run_data['alg'].isin(
        algs) & run_data['instance'].isin(insts)]

    fig = go.Figure()
    for a in algs:
        colour, dash = alg_colour_dash_scale(a)
        fig.add_trace(go.Histogram(
            x=data.loc[data['alg'] == a, 'time'], name=a, marker=dict(color=colour)))

    fig.update_xaxes(title_text='cpu time')
    fig.update_yaxes(title_text='# runs')

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20), barmode='stack')

    return fig


@app.callback(
    Output('run-table', 'columns'),
    Output('run-table', 'data'),
    Input('filtered-alg', 'data'),
    Input('filtered-inst', 'data'),
)
def update_table(algs, insts):
    data = table_data.filter(items=insts, axis=0)[algs + ['instance']]

    columns = [{'title': 'Instance', 'field': 'instance'}]
    for a in algs:
        columns.append({'title': a, 'field': a, 'formatter': 'progress', 'formatterParams': {
                       'min': 0, 'max': 5400, 'legend': True, 'color': 'orange'},
            'hozAlign': 'left'})

    return columns, data.to_dict(orient='records')


@app.callback(
    Output('splom-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('filtered-inst', 'data'),
    Input('selected-inst', 'data'),
)
def update_splom(algs, insts, selected_inst):
    data = table_data.filter(items=insts, axis=0)[
        algs + ['instance', 'instance type']]

    instance_types = data['instance type'].unique()

    selectedpoints = dict()
    for it in instance_types:
        selectedpoints[it] = np.where(
            data.loc[data['instance type'] == it, 'instance'].isin(selected_inst))[0]

    fig = make_subplots(rows=len(algs), cols=len(
        algs), shared_xaxes=True, shared_yaxes=True)

    for i in range(len(algs)):
        for j in range(len(algs)):
            for it in instance_types:
                fig.add_trace(go.Scatter(x=data.loc[data['instance type'] == it, algs[i]], y=data.loc[data['instance type'] == it, algs[j]],
                                         hovertemplate='<b>instance: %{text}</b><br>' +
                                         algs[i] + ': %{x}<br>' +
                                         algs[j] + ': %{y}',
                                         text=data.loc[data['instance type']
                                                       == it, 'instance'],
                                         ids=data.loc[data['instance type']
                                                      == it, 'instance'],
                                         selectedpoints=selectedpoints[it],
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

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20), dragmode='select', height=1000, width=1000,
                      hovermode='closest', grid_xaxes=['x{}'.format(i) for i in range(1, len(algs)+1)])

    return fig


@app.callback(
    Output('paretofront-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('single-inst', 'data'),
)
def update_paretofront(algs, inst):
    if not inst:
        raise PreventUpdate

    data = run_data[run_data['alg'].isin(
        algs) & (run_data['instance'] == inst)]

    if data.shape[0] == 0:
        raise PreventUpdate

    up_name = data.iloc[0]['up name']
    down_name = data.iloc[0]['down name']

    fig = go.Figure()
    for a in algs:
        colour, dash = alg_colour_dash_scale(a)
        if (data['alg'] != a).all():
            continue
        pf = data.loc[data['alg'] == a, 'paretofront'].iloc[0]
        if data.loc[data['alg'] == a, 'up name'].iloc[0] == down_name:
            # Flip pareto front
            pf = [dict(up=pp['down'], down=pp['up'], **dict((k, pp[k])
                       for k in pp if k not in ('up', 'down'))) for pp in pf]
        fig.add_trace(go.Scatter(x=[pp['up'] for pp in pf], y=[pp['down'] for pp in pf],
                                 hovertemplate='<b>algorithm: ' + a +
                                 '</b><br><i>' + up_name +
                                 ': %{x}<br>' + down_name +
                                 ': %{y}</i><br>%{text}',
                                 text=['# sols: {}<br>sols: {}'.format(
                                     pp['n_sols'], pp['models']) for pp in pf],
                                 name=a, mode='lines+markers',
                                 line=dict(color=colour, dash=dash, shape='hv'), marker=dict(size=5)))

    fig.update_xaxes(title_text=up_name)
    fig.update_yaxes(title_text=down_name)

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20))

    return fig


@app.callback(
    Output('runtime-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('single-inst', 'data'),
)
def update_runtimes(algs, inst):
    if not inst:
        raise PreventUpdate

    data = run_data[run_data['alg'].isin(
        algs) & (run_data['instance'] == inst)].sort_values('time')

    if data.shape[0] == 0:
        raise PreventUpdate

    fig = go.Figure([go.Bar(x=data['alg'], y=data['time'], )])
    fig.update_traces(marker=dict(
        color=[alg_colour_dash_scale(a)[0] for a in data['alg'].unique()]))

    fig.update_xaxes(title_text='Algorithm')
    fig.update_yaxes(title_text='cpu time', type='log', range=[-2, 4])

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20))

    return fig


@app.callback(
    Output('progress-plot', 'figure'),
    Input('filtered-alg', 'data'),
    Input('single-inst', 'data'),
)
def update_progress(algs, inst):
    if not inst:
        raise PreventUpdate

    data = run_data[run_data['alg'].isin(
        algs) & (run_data['instance'] == inst)]

    if data.shape[0] == 0:
        raise PreventUpdate

    fig = go.Figure()
    for a in algs:
        colour, dash = alg_colour_dash_scale(a)
        if (data['alg'] != a).all():
            continue
        prog = data.loc[data['alg'] == a, 'progress'].iloc[0]
        up_name = data.loc[data['alg'] == a, 'up name'].iloc[0]
        down_name = data.loc[data['alg'] == a, 'down name'].iloc[0]
        if not isinstance(prog, list) or len(prog) == 0:
            # Progress data is not available for all algorithms
            continue
        fig.add_trace(go.Scatter(x=[0] + [i for i in range(1, len(prog)+1)], y=[0] + [pp['absolute cpu time'] for pp in prog],
                                 hovertemplate='<b>algorithm: ' + a +
                                 '</b><br><i># pareto points: %{x}<br>cup time: %{y}s</i><br>%{text}',
                                 text=['{}: {}<br>{}: {}'.format(
                                     up_name, pp['up'], down_name, pp['down']) for pp in prog],
                                 name=a, mode='lines+markers',
                                 line=dict(color=colour, dash=dash), marker=dict(size=5)))

    fig.update_xaxes(title_text='# pareto points')
    fig.update_yaxes(title_text='cpu time')

    fig.update_layout(margin=dict(l=20, r=20, b=20, t=20))

    return fig


@app.callback(
    Output('run-data-table', 'data'),
    Output('run-data-table', 'columns'),
    Input('filtered-alg', 'data'),
    Input('single-inst', 'data'),
)
def update_inst_run_table(algs, inst):
    if not inst:
        raise PreventUpdate

    data = run_data[run_data['alg'].isin(
        algs) & (run_data['instance'] == inst)]

    if data.shape[0] == 0:
        raise PreventUpdate

    table_data = pd.DataFrame(columns=algs)
    table_rows = ['time', 'timeout', 'memout', 'cluster node']
    for a in algs:
        if (data['alg'] != a).all():
            continue
        for row in table_rows:
            table_data.loc[row, a] = data.loc[data['alg'] == a, row].iloc[0]
        solver_stats = data.loc[data['alg'] == a, 'satsolver stats'].iloc[0]
        if isinstance(solver_stats, dict):
            for k, v in solver_stats.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        table_data.loc['{}.{}'.format(k, k2), a] = v2
                else:
                    table_data.loc[k, a] = v

    table_data['datum'] = table_data.index

    return table_data.to_dict('records'),\
        [{'title': 'Datum', 'field': 'datum', 'headerSort': False}] + \
        [{'title': a, 'field': a, 'headerSort': False} for a in algs]


@app.callback(
    Output('inst-data-table', 'data'),
    Output('inst-data-table', 'columns'),
    Input('single-inst', 'data'),
)
def update_inst_run_table(inst):
    if not inst:
        raise PreventUpdate

    data = None
    if inst.startswith('fixed-element-prob'):
        data = scep_inst_data.loc[inst]
    elif inst.startswith('fixed-set-card'):
        data = scsc_inst_data.loc[inst]
    else:
        data = mlic_inst_data.loc[inst]

    table_data = pd.DataFrame(columns=algs)
    for row in data.index:
        table_data.loc[row, 'value'] = data[row]

    table_data['datum'] = table_data.index

    return table_data.to_dict('records'),\
        [{'title': 'Datum', 'field': 'datum', 'headerSort': False},
         {'title': 'Value', 'field': 'value', 'headerSort': False}]


setup_anc(app)


@app.callback(
    Output('filtered-inst', 'data'),
    Input({'type': AllNoneChecklist.checklist_type, 'group': 'inst-filter', 'index': ALL}, 'value'),
)
def update_filtered_insts(values):
    return list(itertools.chain.from_iterable(values))


@app.callback(
    Output('single-inst', 'data'),
    Output('radio-mlic-inst', 'value'),
    Output('radio-scep-inst', 'value'),
    Output('radio-scsc-inst', 'value'),
    Input('radio-mlic-inst', 'value'),
    Input('radio-scep-inst', 'value'),
    Input('radio-scsc-inst', 'value'),
)
def update_radio_inst(mlic, scep, scsc):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'mlic' in trigger:
        return mlic, mlic, None, None
    elif 'scep' in trigger:
        return scep, None, scep, None
    elif 'scsc' in trigger:
        return scsc, None, None, scsc
    else:
        raise PreventUpdate


@app.callback(
    Output('filtered-alg', 'data'),
    Input({'type': AllNoneChecklist.checklist_type, 'group': 'alg-filter', 'index': ALL}, 'value'),
)
def update_filtered_algs(values):
    return list(itertools.chain.from_iterable(values))
