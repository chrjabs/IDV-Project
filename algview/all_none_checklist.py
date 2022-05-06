# Helper class to create checklists with selector buttons for all or none

import json
from dash import dcc, html, Input, Output, MATCH, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


def setup_anc(app):
    app.callback(_callback_output, _callback_inputs)(_callback)


class AllNoneChecklist(html.Div):
    nextIdx = 0
    all_button_type = 'anc-all-button'
    none_button_type = 'anc-none-button'
    checklist_type = 'anc-checklist'

    alls = []

    def __init__(self, options, value, group='default', max_height=None):
        self.idx = AllNoneChecklist.nextIdx
        self.alls.append(options)
        AllNoneChecklist.nextIdx += 1

        style = dict(**({'maxHeight': max_height}
                     if max_height else {}), overflow='scroll')

        html.Div.__init__(self, children=[
            dbc.Row([
                    dbc.Col(dbc.Button('All', outline=False, color='primary',
                                       id={'type': self.all_button_type, 'group': group,
                                           'index': self.idx}), width=6),
                    dbc.Col(dbc.Button('None', outline=True, color='primary',
                                       id={'type': self.none_button_type, 'group': group,
                                           'index': self.idx}), width=6),
                    ]),
            html.Div(dcc.Checklist(options=options, value=value,
                                   id={'type': self.checklist_type, 'group': group, 'index': self.idx}),
                     style=style)
        ], id={'type': 'anc', 'group': group, 'index': self.idx})


_callback_inputs = [
    Input({'type': AllNoneChecklist.all_button_type, 'group': MATCH, 'index': MATCH}, 'n_clicks'),
    Input({'type': AllNoneChecklist.none_button_type, 'group': MATCH, 'index': MATCH}, 'n_clicks')
]
_callback_output = Output(
    {'type': AllNoneChecklist.checklist_type, 'group': MATCH, 'index': MATCH}, 'value')


def _callback(all_button, none_button):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])

    if 'all' in trigger['type']:
        return AllNoneChecklist.alls[trigger['index']]
    elif 'none' in trigger['type']:
        return []
    else:
        raise PreventUpdate
