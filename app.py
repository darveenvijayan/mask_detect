import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='3D Clusters'

# make a sample data frame with 6 columns
np.random.seed(0)
df = pd.DataFrame({"Col " + str(i+1): np.random.rand(30) for i in range(6)})

app.layout = html.Div([
    
    html.Div([
    
        html.H1(children='A 3-Dimensional perspective of points'),
    ], style={'width':'75%', 'margin':25, 'textAlign': 'center'}),

    html.Div([
        html.Div(
            dcc.Graph(id='g1', config={'displayModeBar': False}),
            className='four columns'
        ),
        html.Div(
            dcc.Graph(id='g2', config={'displayModeBar': False}),
            className='four columns'
            ),
        html.Div(
            dcc.Graph(id='g3', config={'displayModeBar': False}),
            className='four columns'
        )
    ], className='row'),
    
    html.H4(children='Visualization by Darveen'),
    
    html.A(html.Button('Checkout my Github', className='three columns'), href='https://github.com/darveenvijayan'),
    
], className='row')

# app.layout = html.Div([
    
#     html.H1(children='A 3-Dimensional perspective of points'),
    
    
#     html.Div([
 
#     html.Div(
#         dcc.Graph(id='g1', config={'displayModeBar': False}),
#         className='four columns'
#     ),
#     html.Div(
#         dcc.Graph(id='g2', config={'displayModeBar': False}),
#         className='four columns'
#         ),
#     html.Div(
#         dcc.Graph(id='g3', config={'displayModeBar': False}),
#         className='four columns'
#     )], className='row'),

    
#     html.H4(children='Visualization by Darveen'),
    
# #     html.A(html.Button('Darveen', className='three columns'),
# #     href='https://github.com/darveenvijayan')
# #     ),
#     )]

def get_figure(df, x_col, y_col, selectedpoints, selectedpoints_local):

    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    else:
        selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    return {
        'data': [{
            'x': df[x_col],
            'y': df[y_col],
            'text': df.index,
            'textposition': 'top',
            'selectedpoints': selectedpoints,
            'customdata': df.index,
            'type': 'scatter',
            'mode': 'markers+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 12 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'margin': {'l': 20, 'r': 0, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': False,
            # Display a rectangle to highlight the previously selected region
            'shapes': [dict({
                'type': 'rect',
                'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }
            }, **selection_bounds
            )]
        }
    }

# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output('g1', 'figure'),
     Output('g2', 'figure'),
     Output('g3', 'figure')],
    [Input('g1', 'selectedData'),
     Input('g2', 'selectedData'),
     Input('g3', 'selectedData')]
)
def callback(selection1, selection2, selection3):
    selectedpoints = df.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])

    return [get_figure(df, "Col 1", "Col 2", selectedpoints, selection1),
            get_figure(df, "Col 3", "Col 4", selectedpoints, selection2),
            get_figure(df, "Col 5", "Col 6", selectedpoints, selection3)]


if __name__ == '__main__':
    app.run_server()

# import dash
# import dash_core_components as dcc
# import dash_html_components as html

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# server = app.server
# app.title='3D Clusters'

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])

# if __name__ == '__main__':
#     app.run_server()
