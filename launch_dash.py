import pickle
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np
import sys
import json
from itertools import chain


sankey_path = 'data/sankey_dash/'
figs = pickle.load(open(sankey_path+'figs.pkl', 'rb'))
tops = pickle.load(open(sankey_path+'tops.pkl', 'rb'))
top_words = pickle.load(open(sankey_path+'top_words.pkl', 'rb'))
combined = pickle.load(open(sankey_path+'combined.pkl', 'rb'))
author_list = pickle.load(open(sankey_path+'author_list.pkl', 'rb'))
labels = pickle.load(open(sankey_path+'labels.pkl', 'rb'))
positions = pickle.load(open(sankey_path+'positions.pkl', 'rb'))
sources = pickle.load(open(sankey_path+'sources.pkl', 'rb'))
targets = pickle.load(open(sankey_path+'targets.pkl', 'rb'))
locations = pickle.load(open(sankey_path+'locations.pkl', 'rb'))
models = pickle.load(open(sankey_path+'models.pkl', 'rb'))
names = pickle.load(open(sankey_path+'names.pkl', 'rb'))
num_authors = pickle.load(open(sankey_path+'num_authors.pkl', 'rb'))

num_topics_list = json.load(open('config/sankey-params.json', 'r'))['num_topics_list']

# read google scholar labels
with open('data/raw/gs_labels.json', 'r') as f:
    gs_labels = json.load(f)

threshold = .1
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.layout = html.Div([
  dbc.Row([
      html.Div([
          html.H1(children="HDSI Faculty Exploration Tool", className="header-title",
              style={
                'color' :'#666699',
                'margin': '4px auto',
                'text-align': 'center',
                'padding-top': '20px'
              }),

          html.P(

              children="Analyze and search through"

          " HDSI's faculty powered by Topic Modeling",

              className="header-description",
              style={
                'color' :'#666699',
                'margin': '4px auto',
                'padding-bottom': '20px',
                'text-align': 'center'
              },

          ),
      ]),
  ]),

  dbc.Row([
    dbc.Col(html.Div([
      dcc.Dropdown(
        id='graph-dropdown',
        placeholder='select number of LDA topics',
        options=[{'label' : f'{i} Topic Model', 'value' : i} for i in num_topics_list],
        style={
          'color' : 'black',
          #'background-color' : '#666699',
          #'width' : '100%',
          'align-items' : 'left',
          'justify-content' : 'space-around',
          'padding' : '3px',
          'padding-left': '10px',
          #'width' : '50vw',
          #'background-size':'cover',
          #'background-color' : '#666699'
          'animation': 'mymove 5s infinite',
          'animation-delay': '2s'

        },
        value=10
      ),
      dcc.Graph(
        id = 'graph',
        figure = figs[.1][10]
      )
      ],
      style={
        'width' : '50vw',
        'height' : '100vh',
        'overflow-y' : 'scroll'
      }
    )
    ),
      dbc.Col(html.Div([dbc.Col([
        dcc.Dropdown(
          id='dropdown_menu',
          placeholder='Select a topic',
          options=[{'label' : f'Topic {topic}: {top_words[10][topic]}', 'value' : topic} for topic in range(10)],
          optionHeight=100,
          style={
            'color' : 'black',
            #'background-color' : 'white',
            'align-items' : 'left',
            'justify-content' : 'left',
            'padding' : '3px',
            #'background-size':'contain',

          }
        ),
        dcc.Dropdown(
          id='researcher-dropdown',
          placeholder='Select Researchers',
          options=[{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)],
          style={
            'color' : 'black',
            #'background-color' : 'white',
            'align-items' : 'left',
            'justify-content' : 'left',
            'padding' : '3px'
          }
        )]),
        dbc.Col(
          dcc.Dropdown(
            id='word-search',
            placeholder='Search by Keyword',
            options=[{'label' : word, 'value' : word} for word in names],
            style={
              'color' : 'black',
              #'background-color' : 'white',
              'align-items' : 'center',
              #'justify-content' : 'left',
              'padding' : '3px',
              #'padding-bottom': '10px'
            },
            value=[],
            multi=True
          )
        ),
        html.Div(
          id='paper_container',
          children=[
            html.P(
              children=['Top 5 Papers'],
              #className='card',
              id='titles_and_authors',
              draggable=False,
              style={
                'font-size' :'100%',
                'font-family' : 'Verdana',
                'padding-left': '3px',
                'padding-top': '10px'
              }
            ),
          ],
        ),
        #html.Div(
        #  id='paper_container',
        #  children=[
        #    html.P(
        #      children=['Top 5 Papers'],
        #      id='titles_and_authors',
        #      draggable=False,
        #      style={
        #        'font-size' :'100%',
        #        'font-family' : 'Verdana'
        #      }
        #    ),
        #  ],
        #),

      ],
        style={
          'height' : '100vh',
          'overflow-y' : 'scroll'
        }
      )
      )
    ]
  )]
)

@app.callback(
  Output('titles_and_authors', 'children'),
  Output('researcher-dropdown', 'options'),
  Input('dropdown_menu', 'value'),
  Input('graph-dropdown', 'value'),
  Input('researcher-dropdown', 'value'),
  Input('word-search', 'value')
)
def update_p(topic, num_topics, author, words):
  if len(words) != 0:
    doc_vec = np.zeros((1, len(names)))
    for word in words:
      doc_vec[0][locations[word]] += 1
    relations = np.round(models[f'{num_topics}'].transform(doc_vec), 5).tolist()[0]
    pairs = [(i, relation) for i, relation in enumerate(relations)]
    pairs.sort(reverse=True, key=lambda x: x[1])
    to_return = [[html.Br(), f'Topic{pair[0]}: {pair[1]}', html.Br()] for pair in pairs]
    return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

  if topic == None and author == None:
    return ['Make a selection'], [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

  if topic != None and author == None:
    df = tops[threshold][num_topics][topic]
    df_authors = df.author.unique()
    max_vals = df.groupby('author').max()[f'{topic}_relevance']

    to_return = [[f'{name}:', html.Br(),
      f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}',
      html.Details([html.Summary('Abstract'),
                    html.Div(combined[combined.title == f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}'].abstract)],
                    style={
                      'font-size' :'80%',
                      'font-family' : 'Verdana'}),
      html.Br()] for i, name in enumerate(max_vals.index)]
    return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].author.unique()]

  if topic == None and author != None:
    to_return = []
    to_return.append([f"Google Scholar Labels: ",f'{gs_labels[author]}', html.Br(), html.Br()])
    for topic_num in range(num_topics):
      df = tops[threshold][num_topics][topic_num]
      if author in df.author.unique():
        max_vals = df.groupby('author').max()[f'{topic_num}_relevance']

        to_return.append([
          f'Topic {topic_num}:', html.Br(),
          f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}',
          html.Details([html.Summary('Abstract'),
                        html.Div(combined[combined.title == f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}'].abstract)],
                        style={
                          'font-size' :'80%',
                          'font-family' : 'Verdana'},
                        ),
          html.Br()])
    return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

  if topic != None and author != None:
    df = tops[threshold][num_topics][topic]
    df = df[df['author'] == author]
    df.sort_values(by=f'{topic}_relevance', ascending=False, inplace=True)
    titles = df.head(10)['title'].to_list()

    gs = [f"Google Scholar Labels: ",f'{gs_labels[author]}', html.Br(), html.Br()]

    to_return_more = [
      [html.Br(),
      f'{i} : {title}',
      html.Details([html.Summary('Abstract'),
                    html.Div(combined[combined.title == title].abstract)],
                    style={
                      'font-size' :'80%',
                      'font-family' : 'Verdana'})
      ] for i, title in enumerate(titles)]

    to_return = gs + to_return_more
    return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].author.unique()]



@app.callback(
  [Output('graph', 'figure'), Output('dropdown_menu', 'options')],
  [Input('graph-dropdown', 'value'), Input('dropdown_menu', 'value'), Input('researcher-dropdown', 'value'), Input('word-search', 'value')],
  State('graph', 'figure')
)
def update_graph(value, topic, author, words, previous_fig):
  if len(previous_fig['data'][0]['node']['color']) != value + num_authors:
    figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

  if len(words) != 0:
    doc_vec = np.zeros((1, len(names)))
    for word in words:
      doc_vec[0][locations[word]] += 1
    relations = np.round(models[f'{value}'].transform(doc_vec), 3).tolist()[0]
    opacity = {(i+num_authors) : relation for i, relation in enumerate(relations) if relation > .1}
    node_colors = ['#666699' if (i not in opacity.keys()) else f'rgba(255, 255, 0, {opacity[i]})' for i in range(len(labels[value]))]
    valid_targets = [positions[value][f'Topic{i-num_authors}'] for i in opacity.keys()]
    link_colors = ['rgba(204, 204, 204, .5)' if target not in valid_targets else f'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
    figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors)),
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]


  if topic == None and author == None:
    figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

  if topic != None and author == None:
    node_colors = ['#666699' if (i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
    link_colors = ['rgba(204, 204, 204, .5)' if target != positions[value][f'Topic{topic}'] else 'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
    figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

  if topic == None and author != None:
    node_colors = ['#666699' if (i != positions[value][author]) else '#ffff00' for i in range(len(labels[value]))]
    link_colors = ['rgba(204, 204, 204, .5)' if source != positions[value][author] else 'rgba(255, 255, 0, .5)' for source in sources[threshold][value]]
    figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

  if topic != None and author != None:
    node_colors = ['#666699' if (i != positions[value][author] and i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
    link_colors = ['rgba(204, 204, 204, .5)' if (source != positions[value][author] or target != positions[value][f'Topic{topic}']) else 'rgba(255, 255, 0, .5)' for source, target in zip(sources[threshold][value], targets[threshold][value])]
    figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
    return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

@app.callback(
  Output('researcher-dropdown', 'value'),
  Input('dropdown_menu', 'value'),
  State('dropdown_menu', 'value')
)
def reset_author(topic, previous):
  if topic != previous:
    return None


if __name__ == '__main__':
    app.run_server(debug=True)