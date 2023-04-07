import pickle
import sys
from src.model.lda import train_lda_5k_dash
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
import dash
# from dash import dcc
# from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

with open('data/output/year_info.json', 'r') as f:
    MOST_RECENT, YEAR_THRESHOLD, NUM_YEARS_TO_INCLUDE = json.load(f).values()

def prepare_sankey(data_processed_path, data_raw_path, missing_author_years_path, corpus_path, authors_path, models_path, results_path, labels_path, sankey_output_folder, num_topics_list = [5,10,15,20,30]):
    # run 5k models
    # models, results, topic_labels = train_lda_5k_dash(corpus_path, authors_path, models_path, results_path, labels_path, missing_author_years_path, num_topics_list)
    models = pickle.load(open(models_path, 'rb'))
    results = pickle.load(open(results_path, 'rb'))
    topic_labels = pickle.load(open(labels_path, 'rb'))
    data = pd.read_csv(data_processed_path)
    data = data.fillna('')

    all_docs = pickle.load(open(corpus_path, 'rb'))
    authors = pickle.load(open(authors_path, 'rb'))
    missing_author_years = pickle.load(open(missing_author_years_path, 'rb'))

    countVec = CountVectorizer()
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()

    topicnames = {
        num_topics : ["Topic" + str(i) for i in range(num_topics)] for num_topics in num_topics_list
    }

    # index names
    docnames = ["Doc" + str(i) for i in range(len(all_docs))]

    # Make the pandas dataframe
    df_document_topic = {
        num_topics : pd.DataFrame(results[f'{num_topics}'], columns=topicnames[num_topics], index=docnames) for num_topics in num_topics_list
    }

    # Get dominant topic for each document
    dominant_topic = {
        num_topics : np.argmax(df_document_topic[num_topics].values, axis=1) for num_topics in num_topics_list
    }

    for num_topics, df in df_document_topic.items():
        df['dominant_topic'] = dominant_topic[num_topics]
   
    author_list = []
    year_list = []
    for author in authors.keys():
        for i in range(NUM_YEARS_TO_INCLUDE):
            if (YEAR_THRESHOLD + i) not in missing_author_years[author]:
                author_list.append(author)
                year_list.append(YEAR_THRESHOLD+ i)

    for df in df_document_topic.values():
        df['author'] = author_list
        df['year'] = year_list

    averaged = {
        num_topics : df_document_topic[num_topics].groupby('author').mean().drop(['dominant_topic', 'year'], axis=1) for num_topics in df_document_topic.keys()
    }

    filtered = {
        threshold : {num_topics : averaged[num_topics].mask(averaged[num_topics] < threshold, other=0) for num_topics in averaged.keys()} for threshold in [.1]
    }
    
    labels = {}
    for num_topics in num_topics_list:
        labels[num_topics] = filtered[.1][num_topics].index.to_list()
        labels[num_topics].extend(filtered[.1][num_topics].columns.to_list())


    sources = {threshold : {} for threshold in [.1]}
    targets = {threshold : {} for threshold in [.1]}
    values = {threshold : {} for threshold in [.1]}

    num_authors = len(np.unique(author_list))

    for threshold in [.1]:
        for num_topics in num_topics_list:
            curr_sources = []
            curr_targets = []
            curr_values = []
            index_counter = 0
            for index, row in filtered[threshold][num_topics].iterrows():
                for i, value in enumerate(row):
                    if value != 0:
                        curr_sources.append(index_counter)
                        curr_targets.append(num_authors + i)
                        curr_values.append(value)
                index_counter += 1
            sources[threshold][num_topics] = curr_sources
            targets[threshold][num_topics] = curr_targets
            values[threshold][num_topics] = curr_values


    positions = {
        num_topics : {label : i for i, label in enumerate(labels[num_topics])} for num_topics in averaged.keys()
    }

    def split_into_ranks(array):
        ranks = []
        for value in array:
            for i, percentage in enumerate(np.arange(.1, 1.1, .1)):
                if value <= np.quantile(array, percentage):
                    ranks.append(i + 1)
                    break
        return ranks

    final_values = {threshold : {} for threshold in [.1]}

    for threshold in [.1]:
        for num_topics in num_topics_list:
            curr_values_array = np.array(values[threshold][num_topics])
            final_values[threshold][num_topics] = split_into_ranks(curr_values_array)

    def display_topics_list(model, feature_names, no_top_words):
        topic_list = []
        for topic_idx, topic in enumerate(model.components_):
            topic_list.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return topic_list

    link_labels = {}
    for num_topics in num_topics_list:
        link_labels[num_topics] = labels[num_topics].copy()
        # link_labels[num_topics][num_authors:] = ['Topic key words: '+a+'<br> Categories: '+b for a,b in zip(display_topics_list(models[str(num_topics)], names, 10), topic_labels[num_topics])]
        link_labels[num_topics][num_authors:] = ['Categories: '+b for a,b in zip(display_topics_list(models[str(num_topics)], names, 10), topic_labels[num_topics])]

    
    counts = CountVectorizer().fit_transform(data['abstract_processed'])
    transformed_list = []
    for model in models.values():
        transformed_list.append(model.transform(counts))


    dataframes = {threshold : {} for threshold in [.1]}
    for i, matrix in enumerate(transformed_list):
        for threshold in [.1]:
            df = pd.DataFrame(matrix)
            df.mask(df < threshold, other=0, inplace=True)
            df['author'] = data['HDSI_author']
            df['year'] = data['year']
            df['citations'] = data['times_cited'] + 1

            # noralization of citations: Scaling to a range [0, 1]
            df['citations_norm'] = df.groupby(by=['author', 'year'])['citations'].apply(lambda x: (x-x.min())/(x.max()-x.min()))#normalize_by_group(df=df, by=['author', 'year'])['citations']
            df['abstract'] = data['abstract']
            df['title'] = data['title']
            df.fillna(1, inplace=True)
            
            #alpha weight parameter for weighting importance of citations vs topic relation
            alpha = .75

            for topic_num in range(num_topics_list[i]):
                df[f'{topic_num}_relevance'] = alpha * df[topic_num] + (1-alpha) * df['citations_norm']
            dataframes[threshold][num_topics_list[i]] = df

    def create_top_list(data_frame, num_topics, threshold):
        top_5s = []
        the_filter = filtered[threshold][num_topics]
        for topic in range(num_topics):
            relevant = the_filter[the_filter[f'Topic{topic}'] != 0].index.to_list()
            to_append = data_frame[data_frame[f'{topic}_relevance'] > 0].reset_index()
            to_append = to_append[to_append['author'].isin(relevant)].reset_index()
            top_5s.append(to_append) 
        return top_5s

    tops = {
        threshold : {num_topics : create_top_list(dataframes[threshold][num_topics], num_topics, threshold) for num_topics in num_topics_list} for threshold in [.1]
    }
 
    # sankey diagrams for diff numbers of topics

    heights = dict(zip(num_topics_list, [2000]*5))
    
    
    figs = {threshold : {} for threshold in [.1]}
    for threshold in [.1]:
        for num_topics in num_topics_list:
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = 'black', width = 0.5),
                    label = labels[num_topics],
                    color = ['#666699' for i in range(len(labels[num_topics]))],
                    customdata = link_labels[num_topics],
                    hovertemplate='%{customdata} Total Flow: %{value}<extra></extra>'
                ),
                link = dict(
                    color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][num_topics]))],
                    source = sources[threshold][num_topics],
                    target = targets[threshold][num_topics],
                    value = final_values[threshold][num_topics]
                )
            )])
            fig.update_layout(title_text="Author Topic Connections", font=dict(size = 10, color = 'white'), height=heights[num_topics], paper_bgcolor="black", plot_bgcolor='black')
            figs[threshold][num_topics] = fig

    ### Add topic labels to the top_words

    # top_words = dict(zip
    #     (num_topics_list, 
    #         [
    #             [
    #                 a+' ------Categories: '+b for a,b in zip(display_topics_list(models[str(i)], names, 10), topic_labels[i])
    #             ] for i in num_topics_list
    #         ]
    #     )
    # )

    top_words = dict(zip
        (num_topics_list, 
            [
                [
                    b + ' <-----> ' + a for a,b in zip(display_topics_list(models[str(i)], names, 10), topic_labels[i])
                ] for i in num_topics_list
            ]
        )
    )


    locations = {}
    for i, word in enumerate(names):
        locations[word] = i
    
    
    pickle.dump(figs, open(sankey_output_folder+'figs.pkl', 'wb'))
    pickle.dump(tops, open(sankey_output_folder+'tops.pkl', 'wb'))
    pickle.dump(author_list, open(sankey_output_folder+'author_list.pkl', 'wb'))
    pickle.dump(labels, open(sankey_output_folder+'labels.pkl', 'wb'))
    pickle.dump(positions, open(sankey_output_folder+'positions.pkl', 'wb'))
    pickle.dump(sources, open(sankey_output_folder+'sources.pkl', 'wb'))
    pickle.dump(targets, open(sankey_output_folder+'targets.pkl', 'wb'))
    pickle.dump(top_words, open(sankey_output_folder+'top_words.pkl', 'wb'))
    pickle.dump(locations, open(sankey_output_folder+'locations.pkl', 'wb'))
    pickle.dump(models, open(sankey_output_folder+'models.pkl', 'wb'))
    pickle.dump(names, open(sankey_output_folder+'names.pkl', 'wb'))
    pickle.dump(num_authors, open(sankey_output_folder+'num_authors.pkl', 'wb'))
    combined = pd.read_csv(data_raw_path)
    pickle.dump(combined, open(sankey_output_folder+'combined.pkl', 'wb'))

