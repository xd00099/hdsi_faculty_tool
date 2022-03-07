import pandas as pd
import numpy as np
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
import nltk
import json
nltk.download('wordnet')

NUM_YEARS_TO_INCLUDE = 6

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_abstract(text):
    result = []
    redundant = ['abstract', 'purpose', 'paper', 'goal']
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def get_cleaned_doc_author_info(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data = data.fillna('')
    
    
    stemmer = PorterStemmer()

    data['abstract_processed'] = data['abstract'].apply(preprocess_abstract)
    
    data['year'] = data['year'].astype(int)

    MOST_RECENT = np.max(data['year'])
    YEAR_THRESHOLD = MOST_RECENT - NUM_YEARS_TO_INCLUDE + 1
    data = data[data['year'] >= YEAR_THRESHOLD]
    
    

    # organzie author's abstracts by year
    authors = {}
    for author in data['HDSI_author'].unique():
        authors[author] = {year: [] for year in np.arange(YEAR_THRESHOLD, MOST_RECENT+1)}

    for i, row in data.iterrows():
        authors[row['HDSI_author']][row['year']].append(row['abstract_processed'])

    all_docs = []
    missing_author_years = {author : list() for author in data['HDSI_author'].unique()}
    for author, author_dict in authors.items():
        for year, documents in author_dict.items():
            if len(documents) == 0:
                missing_author_years[author].append(year)
                continue
            all_docs.append(" ".join(documents))
    
    # save the year thresholds
    year_info = {
        'MOST_RECENT': int(MOST_RECENT),
        'YEAR_THRESHOLD': int(YEAR_THRESHOLD),
        'NUM_YEARS_TO_INCLUDE': int(NUM_YEARS_TO_INCLUDE)
    }
    with open('data/output/year_info.json', 'w') as f:
        json.dump(year_info, f)

    return all_docs, authors, missing_author_years, data

def save_article_level_labels(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data = data[data['category_for'].notnull()]

    data.loc[:, 'category_for'] = data.category_for.apply(lambda x: eval(x)) # evaluate '[]' --> []
    data.category_for = data.category_for.apply(lambda x: [i['name'] for i in x])
    data.year = data.year.astype(int)
    df_with_category = data.groupby(['HDSI_author', 'year'])[['category_for']].agg(lambda x: list(set(x.sum()))).reset_index()

    df_with_category.to_csv('data/saved_results/topic_results/article_level_labels.csv')

def save_cleaned_corpus(data_path, output_path_corpus, output_path_authors, output_path_missing_author_years, output_processed_data_path):
    corpus, authors, missing_author_years, processed_data = get_cleaned_doc_author_info(data_path)
    pickle.dump(corpus, open(output_path_corpus, 'wb'))
    pickle.dump(authors, open(output_path_authors, 'wb'))
    pickle.dump(missing_author_years, open(output_path_missing_author_years, 'wb'))
    processed_data.to_csv(output_processed_data_path)
    save_article_level_labels(data_path)