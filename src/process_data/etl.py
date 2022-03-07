import dimcli
from dimcli.utils import *
import sys
import pandas as pd
import time
from tqdm import tqdm

def get_data(faculty_data, key, outpath=None):
    '''
    Retrieve data using dimensions api from the internet
    and return data (or write data to file, if `outpath` is
    not `None`).
    '''
    
    print("==\nLogging in..")
    # https://digital-science.github.io/dimcli/getting-started.html#authentication
    ENDPOINT = "https://app.dimensions.ai"
    if 'google.colab' in sys.modules:
        import getpass
        KEY = getpass.getpass(prompt='API Key: ')
        dimcli.login(key=KEY, endpoint=ENDPOINT)
    else:
        KEY = key
        dimcli.login(key=KEY, endpoint=ENDPOINT)
    dsl = dimcli.Dsl()
    
    # get a list of all faculty members and their dimension ids
    members = pd.read_csv(faculty_data)

    members_withId = members[members['Dimensions ID'].str.startswith('ur')]
    ids = list(members_withId['Dimensions ID'])
    names = list(members_withId['Name'])

    # send request through dimension API
    results = pd.DataFrame()
    for i in tqdm(range(len(ids))):
        query = """search publications
                where year > 2015 and researchers.id="{0}"
                return publications[category_for+concepts+authors+year+id+journal+times_cited+title+abstract]""".format(ids[i])
        result = dsl.query_iterative(query).as_dataframe()
        results = results.append(result.assign(HDSI_author=names[i]))
        break

    results = results.drop_duplicates(subset=['title'])

    # add justin and aaron to the dataset
    justin_aaron = pd.read_csv("./data/raw/justin&aaron.csv")
    ja_df = justin_aaron.assign(HDSI_author = ['Justin Eldridge']*4+['Aaron Fraenkel'])
    final_df = pd.concat([results, ja_df], axis=0, join='outer').reset_index(drop=True)
    
    if outpath is None:
        return final_df
    else:
        # write data to outpath
        final_df.to_csv(outpath)

