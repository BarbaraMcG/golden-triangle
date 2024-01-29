from collections import defaultdict
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sentence_transformers import SentenceTransformer

json_dir = 'data/json'

def print_fields_contents():
    all_fields = defaultdict(list)
    for fn in os.listdir(json_dir):
        if fn.endswith('.json'):
            fp = os.path.join(json_dir, fn)        
            with open(fp, 'r') as file:
                data = json.load(file)
                for k,v in data.items():
                    all_fields[k].append({fn: v})

    for k in sorted(all_fields.keys()):
        vs = all_fields[k]
        print(f'# {k}')
        for v in vs:
            if k != 'authorships':
                print(v)
            elif k == 'abstract_inverted_index':
                assert len(v) != 0
            else:
                ak,av = list(v.items())[0]
                print("{'" + ak + "': " + str(len(av)) + '}')
        print()

# print_fields_contents()

# see https://docs.openalex.org/api-entities/works/work-object for description of the fields
fields = ['title', 'display_name', 'authorships', 'cited_by_api_url', 'concepts', 
          'grants', 'publication_date', 'publication_year', 'referenced_works', 'related_works', 'abstract_inverted_index']


def re_invert_abstract(inverted_index):
    if inverted_index == None:
        print('Abstract None')
        return ''
    
    max_index = max([index for indices in inverted_index.values() for index in indices])
    text_list = [""] * (max_index + 1)
    for word, indices in inverted_index.items():
        for index in indices:
            text_list[index] = word
    text = " ".join(text_list)
    return text

test_inverted_index = {
    "The":[0],"MuSe":[1],"(Music":[2],"Sentiment)":[3],"dataset":[4,59],
    "contains":[5],"sentiment":[6],"information":[7],"for":[8,14,31],
    "90,001":[9],"songs.":[10],"We":[11],"computed":[12],"scores":[13],
    "the":[15,25,58],"affective":[16],"dimensions":[17],"of":[18],
    "valence,":[19],"dominance,":[20],"and":[21,42,45,49],"arousal,":[22],
    "based":[23],"on":[24],"user-generated":[26],"tags":[27],"that":[28],
    "are":[29],"available":[30],"each":[32],"song":[33],"via":[34],
    "Last.fm.":[35],"In":[36],"addition,":[37],"we":[38],"provide":[39],
    "artist,":[40],"title":[41],"genre":[43],"metadata,":[44],"a":[46],
    "MusicBrainz":[47],"ID":[48],"a<br":[50],"/>Spotify":[51],"ID,":[52],
    "which":[53],"allow":[54],"researchers":[55],"to":[56],"extend":[57],
    "with":[60],"further":[61],"metadata.":[62]
}

# print(re_invert_abstract(test_inverted_index))

links_df = pd.read_csv('data/links.csv')
all_data_papers = set(links_df['data_paper_doi'])
all_research_papers = set(links_df['research_paper_doi'])

df = []
for fn in os.listdir(json_dir):
    if fn.endswith('.json'):
        fp = os.path.join(json_dir, fn)
        
        with open(fp, 'r') as file:
            for doi in [
                fn.removesuffix('.json').replace('_', '/', 1),
                fn.removesuffix('.json').replace('_', '/')]:
            
                if doi in all_data_papers:
                    paper_type = 'data_paper'
                    break
                elif doi in all_research_papers:
                    paper_type = 'research_paper'
                    break
            else:
                raise AssertionError()
            
            data = json.load(file)
            title = data['title']
            display_name = data['display_name']
            publication_date = data['publication_date']
            publication_year = data['publication_year']
            authors = [row['author']['id'] for row in data['authorships']]
            concepts = {row['display_name']: row['score'] for row in data['concepts']}
            grants = [row['funder'] for row in data['grants']]
            abstract = re_invert_abstract(data['abstract_inverted_index'])
            # cited_by_api_url, 'referenced_works', 'related_works', 'abstract_inverted_index'
            df.append([doi, paper_type, title, display_name, publication_date, publication_year, authors, 
                       concepts, grants, abstract])
            
            
df = pd.DataFrame(df, columns=['doi', 'paper_type', 'title', 'display_name', 'publication_date', 'publication_year',
                                'authors', 'concepts', 'grants', 'abstract'])

assert len(df[df['display_name'] != df['title']]) == 0
assert len(df[df['publication_year'].isna()]) == 0

# authors features (hashed, 10 dimensional)
# hasher = FeatureHasher(n_features=10, input_type='string')
# author_features = hasher.transform(df['authors'])
# author_features = author_features.toarray()
# author_features_df = pd.DataFrame(author_features, columns=[f'af_{i}' 
#                                                             for i in range(author_features.shape[1])])

# concepts features (one hot, about 800 dimensional)
# all_concepts = []
# for concepts in df['concepts']:
#    all_concepts.extend(concepts.keys())
# concepts_dict = {k: i for i,k in enumerate(set(all_concepts))}
# concepts_df = []
# for index, row in df.iterrows():
#    new_row = {concept: np.nan for concept in concepts_dict.keys()}
#    new_row.update(row['concepts'])
#    concepts_df.append(new_row)
# concepts_df = pd.DataFrame(concepts_df, columns=concepts_dict.keys())

# df.drop(['authors', 'concepts', 'display_name', 'publication_year'], axis=1)
# df = pd.concat([df,concepts_df,author_features_df], axis=1)
# df.info()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

df2 = pd.DataFrame()
df2['doi'] = df['doi']
df2['title'] = df['title']
df2['paper_type'] = df['paper_type']
df2['concepts'] = df['authors']
df2['authors'] = df['authors']
df2['publication_date'] = df['publication_date']

embeddings = model.encode(df['title'].tolist(), convert_to_tensor=True)
df2['title_sbert_embeddings'] = embeddings.cpu().numpy().tolist()

df2.to_csv('data/paper-features.csv')