import json
import spacy
from tqdm import tqdm
from collections import Counter
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

nlp = None
#nlp = spacy.load("en_core_web_sm")

def get_ents_spacy(text):
    doc = nlp(text)
    return set([ent.text for ent in doc.ents])


nltk.download("punkt")
nltk.download("maxent_ne_chunker")
nltk.download("words")

def get_ents(text):
    """Extract named entities using NLTK."""
    words = word_tokenize(text)  # Tokenize words
    pos_tags = pos_tag(words)  # Part-of-speech tagging
    tree = ne_chunk(pos_tags)  # Named entity recognition

    # Extract named entities
    entities = set()
    for subtree in tree:
        if hasattr(subtree, "label"):  # Check if it's an entity
            entity = " ".join(word for word, _ in subtree.leaves())
            entities.add(entity)
    return entities

def get_all_ents_dict(fn):
    all_ents_dict_dp, all_ents_dict_rp = {}, {}
    with open(fn, 'r') as f:
        for line in tqdm(f):
            if len(all_ents_dict_dp) < 1000 and len(all_ents_dict_dp) == 300:
                break
            d = json.loads(line.strip())
            if d['gt_is_data_journal']:
                if len(all_ents_dict_dp) < 1000:
                    all_ents_dict_dp[id] = get_ents(d['gt_abstract'])
            else:
                if len(all_ents_dict_dp) < 300:
                    all_ents_dict_rp[id] = get_ents(d['gt_abstract'])
    
    counts_dp = Counter(len(v) for v in all_ents_dict_dp.values())
    counts_rp = Counter(len(v) for v in all_ents_dict_rp.values())

    print(f'{counts_dp=}')
    print()
    print(f'{counts_rp=}')

    return all_ents_dict_dp, all_ents_dict_rp

def link_by_ner(all_ents_dict_dp, all_ents_dict_rp, thr=.9):
    # Basic sanity checking
    all_ents_dict_dp = {id: ents for id, ents in all_ents_dict_dp.items() if len(ents) > 5}
    all_ents_dict_rp = {id: ents for id, ents in all_ents_dict_rp.items() if len(ents) > 5}
    
    ids_dp, ids_rp = list(all_ents_dict_dp.keys()), list(all_ents_dict_rp.keys())
    jac_ds = []
    ans = []
    for id_dp in ids_dp:
        ents_dp = all_ents_dict_dp[id_dp]
        for id_rp in ids_rp:
            ents_rp = all_ents_dict_rp[id_rp]
            intersection_len = len(ents_dp & ents_rp)
            if not intersection_len:
                continue
            jac_d = len(ents_dp | ents_rp) / len(ents_dp & ents_rp)
            jac_ds.append(jac_d)
    

if __name__ == '__main__':
    fn = '/scratch_tmp/prj/dh_golden_triangle/filtered_results_2022.jsonl'
    all_ents_dict_dp, all_ents_dict_rp = get_all_ents_dict(fn)
    # link_by_ner(all_ents_dict_dp, all_ents_dict_rp, thr=.9)
            
