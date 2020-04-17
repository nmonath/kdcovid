from tqdm import tqdm
import nltk

def parse_befree_output(doc2sec2text, disease_output, gene_output, gene_mapping={}):
    '''

    :param doc2sec2text: Data dict
    :param disease_output: Disease links from BeFree
    :param gene_output: Gene links from BeFree
    :param gene_mapping: Uniprot gene mappings
    :return: combined parsed links
    '''
    # calculate sentence starts
    sent_start = {}
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for sha, doc in tqdm(doc2sec2text.items()):
        sent_start_ids = []
        for sec_num, section in doc.items():
            spans = tokenizer.span_tokenize(section)
            for span in spans:
                sent_start_ids.append(span[0])
        sent_start[sha] = sent_start_ids
    disease_links = {sha: {para_id: [] for para_id in doc} for sha, doc in tqdm(doc2sec2text.items())}

    for l in tqdm(disease_output):
        line = l.strip().split('\t')
        sha = line[0]
        par = int(line[5])
        sen = int(line[6])
        cid = line[7]
        loc = line[11]
        sid, eid = loc.split('#')
        sid = int(sid)
        eid = int(eid)
        disease_links[sha][par].append({'start': sent_start[sha][sen] + sid,
                                        'end': sent_start[sha][sen] + eid,
                                        'type': 'disease',
                                        'url': 'https://www.ncbi.nlm.nih.gov/medgen/?term={}'.format(cid)})

    gene_links = {sha: {para_id: [] for para_id in doc} for sha, doc in tqdm(doc2sec2text.items())}

    for l in tqdm(gene_output):
        line = l.strip().split('\t')
        sha = line[0]
        par = int(line[5])
        sen = int(line[6])
        cid = line[7]
        loc = line[11]
        sid, eid = loc.split('#')
        sid = int(sid)
        eid = int(eid)
        if '|' in cid:
            cids = cid.split('|')
            new_pids = [gene_mapping[cidsi] for cidsi in sorted(cids, key=int) if cidsi in gene_mapping]
            if len(set(new_pids)) >= 1:
                url = 'https://www.uniprot.org/uniprot/{}'.format(new_pids[0])
                alt_url = 'https://www.ncbi.nlm.nih.gov/gene/{}'.format(min([int(v) for v in cids]))
            else:
                url = 'https://www.ncbi.nlm.nih.gov/gene/{}'.format(min([int(v) for v in cids]))
                alt_url = url
        else:
            url = 'https://www.uniprot.org/uniprot/{}'.format(
                gene_mapping[cid]) if cid in gene_mapping else 'https://www.ncbi.nlm.nih.gov/gene/{}'.format(cid)
            alt_url = 'https://www.ncbi.nlm.nih.gov/gene/{}'.format(cid)
        gene_links[sha][par].append({'start': sent_start[sha][sen] + sid,
                                     'end': sent_start[sha][sen] + eid,
                                     'type': 'gene',
                                     'url': url,
                                     'alt_url': alt_url})
    combinded_links = {sha: {para_id: [] for para_id in doc} for sha, doc in tqdm(doc2sec2text.items())}
    for sha, doc in tqdm(doc2sec2text.items()):
        for para_id in doc:
            combinded_links[sha][para_id] += gene_links[sha][para_id] + disease_links[sha][para_id]


    return combinded_links

