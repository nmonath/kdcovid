import csv
import pickle
import time
from string import punctuation

import re
import numpy as np
import sent2vec
import torch
from absl import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from spacy import displacy
from dateutil import parser as dateparser

logging.set_verbosity(logging.INFO)

DEFAULT_DATE = "2019"

covid_strings = ["covid-19", "covid19", "covid", "sars-cov-2",
                 "sars-cov2", "sarscov2", "novel coronavirus",
                 "2019-ncov", "2019ncov"]
patterns = [re.compile(s, re.IGNORECASE) for s in covid_strings]

def _check_covid(paper):
  if any(re.search(p, paper["title"] + " " + paper["abstract"])
         for p in patterns):
    return True
  return False

class SearchTool(object):

    def __init__(self, data_dir='./', use_cached=False, paper_id_field='cord_uid', all_vecs=None, all_meta=None,
                 model=None, metadata_file=None, documents=None, entity_links=None, cached_result_file=None):
        t_start = time.time()
        self.cached_results = None
        if use_cached:
            with open('%s/cached_results.pkl' % data_dir, 'rb') as fin:
                self.cached_results = pickle.load(fin)
        elif all_vecs is not None:
            self.all_vecs = all_vecs
            self.all_meta = all_meta
            self.model = model
            t = time.time()
            logging.info('Loading Paper Meta Data...')
            self.paper_id_field = paper_id_field
            self.paper_index = {}
            num_covid = 0
            with open(metadata_file) as f:
                reader = csv.DictReader(f, delimiter=',')
                for paper in reader:
                    self.paper_index[paper[self.paper_id_field]] = paper
                    self.paper_index[paper[self.paper_id_field]]['covid'] = _check_covid(paper)
                    try:
                        self.paper_index[paper[self.paper_id_field]]['date'] = dateparser.parse(paper['publish_time'])
                    except:
                        self.paper_index[paper[self.paper_id_field]]['date'] = dateparser.parse(DEFAULT_DATE)

                    try:
                        num_covid += int(self.paper_index[paper[self.paper_id_field]]['covid'])
                    except:
                        pass
                logging.info("Found %d covid papers from %d total" %
                             (num_covid, len(self.paper_index)))
            logging.info('Loading Paper Meta Data...Done! %s seconds' % (time.time() - t))
            self.doc2sec2text = documents
            self.entity_links = entity_links
            if cached_result_file is not None:
                with open('%s/cached_results.pkl' % data_dir, 'rb') as fin:
                    self.cached_results = pickle.load(fin)
        else:
            logging.info('Using data dir %s', data_dir)
            self.data_dir = data_dir

            t = time.time()
            logging.info('Loading Paper Meta Data...')
            self.paper_id_field = paper_id_field
            self.paper_index = {}
            num_covid = 0
            with open('%s/metadata.csv' % data_dir) as f:
                reader = csv.DictReader(f, delimiter=',')
                for paper in reader:
                    self.paper_index[paper[self.paper_id_field]] = paper
                    self.paper_index[paper[self.paper_id_field]]['covid'] = _check_covid(paper)
                    try:
                        self.paper_index[paper[self.paper_id_field]]['date'] = dateparser.parse(
                            paper['publish_time'])
                    except:
                        self.paper_index[paper[self.paper_id_field]]['date'] = dateparser.parse(DEFAULT_DATE)
                    num_covid += int(self.paper_index[paper[self.paper_id_field]]['covid'])
                logging.info("Found %d covid papers from %d total" %
                             (num_covid, len(self.paper_index)))
            logging.info('Loading Paper Meta Data...Done! %s seconds' % (time.time() - t))

            t = time.time()
            logging.info('Loading section text...')
            with open('%s/all_sections.pkl' % data_dir, 'rb') as fin:
                self.doc2sec2text = pickle.load(fin)
            logging.info('Loading section text...Done! %s seconds' % (time.time() - t))

            t = time.time()
            logging.info('Loading entity links...')
            with open('%s/combined_links.pickle' % data_dir, 'rb') as fin:
                self.entity_links = pickle.load(fin)
            logging.info('Loading entity links...Done! %s seconds' % (time.time() - t))

            logging.info('Loading sentence vectors...')
            t = time.time()
            self.all_vecs = np.load('%s/all.npy' % data_dir)

            with open('%s/all.pkl' % data_dir, 'rb') as fout:
                self.all_meta = pickle.load(fout)

            logging.info("%s", self.all_meta[0:5])
            logging.info('Loading sentence vectors... done! %s seconds' % (time.time() - t))

            logging.info('Unit norming the vectors')
            t = time.time()
            norms = np.linalg.norm(self.all_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.all_vecs /= norms
            self.all_vecs = torch.from_numpy(self.all_vecs).detach()
            logging.info('Done unit norming the vectors! %s seconds' % (time.time() - t))

            t = time.time()
            logging.info('Loading BioSentVec Model...')
            model_path = '%s/BioSentVec_PubMed_MIMICIII-bigram_d700.bin' % data_dir
            self.model = sent2vec.Sent2vecModel()
            try:
                self.model.load_model(model_path)
            except Exception as e:
                print(e)
            logging.info('Loading BioSentVec Model... done! %s seconds' % (time.time() - t))

        self.colors = {'Highlight': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'disease': '#ffe4b5', 'gene': '#ffa07a'}
        self.stop_words = set(stopwords.words('english'))
        logging.info('Finished setting up constructor in %s seconds' % (time.time() - t_start))

    def preprocess_sentence(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()

        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in self.stop_words]

        return ' '.join(tokens)

    def knn(self, query_vectors, base_vectors, query_metadata, base_metadata, batch_size=1000, K=200):
        t = time.time()
        nn = dict()
        for i in range(0, query_vectors.shape[0], batch_size):
            topk = torch.topk(torch.matmul(query_vectors[i:(i + batch_size)], base_vectors.transpose(1, 0)), k=K, dim=1)
            distances, indices = topk[0].cpu().numpy(), topk[1].cpu().numpy()
            for j in range(distances.shape[0]):
                qr_key = query_metadata[i][-1]
                nn[qr_key] = [{'doc_id': base_metadata[x][0].replace('.json', ''), 'sent_text': base_metadata[x][3],
                               'sent_no': base_metadata[x][2],
                               'sec_id': base_metadata[x][1], 'sim': distances[j, idx]} for idx, x in
                              enumerate(indices[j])]
            logging.info('Finished % out of %s in %s', i, query_vectors.shape[0], time.time() - t)
            del topk
            del distances
            del indices
        logging.info('Done! %s', time.time() - t)
        return nn

    def get_entity_base(self, color, link):
        return """
                <mark class="entity" style="background: {bg}; padding: 0.15em 0.15em; margin: 0 0.25em; line-height: 1.5; border-radius: 0.15em">
                    {text}
                    <span style="font-size: 0.8em; font-weight: bold; line-height: 1.5; border-radius: 0.15em; text-transform: uppercase; vertical-align: middle; margin-right: 0.15rem"><a href="{link}" target="_blank" style="text-decoration: none;  color: black;">{label}</a></span>
                </mark>
                """.replace("{bg}", color).replace("{link}", link)

    def get_entity_string(self, text, color, label, link):
        return self.get_entity_base(color, link).replace("{text}", text).replace("{label}", label)

    def get_highlight_base(self, color):
        return """
                <mark class="entity" style="background: {bg}; padding: 0.15em 0.15em; margin: 0 0.25em; line-height: 1.5; border-radius: 0.15em">
                    {text}
                    <span style="font-size: 0.8em; font-weight: bold; line-height: 1.5; border-radius: 0.15em; text-transform: uppercase; vertical-align: middle; margin-right: 0.15rem">{label}</span>
                </mark>
                """.replace("{bg}", color)

    def get_highlight_string(self, text, color, label):
        return self.get_highlight_base(color).replace("{text}", text).replace("{label}", label)

    def highlight_texts(self, larger_text, entities, highlights, colors):
        # spans = [(start, end, link, type), ...]
        entities = sorted(entities, key=lambda x: (x[0], x[1]))
        highlights = sorted(highlights, key=lambda x: (x[0], x[1]))
        while entities and highlights:
            last_e = entities[-1]
            last_h = highlights[-1]
            # if e is candidate and is not overlapping
            if last_e[0] > last_h[1]:
                l = """<i class="fa">&#xf08e;</i>"""
                s, e, t, link = last_e
                color = colors[t]
                rs = self.get_entity_string(larger_text[s:e], color, l, link)
                larger_text = larger_text[:s] + rs + larger_text[e:]
                entities.pop()
            elif last_e[1] > last_h[0]:
                l = """<i class="fa">&#xf08e;</i>"""
                s, e, t, link = last_e
                color = colors[t]
                rs = self.get_entity_string(larger_text[s:e], color, l, link)
                larger_text = larger_text[:s] + rs + larger_text[e:]
                entities.pop()
                last_h[1] += len(rs) - (e - s)
            else:  # h goes
                l = 'Highlight'
                s, e, t, link = last_h
                color = colors[t]
                rs = self.get_highlight_string(larger_text[s:e], color, l)
                larger_text = larger_text[:s] + rs + larger_text[e:]
                highlights.pop()
        while entities:
            l = """<i class="fa">&#xf08e;</i>"""
            s, e, t, link = entities.pop()
            color = colors[t]
            rs = self.get_entity_string(larger_text[s:e], color, l, link)
            larger_text = larger_text[:s] + rs + larger_text[e:]
        while highlights:
            l = 'Highlight'
            s, e, t, link = highlights.pop()
            color = colors[t]
            rs = self.get_highlight_string(larger_text[s:e], color, l)
            larger_text = larger_text[:s] + rs + larger_text[e:]
        return larger_text

    def h(self, larger_text, smaller_texts):
        ents = []
        for smaller_text in smaller_texts:
            start_offset = larger_text.find(smaller_text)
            ents.append({"start": start_offset, "end": start_offset + len(smaller_text), "label": "HIGHLIGHT"})
        colors = {"HIGHLIGHT": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        ex = [{"text": larger_text,
               "ents": ents,
               "title": None,
               "colors": colors}]
        res = displacy.render(ex, style="ent", manual=True, options={"colors": colors}, page=True, jupyter=False)
        return res

    def format_html(self, sha, title, authors, year_of_publication, link, venue, sentences, sections, section_ids,
                    user_sent):
        try:
            alist = list(csv.reader([authors.strip().replace('[', '').replace(']', '')]))[0]
        except:
            logging.warning('Error parsing author string: %s', authors)
            alist = []
        if len(alist) > 10:
            alist = alist[0:10] + ['et al']
        authors = "; ".join([x.replace('\'', '') for x in alist])
        # s = "<h2><b><a href=\"%s\" rel=\"noopener noreferrer\" target=\"_blank\">%s</a></b></h2><div class=\"wrap\"><div class=\"res_text\"><i>%s</i><BR/>%s<BR/>%s<BR/><BR/>" % (
        # link, title, str(authors), year_of_publication, venue)
        s = '<div class="wrap"><div class="res_text"><div class="paper-details"><span class="year">%s | %s</span><h2><b><a href="%s" rel="noopener noreferrer" target="_blank">%s<i class="fa">&#xf08e;</i></a></b></h2><i>%s</i><BR/><BR/><BR/>' % (
            year_of_publication, venue, link, title, str(authors))
        sec2sent = dict()
        # todo don't copy here...
        secid2sec = dict()
        for sent, sec, sec_id in zip(sentences, sections, section_ids):
            if sec not in sec2sent:
                sec2sent[sec_id] = []
            sec2sent[sec_id].append(sent['sent_text'])
            secid2sec[sec_id] = sec
        for sec_id, sents in sec2sent.items():
            sec = secid2sec[sec_id]
            entity_spans = []
            highlight_spans = []
            for sent in sents:
                start_offset = sec.find(sent)
                if start_offset >= 0:
                    end_offset = start_offset + len(sent)
                    highlight_spans.append([start_offset, end_offset, 'Highlight', None])
            # {sha: {para_id: [ {start: int, end: int, url: string} ] } }
            if sha in self.entity_links:
                entities = self.entity_links[sha][sec_id]
                for ent in entities:
                    entity_spans.append([ent['start'], ent['end'], ent['type'], ent['url']])
            else:
                logging.warning('No links found for document %s', sha)


            s += self.highlight_texts(sec, entity_spans, highlight_spans, self.colors)
        s += '</div><div class="legend"><div><div class="circle yellow"></div><p>Disease</p></div><div><div class="circle orange"></div><p>Gene</p></div><div><div class="circle purple"></div><p>Text Matching Search</p></div></div>'
        s += "</div>"
        #         print(sec)
        #         print(sents)
        # Adding image part
        s += '<div class="res_image"><h2>Gene-Disease Association</h2><p>Click on the gene/disease for more information</p><br><object data="gv_files/{}.gv.svg" type="image/svg+xml"></object><p></p><p class="cite"><br>Graph data from <a href="https://www.disgenet.org">DisGeNET v6.0</a></p></div>'.format(
            sha)
        # s += "<div class=\"res_image\"><img src=\"gv_files/{}.gv.svg\" alt=\"Mini-KB\" width=\"95%\"></div>".format(sha)
        s += "</div>"
        return s

    def get_search_results(self, user_query, sort_by_date=False, covid_only=False, K=100, Kdocs=20):
        if self.cached_results is not None:
            logging.info('getting search results for %s, K=%s, Kdocs=%s', user_query, K, Kdocs)
            return self.cached_results[user_query]

        logging.info('getting search results for %s, K=%s, Kdocs=%s', user_query, K, Kdocs)
        v = self.model.embed_sentence(self.preprocess_sentence(user_query))
        logging.info('starting embedding sentence %s, K=%s, Kdocs=%s', user_query, K, Kdocs)
        query_vecs = torch.from_numpy(v.astype(np.float32))
        logging.info('finished embedding sentence %s, K=%s, Kdocs=%s', user_query, K, Kdocs)
        query_meta = [('query', 0, 0, user_query)]
        logging.info('starting nearest neighbors for %s, K=%s, Kdocs=%s', user_query, K, Kdocs)
        nn = self.knn(query_vecs, self.all_vecs, query_meta, self.all_meta, K=K)
        logging.info('found nearest neighbors for %s, K=%s, Kdocs=%s', user_query, K, Kdocs)

        res = ""
        for sent, nns in nn.items():
            all_results = {}
            for idx, nnv in enumerate(nns):
                if len(nnv["sent_text"].split()) < 5:
                    continue
                sha = nnv['doc_id']
                sha = sha.split(".")[0]
                if covid_only and not self.paper_index[sha]['covid']:
                    continue
                if sha not in all_results:
                    paper_metadata = self.paper_index[sha]
                    score = nnv['sim']
                    sentences = [nnv, ]
                    sections = [self.doc2sec2text[nnv['doc_id']][nnv['sec_id']]]
                    section_ids = [nnv['sec_id']]
                    all_results[sha] = {'paper': paper_metadata, "score": score, "sentences": sentences,
                                        "sections": sections, 'section_ids': section_ids}
                else:
                    all_results[sha]["sentences"].append(nnv)
                    all_results[sha]["sections"].append(self.doc2sec2text[nnv['doc_id']][nnv['sec_id']])
                    all_results[sha]["section_ids"].append(nnv['sec_id'])
            if sort_by_date:
              all_results_sorted = [(sha, all_results[sha]['paper']['date']) for sha in all_results]
            else:
              all_results_sorted = [(sha, all_results[sha]['score']) for sha in all_results]
            all_results_sorted.sort(reverse=True, key=lambda x: x[1])
            for sha, _ in all_results_sorted[0:Kdocs]:
                paper_metadata = all_results[sha]['paper']
                score = all_results[sha]['score']
                sentences = all_results[sha]['sentences']
                sections = all_results[sha]['sections']
                section_ids = all_results[sha]['section_ids']
                title = paper_metadata['title']
                venue = paper_metadata['journal']
                authors = paper_metadata['authors']
                year_of_publication = paper_metadata['publish_time']
                doi = paper_metadata['doi']
                link = "https://doi.org/{}".format(doi)
                if len(title.strip()) > 5:
                    res += self.format_html(sha, title, authors, year_of_publication, link, venue, sentences, sections,
                                            section_ids, user_query)
        return res


    def format_single_page_with_css(self, html_string):
        return """
            <!DOCTYPE html>
            <html>
            <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            <link rel="stylesheet" type="text/css" href="./style.css">
            <link rel="icon" type="image/ico" href="./favicon.ico">
            <link rel="stylesheet" href="https://use.typekit.net/wtq0evn.css">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            html
            {
             overflow: -moz-scrollbars-vertical; 
             overflow-y: scroll;
            }
            
            body
            {
             margin:0 auto;
             text-align:center;
             width:100%;
             padding: 0 0;
             font-family: 'Basic Sans', sans-serif;
             background-color:#fff;
             color: #3E4550;
            }
            #wrapper
            {
             margin:0 auto;
             text-align:center;
            }
            #wrapper h1
            {
             margin-top:50px;
             font-size:45px;
             font-family: 'Basic Sans', sans-serif;
             color:#585858;
            }
            #wrapper h1 p
            {
             font-size:18px;
             font-family: 'Basic Sans', sans-serif;
            }
            
            .q_container {
                display: flex;
            }
            
            .q_container {
                display: flex;
                max-width: 900px;
                margin: auto;
                flex-wrap: wrap;
            }
            
            .q_inner {
                flex: 1;
            }
            
            .q_inner {
                background-color: #fff;
                padding: 10px 20px;
                box-shadow: none;
                border: 2px solid #178EF4;
                margin: 10px;
                color: rgb(0, 61, 114);
                border-radius: 7px;
                cursor: pointer;
                transition: all 0.3s ease;
                align-items: center;
                max-width: 300px;
                display: flex;
                flex: 0 0 calc(33% - 70px);
                text-align: center;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-family: 'Basic Sans', sans-serif;
                /*Added later*/
                -webkit-appearance: none;
                -webkit-rtl-ordering: logical;
                -webkit-writing-mode: unset;
                box-sizing: unset;
                user-select: unset;
                white-space: unset;
                font: unset;
                text-rendering: unset;
                letter-spacing: unset;
                word-spacing: unset;
                text-transform: unset;
                text-indent: unset;
                text-shadow: unset;
            }
            
            .q_inner:hover {
                background-color: #178EF4;
                margin: 10px;
                color: #fff;
                cursor: pointer;
            }
            
            
            #search_box input[type="text"]
            {
             width:450px;
             height:45px;
             padding-left:10px;
             font-size:18px;
             font-family: 'Basic Sans', sans-serif;
             margin-bottom:15px;
             color:#424242;
             border-radius: 7px;
             border: 2px solid #A9B5C7;
            
            }
            
            .search {
                display: flex;
                justify-content: center;
            }
            #search_box input[type="submit"]
            {
             width:100px;
             height:45px;
             background-color:#178EF4;
             color:white;
             border-radius: 7px;
             border:none;
             margin-left: 20px;
             font-size: 1rem;
             text-transform: none;
             cursor: pointer;
             font-family: 'Basic Sans', sans-serif;
             transition: all 0.3s ease;
             outline: none;
            }
            #search_box input[type="submit"]:hover
            {
             transform: scale(1.1);
             background-color: rgb(16, 123, 216);
             
            }
            
            #result_div
            {
             margin: auto;
             text-align:left;
            }
            #result_div li
            {
             margin-bottom:20px;
             list-style-type:none;
            }
            #result_div li a
            {
             text-decoration:none;
             display:block;
             text-align:left;
            }
            #result_div li a .title
            {
             font-weight:bold;
             font-size:18px;
             font-family: 'Basic Sans', sans-serif;
             color:#5882FA;
            }
            #result_div li a .desc
            {
             color:#6E6E6E;
            }
            .topnav {
              background-color: #fff;
              width: 100%;
              overflow: hidden;
            }
            
            .container-inner {
                width: 100%;
                max-width: 1200px;
                margin: auto;
                display: flex;
                align-items: center;
            }
            
            h1 {
                font-size: 1.5rem;
                color: rgb(97, 114, 141);
                margin: 10px;
                margin-bottom: 30px;
                font-weight:400;
            }
            
            .logo img {
                width: 150px;
            }
            
            .container-inner div:nth-child(1) {
                flex: 1;
                text-align: left;
            }
            
            .res_text h2 ~ i {
                color: #0070D0;
                font-size: 14px;
            }
            
            .res_text h2 a {
                color: #3E4550;
                font-weight: 700;
                text-decoration: none;
            }
            
            .year {
                color: #0070D0;
                font-size: 14px;
            }
            
            .res_text h2 {
                margin: 0.5rem 0px ;
            }
            
            .res_image h2 {
                margin: 5px 0px;
                margin-top: 30px;
            }
            .res_image p {
                margin: 0px;
                font-size: 14px;
            
            }
            .jumbotron {
                padding: 100px 20px;
                height: auto;
                padding-bottom: 50px;
            }
            /* Style the links inside the navigation bar */
            .topnav a {
              float: left;
              color: #3E4550;
              text-align: center;
              padding: 14px 16px;
              text-decoration: none;
              font-size: 17px;
            }
            /* Change the color of links on hover */
            .topnav a:hover {
              background-color: #c0e2ff;
              color: black;
            }
            /* Add a color to the active/current link */
            .topnav a.active {
              background-color: #8dd3c7;
              color: white;
            }
            .round {
              stroke-linejoin: round;
              font-family: 'Basic Sans', sans-serif;
            }
            .wrap {
              display: flex;
               width: auto;
                max-width: 1200px;
                margin: auto;
                padding: 40px 20px;
            
            }
            
            .res_text h2 i {
                font-size: 16px;
                margin-left: 10px;
            }
            
            .res_text h2 a {
                transition: all 0.3s ease;
            }
            .res_text h2:hover a {
                color: rgb(103, 114, 131);
            }
            
            .legend {
                border-top: 1px solid #BBC3CE;
                display: flex;
                margin-top: 20px;
                padding: 20px;
            }
            
            .legend div {
                display: flex;
                align-items: center;
            }
            
            .entity {
                padding: 0px 3px !important;
                border-radius: 5rem !important;
            }
            
            .legend p {
                margin: 0 30px 0px 10px;
            }
            .circle {
                width: 15px;
                height: 15px;
                border-radius: 30px;
            }
            
            .yellow {
                background-color: #FDF8B9;
            }
            .orange {
                background-color: #FFA07A;
            }
            .purple {
                background: linear-gradient(90deg, #aa9cfc, #fc9ce7);
            }
            
            .res_text {
              flex: 1;
              display: flex;
              flex-direction: column;
               margin: 0px auto;
               line-height: 1.6rem;
                border: 1px solid #BBC3CE;
                border-top-left-radius: 10px;
                border-bottom-left-radius: 10px;
              /* border: 1px solid green; */
            }
            
            .paper-details {
                padding: 20px;
                flex: 1;
            }
            
            
            
            .j-logo {
                max-width: 700px;
                width: 100%;
            }
            
            .res_image {
              padding: 20px;
              border-top: 1px solid #BBC3CE;
              border-bottom: 1px solid #BBC3CE;
              border-right: 1px solid #BBC3CE;
              border-top-right-radius: 10px;
              border-bottom-right-radius: 10px;
            }
            
            .res_image::-webkit-scrollbar {
                -webkit-appearance: none;
            }
            
            .res_image::-webkit-scrollbar:vertical {
                width: 8px;
            }
            
            .res_image::-webkit-scrollbar:horizontal {
                height: 8px;
            }
            
            .res_image::-webkit-scrollbar-thumb {
                border-radius: 1px;
                border: 1px solid white; /* should match background, can't be transparent */
                background-color: rgba(0, 0, 0, .5);
            }
            
            @media (max-width: 767px) {
              .wrap {
                flex-direction: column;
              }
              .one,
              .two {
                width: auto;
                overflow: -moz-scrollbars-vertical; 
                overflow: scroll;
              }
            
              .res_image object {
                width: 100%;
              }
            
              .res_image {
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
                border-left: 1px solid #BBC3CE;
                border-top: none;
                border-top-right-radius: 0px;
              }
            
              .res_text {
                  border-top-right-radius: 10px;
                  border-bottom-right-radius: 0px;
                  border-bottom-left-radius: 0px;
              }
            
              .q_inner {
                  flex: 0 0 calc(50% - 70px);
              }
            }
            @media (max-width: 464px) {
              .q_container {
                  flex-direction: column;
              }
            
              .q_inner {
                  flex: 1;
                  max-width: none;
              }
            }


            .collapsible {
              background-color: white;
              color: #3E4550;
              border-radius: 7px;
              cursor: pointer;
              padding: 18px;
              text-align: center;
              width: 100%
              outline: none;
              font-size: 18px;
              border-style: solid;
              border-color: #3E4550;
              font-family: 'Basic Sans', sans-serif;
            }

            .active, .collapsible:hover {
              background-color: rgb(36.1472, 145.5616, 241.9456);
              color: white;
            }


            .content1 {
              padding: 0 18px;
              display: none;
              overflow: hidden;
              background-color: #f1f1f1;
              font-size: 15px;
              font-family: 'Basic Sans', sans-serif;
              text-align: left;
            }

            .content {
              padding: 0 18px;
              display: none;
              overflow: hidden;
              background-color: #f1f1f1;
              font-size: 15px;
              font-family: 'Basic Sans', sans-serif;
              text-align: center;
            }

            </style>
            </head>
            <body>


            CONTENT

            <script>
            var coll = document.getElementsByClassName("collapsible");
            var i;

            for (i = 0; i < coll.length; i++) {
              coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                  content.style.display = "none";
                } else {
                  content.style.display = "block";
                }
              });
            }
            </script>

            </body>
            </html>
            """.replace('CONTENT', html_string)
