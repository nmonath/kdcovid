import csv
import json
import pickle
import time

from absl import app
from absl import flags
from absl import logging
from nltk import sent_tokenize

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file_list', '2020-04-10/file-list', 'data path')
flags.DEFINE_string('outfile', '2020-04-10/all_sections.pkl', 'out path')
flags.DEFINE_string('metadata_file', '2020-04-10/metadata.csv', 'list of inputs')

logging.set_verbosity(logging.INFO)


class DocumentLoader(object):
    def __init__(self, input_file_list, metadata, max_files_processed=None):
        self.max_files_processed = max_files_processed
        input_files = []
        self.metadata_file = metadata
        with open(input_file_list) as fin:
            for line in fin:
                input_files.append(line.strip())
        self.input_files = input_files
        logging.info('loaded %s input files', len(self.input_files))
        self.pmc_files = [x for x in self.input_files if 'PMC' in x]
        self.non_pmc_files = [x for x in self.input_files if 'PMC' not in x]
        self.pmc2cordid = dict()
        self.sha2cordid = dict()
        self.cord2docs = dict()
        self.cord_sha_ordering = dict()
        self.all_sections = dict()
        self.load_meta_data()
        self.load_docs()

    def load_meta_data(self):
        with open(self.metadata_file) as fin:
            metacsv_reader = csv.reader(fin)
            metacsv = [x for x in metacsv_reader]
        header = metacsv[0]
        line_length = len(header)
        for line_no in range(1, len(metacsv)):
            line_splt = metacsv[line_no]
            if len(line_splt) != line_length:
                logging.info('ERROR metacsv line no %s', line_no)
            else:
                cord_uid = line_splt[0].strip()
                shas = line_splt[1].strip()
                pmid = line_splt[5].strip()
                if not pmid.startswith("PMC"):
                    pmid = "PMC" + pmid
                if pmid:
                    logging.log_first_n(logging.INFO, 'Adding pmid -> cord = %s --> %s', 100, pmid, cord_uid)
                    if pmid not in self.pmc2cordid:
                        self.pmc2cordid[pmid] = cord_uid
                    else:
                        logging.warning('[pmid with 2 cords] pmid %s has already been given to %s, this cord is %s' % (
                        pmid, self.pmc2cordid[pmid], cord_uid))
                if shas:
                    shas_split = [x.strip() for x in shas.split(";")]
                    for sha in shas_split:
                        logging.log_first_n(logging.INFO, 'Adding sha -> cord = %s --> %s', 100, sha, cord_uid)
                        if sha not in self.sha2cordid:
                            self.sha2cordid[sha] = cord_uid
                        else:
                            logging.warning(
                                '[sha with 2 cords] sha %s has already been given to %s, this cord is %s' % (
                                    sha, self.sha2cordid[sha], cord_uid))

                    self.cord_sha_ordering[cord_uid] = [s.strip() for s in shas_split]

    def load_docs(self):
        # Load PMC files.
        for pmc_file in self.pmc_files:
            sents, section2text, paper_id = load_sents(pmc_file)
            # try to get coord id
            paper_id_no_pmc = paper_id
            if paper_id_no_pmc in self.pmc2cordid:
                logging.log_first_n(logging.INFO, 'Found PMC id %s', 10, paper_id_no_pmc)
                cord_id = self.pmc2cordid[paper_id_no_pmc]
            else:
                logging.info('No cord id for PMC id %s', paper_id_no_pmc)
                continue

            if cord_id not in self.cord2docs:
                self.cord2docs[cord_id] = dict()
            else:
                logging.warning('Two PMIDs for the same cord %s', cord_id)

            self.cord2docs[cord_id][paper_id] = section2text

            if self.max_files_processed is not None and len(self.cord2docs) > self.max_files_processed:
                break

        # Load non pmc files.
        for non_pmc_file in self.non_pmc_files:
            sents, section2text, paper_id = load_sents(non_pmc_file)
            # try to get coord id
            if paper_id in self.sha2cordid:
                logging.log_first_n(logging.INFO, 'Found sha id %s', 10, paper_id)
                cord_id = self.sha2cordid[paper_id]
            else:
                logging.info('No cord id for sha id %s', paper_id)
                continue

            if cord_id not in self.cord2docs:
                self.cord2docs[cord_id] = dict()
            self.cord2docs[cord_id][paper_id] = section2text

            if self.max_files_processed is not None and len(self.cord2docs) > self.max_files_processed:
                break

        # Reduce into a single pickle
        counter = 0
        for cord_id, doc2sections in self.cord2docs.items():
            logging.info('Processing cord_id %s with %s sections (%s of %s)', cord_id, len(doc2sections), counter, len(self.cord2docs))
            counter += 1
            last_section_offset = 0
            for doc_id, doc_sections in doc2sections.items():
                if 'PMC' in doc_id:
                    if cord_id not in self.all_sections:
                        self.all_sections[cord_id] = dict()
                    for sec_id, sec in doc_sections.items():
                        self.all_sections[cord_id][sec_id + last_section_offset] = sec
                    last_section_offset += len(doc_sections)
            # if we haven't added any PMC files:
            if cord_id not in self.all_sections:
                ordered_shas = self.cord_sha_ordering[cord_id]
                for sha in ordered_shas:
                    if sha in doc2sections:
                        doc_sections = doc2sections[sha]
                        if cord_id not in self.all_sections:
                            self.all_sections[cord_id] = dict()
                        for sec_id, sec in doc_sections.items():
                            self.all_sections[cord_id][sec_id + last_section_offset] = sec
                        last_section_offset += len(doc_sections)
                    else:
                        logging.warning("Missing sha %s for cord %s", sha, cord_id)


def load_sents(filename):
    gt = time.time
    logging.info('Loading from filename %s', filename)
    jobj = json.load(open(filename))

    paper_id = jobj['paper_id']
    logging.info('paper_id %s', jobj['paper_id'])

    sent_id = 0
    sect_id = 0
    sents = []
    section2text = dict()
    t = gt()
    if 'metadata' in jobj and 'title' in jobj['metadata']:
        title = jobj['metadata']['title']
        title = (title, sect_id)
        logging.info('Title: %s', str(title))
        sect_id += 1
        sent_id += 1
        sents.append(title)
        section2text[0] = title[0]
    else:
        logging.info('Missing document title - %s', str(filename))

    if 'abstract' in jobj and len(jobj['abstract']) > 0:
        abstract = [(x['text'], sect_id + idx) for idx, x in enumerate(jobj['abstract'])]
        for text, sect in abstract:
            section2text[sect] = text
        abstract_sentences = [(s, section_id) for x, section_id in abstract for sent_idx, s in
                              enumerate(sent_tokenize(x))]
        logging.info('Last Section Id in Abstract %s', abstract_sentences[-1][1])
        sents.extend(abstract_sentences)
        sect_id = abstract_sentences[-1][1] + 1
        logging.info('Number of sentences so far %s', len(sents))
    else:
        logging.info('Missing document abstract - %s', str(filename))

    if 'body_text' in jobj and len(jobj['body_text']) > 0:
        texts = [(x['text'], idx + sect_id) for idx, x in enumerate(jobj['body_text'])]
        for text, sect in texts:
            section2text[sect] = text
        texts_sentences = [(s, section_id) for x, section_id in texts for sent_idx, s in enumerate(sent_tokenize(x))]
        if texts_sentences:
            sents.extend(texts_sentences)
            logging.info('Last Section Id in Body %s', texts_sentences[-1][1])
            logging.info('Number of sentences so far %s', len(sents))
        else:
            logging.warning('No sentences available in body_text....')
    e_t = gt()
    logging.info('Finished processing document in %s', e_t - t)
    return sents, section2text, paper_id


def setup_corpus(argv):
    logging.info('Running with args %s', str(argv))
    doc_loader = DocumentLoader(FLAGS.input_file_list, FLAGS.metadata_file)
    all_section2text = doc_loader.all_sections

    with open(FLAGS.outfile, 'wb') as fout:
        pickle.dump(all_section2text, fout)


def run_setup():
    app.run(setup_corpus)


if __name__ == "__main__":
    run_setup()
