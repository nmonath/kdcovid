import pickle
import time
from string import punctuation

import nltk
import numpy as np
import sent2vec
from absl import app
from absl import flags
from absl import logging
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

FLAGS = flags.FLAGS
flags.DEFINE_string('model_file', '2020-04-10/BioSentVec_PubMed_MIMICIII-bigram_d700.bin', 'model path')
flags.DEFINE_string('out_dir', '2020-04-10/sent2vec/', 'out path')
flags.DEFINE_integer('chunk', 0, 'which chunk')
flags.DEFINE_integer('chunk_size', 2500, 'how many files')
flags.DEFINE_string('all_sections', '2020-04-10/new_all_sections.pkl', 'all sections pickle file')

logging.set_verbosity(logging.INFO)


def load_sents(all_sections, key):
    sents = []
    for sec_id, sec_text in all_sections[key].items():
        if type(sec_text) is str:
            sec_sents = sent_tokenize(sec_text)
            for idx, sent in enumerate(sec_sents):
                sents.append([key, sec_id, idx, sent])
        else:
            logging.info('error: section text is not a string %s %s', key, sec_id)
    return sents


def main(argv):

    logging.info('Running with args %s', str(argv))

    nltk.download('punkt')
    with open(FLAGS.all_sections, 'rb') as fin:
        all_sections = pickle.load(fin)

    model_path = FLAGS.model_file
    chunk = FLAGS.chunk
    chunk_size = FLAGS.chunk_size

    chunk_vecs, chunk_meta = encode(all_sections, model_path, chunk, chunk_size)

    np.save(FLAGS.out_dir + '/chunk_%s.vectors.npy' % FLAGS.chunk, np.vstack(chunk_vecs))

    with open(FLAGS.out_dir + '/chunk_%s.sentences.pkl' % FLAGS.chunk, 'wb') as fout:
        pickle.dump(chunk_meta, fout)

def encode(all_sections, model_path, chunk=0, chunk_size=2500):
    logging.info('loading model...')
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    logging.info('model successfully loaded')

    stop_words = set(stopwords.words('english'))

    chunk_meta = []
    chunk_vecs = []

    sorted_keys = list(all_sections.keys())
    sorted(sorted_keys)

    chunk_keys = sorted_keys[(chunk * chunk_size):((chunk + 1) * chunk_size)]

    logging.info('Running on keys %s...', str(chunk_keys[0:5]))

    def preprocess_sentence(text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()

        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

        return ' '.join(tokens)

    for k_idx, k in enumerate(chunk_keys):
        s_doc = time.time()
        logging.info('key %s (%s of %s) ', k, k_idx, len(chunk_keys))
        sentences = load_sents(all_sections, k)

        dim = 700
        vectors = np.zeros((len(sentences), dim))
        gt = time.time
        t = gt()
        for doc_id, sec_id, sentence_id, s in sentences:
            vectors[sentence_id] = model.embed_sentence(preprocess_sentence(s))
            logging.log_every_n(logging.INFO, 'Processed %s sentences | %s seconds', 10, sentence_id, str(gt() - t))
        e_t = gt()
        logging.info('Done! Processed %s Sentences | %s seconds', len(sentences), str(e_t - t))
        chunk_meta.extend(sentences)
        chunk_vecs.append(vectors)
        e_doc = time.time()
        logging.info('key %s (%s of %s)... %s seconds ', k, k_idx, len(chunk_keys), e_doc - s_doc)
    return chunk_vecs, chunk_meta

def run_encode():
    app.run(main)

if __name__ == "__main__":
    run_encode()