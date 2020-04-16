
import pickle

from absl import flags
from absl import app
from absl import logging
import time
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('sent2vec_dir', '2020-04-10/sent2vec/', 'out path')
flags.DEFINE_integer('num_chunks', 36, 'how many files')
flags.DEFINE_string('out_dir', '2020-04-10/', 'out path')

logging.set_verbosity(logging.INFO)

def load_all_vectors(num_chunks):
    all_vectors = []
    meta_data = []  # (doc_id, section_id, sentence_id, sentence)
    for chunk_id in range(num_chunks):
        logging.info('Processing file %s', chunk_id)
        t = time.time()
        vectors = np.load(FLAGS.sent2vec_dir + '/chunk_%s.vectors.npy' % chunk_id).astype(np.float32)
        with open(FLAGS.sent2vec_dir + '/chunk_%s.sentences.pkl' % chunk_id, 'rb') as fin:
            meta = pickle.load(fin)

        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vector_norms[vector_norms == 0] = 1.0
        vectors /= vector_norms
        all_vectors.append(vectors)
        meta_data.extend(meta)
        e = time.time()

        logging.info('Finished processing chunk %s in %s seconds', chunk_id, str(e-t))
    all_vec = np.concatenate(all_vectors)
    logging.info('Concatenated shape %s' % str(all_vec.shape))
    return all_vec, meta_data


def main(argv):
    logging.info('Running reduce vecs with args %s', str(argv))
    logging.info('Running on %s files', str(FLAGS.num_chunks))
    all_vecs, all_meta = load_all_vectors(FLAGS.num_chunks)
    np.save('%s/all.npy' % FLAGS.out_dir, all_vecs)
    with open('%s/all.pkl' % FLAGS.out_dir, 'wb') as fout:
        pickle.dump(all_meta, fout)


if __name__ == "__main__":
    app.run(main)
