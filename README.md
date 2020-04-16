# KDCOVID

![Overview](http://kdcovid.nl/kd_overview.png)

## About

This tool retrieves papers by measuring similarity between queries and
sentences in the full text of papers in CORD19 corpus using a similarity
metric derived from BioSentVec. We also use the tool BeFree for entity
linking \[1,2\]. Genes are linked to UniProt and diseases to medgen.
We display relationships between genes and diseases from DisGeNET (v6) \[3\].
Highlighting formatting is based on displaCy.

This tool was developed by:
* [Manzil Zaheer](http://www.manzil.ml/) (Google)
* [Nicholas Monath](https://people.cs.umass.edu/~nmonath/) (UMass Amherst)
* [Shehzaad Dhuliawala](https://people.cs.umass.edu/~sdhuliawala/) (Microsoft Research, Montreal)
* [Taamannae Taabassum](https://taamannae.dev/) (University of Toronto)
* [Rajarshi Das](http://rajarshd.github.io/) (UMass Amherst)
* [Bhuwan Dhingra](http://www.cs.cmu.edu/~bdhingra/) (CMU)
* [Andrew McCallum](https://people.cs.umass.edu/~mccallum/) (UMass Amherst).

## Overview of our system

We build our tool based on the (36K) research papers in the COVID-19
Open Research Dataset (CORD-19). It retrieves papers by measuring
similarity between queries and sentences in the full text of papers
in CORD19 corpus. We also extract the bio-medical entities (diseases,
genes, drugs) and link them to various ontologies. Lastly we also
display various gene-disease association between the retrieved entities.
This tool is a work-in-progress and below we provide a simple
description of different components of our system.

### Document Preprocessing

We first preprocess the corpus by splitting the text of research papers
into sentences using the sentence tokenizer in nltk.

![Sentence Segmentation](http://kdcovid.nl/fig1.jpg)

### Retrieving relevant evidence for a query

Given a query, our tool retrieves papers and highlights relevant
sentences within it. We do this by assigning a score to all sentences in
the corpus. The score reflects the similarity of the sentence w.r.t the
query. We also assign score to each individual research paper as the
maximum relevance score of a sentence in the paper. Our tool also
displays context surrounding the high scoring sentence as it might
contain relevant information as well. Next, we briefly describe how we
score each sentence w.r.t a given query.

### Sentence relevance using BioSentVec

The relevance of each sentence is computed as the inner product between the
sentence and the query vector. We create vector representation of each
sentence using a pre-trained BioSentVec model \[4\]. The BioSentVec model
is a sent2vec model \[5\] which has been trained on the PubMed
corpus and outputs a sentence embedding by averaging the embeddings of
the words in them. Although simple, sent2vec has shown to be effective
for many downstream applications. We compute all sentences vectors and
store them offline. Given a query, we first compute its distributed
representation and then find the K-nearest sentence vectors (i.e. the
top-K scoring sentences) from the corpus. As mentioned before, we score
a document as the maximum score of any sentence within it. The figure
below summarizes our system.

![Retrieval](http://kdcovid.nl/fig2.jpg)

### Extracting Knowledge Graphs (KGs) from the evidence

The main aim for building this tool is to help bio-medical researchers
find relevant information. However, it is unlikely that all the answers
that they are looking for would be answered by our initial set of
retrieved papers. Often times, the initial retrieved evidence provides
a good starting point for retrieving more evidence (e.g. think about
those instances where you click an anchor link in a wikipedia page to
go into another wiki page). We hypothesize that the bio-medical entities
(viruses, genes, drugs etc) which are mentioned in the retrieved
research papers would be useful for the researchers. Therefore, we
identify the entities present in the papers, and link them to
knowledge bases from which the researchers can browse further
information. We also display relationship edges between the entities
(e.g. associations between genes and diseases). Below we briefly
summarize each component:

#### Identifying entities in text

The first step is to identify the entities that are present in the
retrieved papers. We use a simple approach of fuzzy string matching
to match to a dictionary containing various entity names and their
aliases. The dictionary is derived from the BeFree system. After
finding the entities, we link the genes to the UniProt and diseases
to the MedGen ontologies. We are also adding entity linking drugs
to Drugbank soon.

#### Finding relations between entities

There are several knowledge bases (KBs) that capture relation
between entities. These KBs are either manually curated from
the findings of several research papers over many years or have
been automatically inferred from text using various relation
extraction approaches. Finding and displaying these relations
in our tool will allow researchers to get insights about findings
that are not restricted to the retrieved paper. For our tool, we use
the gene-disease association present in DisGeNET. We are working on
adding relations between genes, diseases and drugs as well.
The overview figure above summarizes this component.

#### References

[1] À. Bravo, M. Cases, N. Queralt-Rosinach, F. Sanz, and L. I. Furlong, "A Knowledge-Driven Approach to Extract Disease-Related Biomarkers from the Literature", BioMed Research International, vol. 2014, Article ID 253128, 11 pages, 2014. doi:10.1155/2014/253128. (Article, for the "Big Data and Network Biology" special issue at BioMed Research International).

[2] À. Bravo, J. Piñero, N. Queralt, M. Rautschka and L.I. Furlong, "Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research". BMC Bioinformatics 2015, Article, doi:10.1186/s12859-015-0472-9.

[3] Janet Piñero, Juan Manuel Ramírez-Anguita, Josep Saüch-Pitarch, Francesco Ronzano, Emilio Centeno, Ferran Sanz, Laura I Furlong. The DisGeNET knowledge platform for disease genomics: 2019 update. Nucl. Acids Res. (2019) doi:10.1093/nar/gkz1021

[4] Chen Q, Peng Y, Lu Z. BioSentVec: creating sentence embeddings for biomedical texts. The 7th IEEE International Conference on Healthcare Informatics. 2019.

[5] Prakhar Gupta, Matteo Pagliardini, Martin Jaggi Better Word Embeddings by Disentangling Contextual n-Gram Information. NAACL 2019

## Setup

### Install

```
pip install cython
pip install numpy
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install git+git://github.com/nmonath/kdcovid.git
```


### Download Data Required to Run Server

Coming soon.

### Recreate Server Data

Download the zipped data from the [Kaggle competition](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
and place it in the `2020-04-10` folder.

Set up the corpus by running:

```bash
sh bin/run_setup_corpus.sh
```

If you have a slurm-based system, you can use ```sh bin/launch_setup_corpus.sh```.

Then encode all sentences using:

```bash
for i in {0..15}
do
    sh bin/run_encode_sentences.sh $i
done

# If you have slurm:
for i in {0..15}
do
    sh bin/launch_encode_sentences.sh $i
    sleep 1
done
```

Gather the chunks of sentences into single pickles:

```bash
sh bin/run_gather_sentence_embeddings.sh
```

or ```sh bin/launch_encode_sentences.sh```.


