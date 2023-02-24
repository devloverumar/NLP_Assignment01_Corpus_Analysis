# simple tokenization/normalization 

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import re
email_pat = re.compile(r"\S+@\S+\.\S+")
url_pat = re.compile("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")

            
# document is a string containing 1 or more sentences
# returns a list of all of the tokens in the document
def tokenize(document, skip_header = False):
    doc_tokens = []
    if skip_header:
        document = document.split('\n\n',1)[1]
    # use nltk sentence tokenization
    sentences = nltk.sent_tokenize(document)
    for sentence in sentences:
        # use nltk word tokenization
        # remove email addresses
        sentence = re.sub(email_pat,'',sentence)
        sentence = re.sub(url_pat,'',sentence)
        sent_tokens = nltk.word_tokenize(sentence)
        # remove punctuation
        sent_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent_tokens]
        # lowercase and remove empty strings, stopwords, and numbers (all punctuation will become empty after previous line)
        sent_tokens = [word.lower() for word in sent_tokens if word]

        sent_tokens = ([word for word in sent_tokens if 
                            word not in stopwords 
                            #and word in vocab
                            and not re.search('\d+',word)
                            and len(word) > 2])
        # either use char ngrams or full words
        doc_tokens += sent_tokens
    return doc_tokens

tokenize("header\n\nthis is a test with darwinsomething@ics.edu  and cantelopes@csc.smu.edu in it and some words like computers and baseball the.")

import collections
import numpy as np
from tqdm import tqdm
import math
from scipy.sparse import csc_matrix

# vector_type should be one of ['count','freq','binary','tfidf']
# Update 1: also return id2token: reverse map from col ids back to tokens
def compute_doc_vectors(documents, vector_type='count'):

    document_dicts = []

    # Now we need to make a vector based on these tokens
    # We don't know the full vocab yet (until we process everything)
    # So instead of doing a full pass over everything to compute that
    # Let's make a column_id for each new word we see and use a dict
    # then we can rely on that later to build the vectors without
    # processing the documents again (just need to check the dict objects)
    token2id = {}
    current_next_id = 0

    for document in tqdm(documents):
      tokens = tokenize(document)

      # map from token_id -> tf (count, binary, frequency, etc.)
      document_dict = collections.defaultdict(int)

      doc_length = len(tokens)
      for token in tokens:
          if token not in token2id:
              token2id[token] = current_next_id
              current_next_id += 1
          token_id = token2id[token]

          document_dict[token_id] += 1

      document_dicts.append(document_dict)

    data = []
    rows = []
    cols = []
    for doc_id, document_dict in enumerate(document_dicts):
        for word_id, count in document_dict.items():
            data.append(count)
            rows.append(doc_id)
            cols.append(word_id)
    sparse_mat = csc_matrix((data, (rows, cols)))

    vectors = []
    for document_dict in document_dicts:
        vector = [document_dict[token_id] for token_id in range(current_next_id)]
        vectors.append(vector)


    mat = np.array(vectors, dtype='float64')
    if vector_type == 'count':
        # already in this format
        pass
    elif vector_type == 'binary':
        mat = np.where(mat > 0, 1, 0)
    elif vector_type == "freq" or vector_type == "tfidf":
        mat /= mat.sum(axis=1).reshape(-1,1)
        if vector_type == "tfidf":
            doc_freq = np.where(mat > 0, 1, 0).sum(axis=0)
            idf = np.log(1/doc_freq)
            mat *= idf

    id2token = {id:tok for tok,id in token2id.items()}

    return mat,sparse_mat,id2token

# subset_docs = paths[::100]
# vectors, id2token = compute_doc_vectors(subset_paths, vector_type='count')

import glob
paths = glob.glob("./books/*")
# paths
docs=[]
docs_n_labels=[]
categories=set([path.split('__')[1].split('.')[0] for path in paths])
print(categories)
for file_path in paths:
  with open(file_path,'r',encoding='utf-8') as file_handle:
    book_contents = file_handle.read()
    book_text=book_contents.lower().split('contents')[1] # exclue preface etc.
    documents=book_text.split('\n\nchapter')
    category=file_path.split('__')[1].split('.')[0]
    docs_n_labels.extend([(category,document) for document in documents])
    docs.extend(documents)
    

print(len(docs))

avg_word_count_per_doc = sum(map(len, docs))/float(len(docs))
avg_word_count_per_doc

vectors,sparse_vectors, id2token = compute_doc_vectors(docs, vector_type='tfidf')
print(vectors.shape)
print(id2token[1])


def naiveB_Probs(docs_n_labels):
    label_counts = collections.defaultdict(int)
    total_words_with_label = collections.defaultdict(int)
    word_with_label_counts = {}
    word_with_label_0_counts = {}
    vocab = set()
    total_docs = len(docs_n_labels)
    for label,document in tqdm(docs_n_labels):
        label_counts[label] += 1
        if label not in word_with_label_counts:
            word_with_label_counts[label] = collections.defaultdict(int)
        words = tokenize(document)
        for word in words:
            vocab.add(word)
            word_with_label_counts[label][word] += 1
            total_words_with_label[label] += 1
    priors = {label:count/total_docs for label,count in label_counts.items()}

    
    log_likelihoods = {}

    for label, word_counts in word_with_label_counts.items():
        log_likelihoods[label] = {}
        for word in vocab:
            count_c = word_counts[word]


            log_likelihood_c = math.log( (count_c + 1) / 
                                                    (total_words_with_label[label] + len(vocab)) )


            # compute count of w for c_0
            # compute sum of count of all vocab words for c_0
            count_c0=0
            total_words_with_c0=0
            for label_c0, word_counts in word_with_label_counts.items():
              if label_c0 != label:
                count_c0 += word_counts[word]
                total_words_with_c0 += total_words_with_label[label_c0]

            log_likelihood_c0 = math.log( (count_c0 + 1) / 
                                                    (total_words_with_c0 + len(vocab)) )

            
            log_likelihoods[label][word] = log_likelihood_c - log_likelihood_c0


    return log_likelihoods, vocab

log_likelihoods, vocab = naiveB_Probs(docs_n_labels)

# sort by log-likelihoods

for category in categories:
  words=log_likelihoods[category]
  # print(type(words))
  sorted_keys = sorted(words, key=words.get, reverse=True)
  for r in sorted_keys[:10]:
    print(category,":",r,":", words[r])
  # sorted(words.items(), key=lamda , reverse=False)


word_counts = sparse_vectors.sum(axis=0)
print(word_counts.shape)
from matplotlib import pyplot as plt
bins = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]
plt.hist(word_counts.tolist()[0], bins)
plt.xlabel("token count")
plt.ylabel("number of tokens with that count")
plt.show()

from gensim.test.utils import common_corpus
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus

num_topics = 15
corpus = Sparse2Corpus(sparse_vectors, documents_columns=False)
# Train the model on the corpus.
# Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
lda = LdaModel(corpus=corpus, id2word=id2token, num_topics=num_topics)

print(lda.get_topics().shape)
print(lda.print_topics())