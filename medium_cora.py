docs = list()
import pickle

import os

import numpy as np
import pandas as pd

data_dir = os.path.expanduser("cora")
feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names = feature_names + ["subject"]
node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
docs = list()

for data in node_data.values:
    doc = ' '
    i = 0
    for p in data:
        if p == 1:
            doc += " w_{}".format(i)
        i += 1
    docs.append(doc)

from sklearn.feature_extraction.text import CountVectorizer

no_features = 1433

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(docs)
tf_feature_names = tf_vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation

no_topics = 10

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=50, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

perplexity = lda.perplexity(tf)

print(perplexity)
print(no_topics)

#
# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic %d:" % (topic_idx))
#         print(" ".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]]))


doc_scores = list()
for doc in docs:
    score = np.zeros(no_topics)
    sum = np.zeros(no_topics)
    for w in range(0, len(tf_feature_names)):
        if doc.__contains__(tf_feature_names[w]):
            for j in range(0, no_topics):
                sum[j] += lda.components_[j, w]

    np_sum = np.sum(sum)
    for j in range(0, no_topics):
        score[j] = sum[j] / np_sum
    doc_scores.append(score)

with open('cora_10.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(doc_scores, filehandle)
# display_topics(lda, feature_names, no_top_words)

no_topics = 15

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=50, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

perplexity = lda.perplexity(tf)

print(perplexity)
print(no_topics)

doc_scores = list()
for doc in docs:
    score = np.zeros(no_topics)
    sum = np.zeros(no_topics)
    for w in range(0, len(tf_feature_names)):
        if doc.__contains__(tf_feature_names[w]):
            for j in range(0, no_topics):
                sum[j] += lda.components_[j, w]

    np_sum = np.sum(sum)
    for j in range(0, no_topics):
        score[j] = sum[j] / np_sum
    doc_scores.append(score)

with open('cora_15.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(doc_scores, filehandle)
# display_topics(lda, feature_names, no_top_words)


no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=50, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

perplexity = lda.perplexity(tf)

print(perplexity)
print(no_topics)

doc_scores = list()
for doc in docs:
    score = np.zeros(no_topics)
    sum = np.zeros(no_topics)
    for w in range(0, len(tf_feature_names)):
        if doc.__contains__(tf_feature_names[w]):
            for j in range(0, no_topics):
                sum[j] += lda.components_[j, w]

    np_sum = np.sum(sum)
    for j in range(0, no_topics):
        score[j] = sum[j] / np_sum
    doc_scores.append(score)

with open('cora_20.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(doc_scores, filehandle)
# display_topics(lda, feature_names, no_top_words)
