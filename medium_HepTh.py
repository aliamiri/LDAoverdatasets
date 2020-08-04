import os

docs = list()
for path, subdirs, files in os.walk("cit-HepTh-abstracts"):
    for name in files:
        print(os.path.join(path, name))
        f = open(os.path.join(path, name), "r")
        lines = f.readlines()[11:]
        txt = ' '
        txt = txt.join(lines)
        # txt = f.read()
        docs.append(txt)

# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data

from sklearn.feature_extraction.text import CountVectorizer

no_features = 1000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(docs)
tf_feature_names = tf_vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation

no_topics = 30

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)


perplexity = lda.perplexity(tf)

score = lda.score(tf)
print(perplexity)
print(score)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


no_top_words = 100
# display_topics(lda, tf_feature_names, no_top_words)
