
import numpy as np
import pandas as pd
import pyLDAvis as pyLDAvis

from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from gensim.utils import simple_preprocess
import gensim, spacy
from gensim.models.ldamulticore import LdaMulticore
import re


from gensim.models import Phrases
from gensim.models.phrases import Phraser

data = pd.read_csv('neg_comment.csv')
print(data.head())
data['Comments']=data['Comments'].astype(str)
data = data.loc[(data['Overall']=="Unsatisfied") & (data['Type']=='Annual')]
#data = data.loc[(data['Overall']=="Unsatisfied")]

rating=data['Overall']
# #data['length'] = data.Comments.apply(lambda row: len(row.split()))
# #print('Mean length: ', data['length'].mean())
#
import seaborn as sns
import matplotlib.pyplot as plt
#
# sns.set_style(style="darkgrid")
# sns.distplot(data['length'])
# plt.show()


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# create N-grams
def make_n_grams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    bigrams_text = [bigram_mod[doc] for doc in texts]
    trigrams_text =  [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
    return trigrams_text
tokens_reviews = list(sent_to_words(data['Comments']))
tokens_reviews = make_n_grams(tokens_reviews)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# I use gensim stop-words and add me own stop-words, based on texts
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in gensim.parsing.preprocessing.STOPWORDS.union(set(['1ndash', 'ndash','unknown', 'donx27t',
                                                                                                                           'service', 'issue', 'none', 'comment',
                                                                                                                           'nx2fa','solve','resolve','resolved','problem','donx','thank','lot','ion','department','need','support','provide','support','user','use','needs']))] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# do lemmatization keeping only noun, vb, adv
# because adj is not informative for reviews topic modeling
reviews_lemmatized = lemmatization(tokens_reviews, allowed_postags=['NOUN', 'VERB', 'ADV'])

# remove stop words after lemmatization
reviews_lemmatized = remove_stopwords(tokens_reviews)
np.random.seed(0)


from mgp import MovieGroupProcess

mgp = MovieGroupProcess(K=6, alpha=0.01, beta=0.01, n_iters=30)

vocab = set(x for review in reviews_lemmatized for x in review)
n_terms = len(vocab)
model = mgp.fit(reviews_lemmatized, n_terms)

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))
doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('\nMost important clusters (by number of docs inside):', top_index)
# show the top 5 words in term frequency for each cluster
top_words(mgp.cluster_word_distribution, top_index, 10)

# I don`t rename the clusters

topic_dict = {}
topic_names = ['type 1',
               'type 2',
               'type 3',
               'type 4',
               'type 5',
               'type 6',

               ]
for i, topic_num in enumerate(top_index):
    topic_dict[topic_num]=topic_names[i]


def create_topics_dataframe(data_text=data.Comments,  mgp=mgp, threshold=0.3, topic_dict=topic_dict, lemma_text=reviews_lemmatized):
    result = pd.DataFrame(columns=['Comments', 'Topic',  'Lemma-text'])
    for i, text in enumerate(data_text):
        result.at[i, 'Comments'] = text
        #result.at[i, 'Overall'] = data.Overall[i]
        result.at[i, 'Lemma-text'] = lemma_text[i]
        prob = mgp.choose_best_label(reviews_lemmatized[i])
        if prob[1] >= threshold:
            result.at[i, 'Topic'] = topic_dict[prob[0]]
        else:
            result.at[i, 'Topic'] = 'Other'
    return result

result = create_topics_dataframe(data_text=data.Comments, mgp=mgp, threshold=0.3, topic_dict=topic_dict, lemma_text=reviews_lemmatized)
result['Rating']=list(rating)

import plotly.express as px

fig = px.pie(result, names='Topic',  title='Topics', color_discrete_sequence=px.colors.sequential.Burg)
fig.show()

import matplotlib.pyplot as plt



rating_counts = result.Rating.value_counts()
types_counts = result.Topic.value_counts()
fig, ax = plt.subplots(1, 2, figsize=(15,5))
rating = sns.barplot(x = rating_counts.index, y = rating_counts.values, palette="pastel", ax=ax[0])
types = sns.barplot(x = types_counts.index, y = types_counts.values, palette="pastel", ax=ax[1])

fig = px.sunburst(result, path=['Topic', 'Rating'], title='Topics and ratings', color_discrete_sequence=px.colors.sequential.Burg)
fig.show()



#Create LDA-model:

id2word = corpora.Dictionary(reviews_lemmatized)
texts = reviews_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

# Use TF-IDF
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

from gensim.models.ldamulticore import LdaMulticore

# Creating the object for LDA model using gensim library

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(corpus, num_topics=10, id2word = id2word, passes=1, random_state=0, eval_every=None)

# Prints the topics with the indexes: 0,1,2 :


#print(ldamodel.print_topics(num_topics=6, num_words=5))

# num_topics mean: how many topics want to extract
# num_words: the number of words that want per topic

# we need to manually check whethere the topics are different from one another or not
from IPython.core.getipython import get_ipython
#pyLDAvis.enable_notebook()
import pyLDAvis.gensim_models
panel = pyLDAvis.gensim_models.prepare(ldamodel, corpus, id2word, mds='tsne')
pyLDAvis.save_html(panel, 'LDA_Visualization.html')