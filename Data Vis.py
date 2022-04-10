import nltk
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob, Word
from helpers import get_top_n_words1,get_keys,keys_to_counts,get_top_n_words2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

from IPython.display import display
from tqdm import tqdm
from collections import Counter
import ast
import matplotlib.mlab as mlab
import seaborn as sb
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from wordcloud import ImageColorGenerator
from PIL import Image


# to download NLTK
import ssl



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
#nltk.download()
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter



"""data loading and extraction"""

df = pd.read_csv("AllCSF.csv")
print(len(df.columns))
print(df.shape)

print(df['Overall'].isna().sum())
df['Overall']=df.Overall.fillna("Neutral")
df['Comments']=df.Comments.replace("D/A","Unknown")
df['Comments'] = df['Comments'].str.replace('[^A-Za-z0-9 ]', "")
df['Comments']=df['Comments'].str.lower()
df['Comments']=df.Comments.fillna("unknown")

df['Overall'].mask(df['Overall'] == "1", 'Satisfied', inplace=True)
df['Overall'].mask(df['Overall'] == "0", 'Unsatisfied', inplace=True)



stop = stopwords.words('english')

stop.append("1ndash")
stop.append("ndash")
stop.append("unknown")
stop.append("donx27t")
stop.append("service")
#stop.append("Thank")
stop.append("issue")
stop.append("need")
stop.append("none")
stop.append("comment")
#stop.append("thank")
stop.append("nx2fa")
stop.append("u")
stop.append("still")
stop.append("problem")
stop.append("ticket")
stop.append("closed")
stop.append("many")
stop.append("solved")
stop.append("resolved")
stop.append("every")
stop.append('saudi')
stop.append('aramco')
#df['Comments']= df['Comments'].apply(lambda t: " ".join([Word(word).lemmatize() for word in t.split()]))
#df['Comments'] = df['Comments'].apply(lambda t: " ".join(word for word in t.split() if word not in stop))

#neg_comment = df.loc[(df['Overall']=="Unsatisfied")& (df['Type']=="Annual")]
#neg_comment = df.loc[(df['Overall']=="Unsatisfied")& (df['Type']=="TT")]
neg_comment = df.loc[(df['Overall']=="Unsatisfied") & (df['Type']=='Annual')]
df.to_csv('neg_comment.csv', index=False)

#allcomment=df.loc[(df['Type']=='Annual')]
reindexed_data=neg_comment['Comments']
print(len(reindexed_data))

vectorizerneg = CountVectorizer(ngram_range=(2, 2), stop_words=stop )
bag_of_words = vectorizerneg.fit_transform(reindexed_data)
vectorizerneg.vocabulary_
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizerneg.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print (words_freq[:100])
#Generating wordcloud and saving as jpg image
words_dict = dict(words_freq)

#Create the mask
colosseum_mask = np.array(Image.open('sad-emoticon-black-face-symbol_icon-icons.com_57440.png'))

#Grab the mask colors
colors = ImageColorGenerator(colosseum_mask)



plt.figure(figsize=(10,8))
wordCloud = WordCloud(background_color="black",colormap='Dark2',mask=colosseum_mask)
wordCloud.generate_from_frequencies(words_dict)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")

plt.show()
wordCloud.to_file('wordcloud_bigram.png')





words, word_values = get_top_n_words1(n_top_words=30,
                                     count_vectorizer=vectorizerneg,
                                     text_data=reindexed_data)

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), word_values);
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
plt.xticks(fontsize=8, rotation=45)
ax.set_title('Top  words (excluding stop words)');
ax.set_xlabel('Word');
ax.set_ylabel('Number of occurences');
plt.show()
plt.savefig('barplotofwordcount.png')

