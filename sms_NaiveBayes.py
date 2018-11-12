############################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
##############################################
### Data sets: sms_spam.csv
### Notes: This code is provided without warranty.

import numpy as np
import pandas as pd

sms_raw = pd.read_csv("d:\Documents\AdvancedStatisticsMachineLearning\sms_spam.csv")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

sms_raw.dtypes
sms_raw['type'].value_counts() / len(sms_raw['type'])

### label encoding
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder() # An empty encoding model
# sms_raw['type'] = le.fit_transform(sms_raw['type']) # "=" 切開，先看又在看左
# print(sms_raw[:5])

### sentence and word tokenization
import nltk  # Natural Language ToolKit

# def tokenize_text(text):
#    sentences = nltk.sent_tokenize(text)
#    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
#    return word_tokens

# token_list = [tokenize_text(text) for text in sms_raw['text']]

# token_list[:5]

### word tokenization only
token_list0 = [nltk.word_tokenize(txt) for txt in sms_raw['text']]

token_list0[:5]

# help(nltk.word_tokenize)

### transforming to lowercase words
# from nltk.stem import WordNetLemmatizer
# lemma = WordNetLemmatizer()
# token_list1 = [[lemma.lemmatize(word.lower()) for word in doc] for doc in token_list0]
token_list1 = [[word.lower() for word in doc] for doc in token_list0]
token_list1[:5]

### removing stopwords
from nltk.corpus import stopwords

type(stopwords.words)  # method
type(stopwords.words('english'))
len(stopwords.words('english'))  # 153 English stopwords

# token_list2 = []
# for i, doc in enumerate(token_list1):
#    for word in doc:
#        if word not in stopwords.words('english'):
#            tokenlist2[i].append(word)
#    return token_list2

token_list2 = [[word for word in doc if word not in stopwords.words('english')] for doc in token_list1]
token_list2[:5]

### removing punctuations
import string

token_list3 = [[word for word in doc if word not in string.punctuation] for doc in token_list2]
token_list3[:5]

### removing all digits tokens
token_list4 = [[word for word in doc if not word.isdigit()] for doc in token_list3]
token_list4[:5]

### removing tokens with digits and punctuations (letters + digits or punctuations)
token_list5 = [[''.join([i for i in word if not i.isdigit() and i not in string.punctuation]) for word in doc] for doc
               in token_list4]
token_list5[:5]

# token_list5 = [[''.join([i for i in word if not i.isdigit()]) for word in doc if not word.isdigit()] for doc in token_list4]

# removing tokens with punctuations
# token_list6 = [[''.join([i for i in word if i not in string.punctuation]) for word in doc] for doc in token_list5]
# token_list6[:5]

### removing empty tokens & lemmatization
token_list6 = [list(filter(None, doc)) for doc in token_list5]
token_list6[:15]

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
token_list6 = [[lemma.lemmatize(word) for word in doc] for doc in token_list6]
token_list6[:15]

# for c in string.punctuation:
#    s= s.replace(c,"")

### https://stackoverflow.com/questions/15899861/efficient-term-document-matrix-with-nltk/28727111
### https://textminingonline.com/dive-into-nltk-part-i-getting-started-with-nltk

# from sklearn.feature_extraction.text import CountVectorizer
# vec = CountVectorizer(min_df = 1, stop_words = 'english')
# dtm = vec.fit_transform(token_list3) # AttributeError: 'list' object has no attribute 'lower'

# import textmining # ModuleNotFoundError: No module named 'stemmer'
# import stemmer # ModuleNotFoundError: No module named 'stemmer'
# tdm = textmining.TermDocumentMatrix()

### concatenate all tokens for each document
token_list7 = [' '.join(doc) for doc in token_list6]  # Attention! 'doc' is a list.
token_list7[:5]

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(token_list7)  # input a Series or a list
# X = vec.fit_transform(sms_raw['text']) # input a Series or a list
sms_dtm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(sms_dtm.shape)
# type(vec.get_feature_names())
len(vec.get_feature_names())  # 8397 words -> 7612 words
print(vec.get_feature_names()[300:306])
print(np.argwhere(sms_dtm['app'] > 0))

print(sms_dtm.iloc[4460:4470, 300:306])

sms_raw_train = sms_raw.iloc[:4170, :]
sms_raw_test = sms_raw.iloc[4170:, :]
sms_dtm_train = sms_dtm.iloc[:4170, :]
sms_dtm_test = sms_dtm.iloc[4170:, :]
token_list6_train = token_list6[:4170]
token_list6_test = token_list6[4170:]

sms_raw_train['type'].value_counts() / len(sms_raw_train['type'])
sms_raw_test['type'].value_counts() / len(sms_raw_test['type'])

### wordcloud (spam vs. ham)
# from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud

# type(STOPWORDS) # already a set
# len(STOPWORDS) # 190 stopwords

### concatenate all tokens across all documents
tokens_train = [token for doc in token_list6_train for token in doc]
print(len(tokens_train))  # 38104

tokens_train_spam = [token for is_spam, doc in zip(sms_raw_train['type'] == 'spam', token_list6_train) if is_spam for
                     token in doc]

tokens_train_ham = [token for is_ham, doc in zip(sms_raw_train['type'] == 'ham', token_list6_train) if is_ham for token
                    in doc]

# tokens_test = [token for doc in token_list6_test for token in doc]
# tokens = [token for doc in token_list6 for token in doc]
# len(tokens) # 61313 -> 57572
str_train = ','.join(tokens_train)
# str_test = ','.join(tokens_test)
str_train_spam = ','.join(tokens_train_spam)
str_train_ham = ','.join(tokens_train_ham)

# mask=np.array(Image.open(os.path.join(currdir, "cloud.png"))
# font_path = '/Users/Vince/cstsouMac/Python/Examples/TextMining/msyh.ttf'
# from wordcloud import WordCloud
wc_train = WordCloud(background_color="white",
                     prefer_horizontal=0.5)  # max_words default is 200 and stopwords default is STOPWORDS.
wc_train.generate(str_train)
import matplotlib.pyplot as plt

plt.imshow(wc_train)
plt.axis("off")
plt.title('Wordcloud of training SMS')
# plt.show()
plt.savefig('wc_train.png')

wc_train_spam = WordCloud(background_color="white", prefer_horizontal=0.5)
wc_train_spam.generate(str_train_spam)
# import matplotlib.pyplot as plt

plt.imshow(wc_train_spam)
plt.axis("off")
plt.title('Wordcloud of training spam SMS')
# plt.show()
plt.savefig('wc_train_spam.png')

wc_train_ham = WordCloud(background_color="white", prefer_horizontal=0.5)
wc_train_ham.generate(str_train_ham)
# import matplotlib.pyplot as plt

plt.imshow(wc_train_ham)
plt.axis("off")
plt.title('Wordcloud of training ham SMS')
# plt.show()
plt.savefig('wc_train_ham.png')

### Multinomial Bayes
from sklearn.naive_bayes import MultinomialNB

# help(MultinomialNB)

### Example from help
# import numpy as np
# X = np.random.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf.fit(X, y)
# print(clf.predict(X[2:3]))
#####################

### Hold-out
clf = MultinomialNB()
clf.fit(sms_dtm_train, sms_raw_train['type'])

train = clf.predict(sms_dtm_train)
sum(sms_raw_train['type'] == train) / len(train)

pred = clf.predict(sms_dtm_test)
sum(sms_raw_test['type'] == pred) / len(pred)  # 0.9910055765425436 (training), 0.9719222462203023 (test)

clf.class_log_prior_  # array([-0.14531691, -2.00061706])
clf.feature_log_prob_
clf.feature_log_prob_.shape  # (2, 7612)
clf.class_count_  # array([3606.,  564.])
clf.feature_count_
np.max(clf.feature_count_)  # 282.0

### Cross-validation
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem


def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator of K folds
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    # len(y, i.e. news.target): total number of elements (18846); K: number of folds (>=2)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))


# import numpy as np
evaluate_cross_validation(clf, sms_dtm, sms_raw['type'], 5)  # 5-fold CV, it needs some time!
# print("The results of 5-fold CV: {0})
