
##Author: Hamza Khanane

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import *
from nltk.stem import *
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import timeit
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial import distance




##CLEANING DATA STARTS HERE
start = timeit.default_timer()

#Your statements here
start = timeit.default_timer()
list = []
f=open("train_file.data", "r")

f1 = f.readlines()

##appending word by word in a list

for x in f1:
    list.append(x)

f.close()

stopWords = set(stopwords.words('english'))

i = 0


filtered_sentence = []

plusandminus = []

for data in list:

    data = list[i]
    plusandminus.append(data[:2])
    data = data.lower()
    table = string.maketrans("", "")
    data = data.translate(table, string.punctuation)
    unfiltered = word_tokenize(data)
    i = i + 1
    data = ' '.join([word for word in data.split() if word not in stopWords])
    filtered_sentence.append(data)


ps = PorterStemmer()
list_stem = [ps.stem(word) for word in filtered_sentence]

split = list_stem

test_reviews = []
final_filtered_sentence = []
f_testfile = open("test.data", "r")  ## open my test file

f2 = f_testfile.readlines() ##use the read line function

for x in f2: ##run a loop for line by line
    test_reviews.append(x)


f_testfile.close()

j = 0

for data in test_reviews:

    data = test_reviews[j]
    data = data.lower()
    table = string.maketrans("", "")
    data = data.translate(table, string.punctuation)
    unfiltered = word_tokenize(data)
    j = j + 1
    data = ' '.join([word for word in data.split() if word not in stopWords])
    final_filtered_sentence.append(data)


ps = PorterStemmer()
list_stem2 = [ps.stem(word) for word in final_filtered_sentence]




stop2 = timeit.default_timer()
print('Time: ', stop2 - start)
## CLEANING DATA ENDS HERE







f10 = open("results.txt", "w") ##OPEN THE WRITING FILE


########TFIDF BEGINS


vectorizer = TfidfVectorizer(max_features=10000)
list_all = list_stem2 + list_stem

all_vec = vectorizer.fit_transform(list_all)

train_vectorizer = vectorizer.fit_transform(np.array(list_stem))
test_vectorizer = vectorizer.transform(np.array(list_stem2))


index = 0
for test_r in test_vectorizer:
    print("review number", index)
    index = index + 1
    similarities = []
    training = 0
    dist = cosine_similarity(test_r, train_vectorizer)
    similarities = dist[0].tolist()

    listC = sorted(similarities, reverse=True)

    res = []
    final = 0
    k = 0
    while k < 19:
        j = similarities.index(listC[k])
        res.append(plusandminus[j])
        k = k+1
    print(res)
    final = max(set(res), key=res.count)
    print(final)
    if final == '+1':
        f10.write('+1\n')
    else:
        f10.write('-1\n')
        print(res)









##print("list of reviews from test file", list_stem2)

##tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=2, max_df=0.9, stop_words = 'english')



##tfidf_matrix = tf.fit_transform(test_reviews)







##print(test_reviews)

stop = timeit.default_timer()
print('Time: ', stop - start)














