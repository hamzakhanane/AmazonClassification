# AmazonClassification
A model which classifies 18,000 reviews into positive or negative.

## Technologies used
* K Nearest Neighbor Algorithm
* NLTK
*TFIDF

##Sample Code snippet of the customized KNN algorithm

````
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
        
````



