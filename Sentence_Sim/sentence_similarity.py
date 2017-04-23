import nltk
import re
from nltk.corpus import wordnet as wn
from itertools import product
from nltk.corpus import wordnet_ic


def get_sim_score(word_1, word_2, info_content):
    """ 
    Calculate the highest path similarity among all pairs. 
    """

    if word_1 == word_2:
        return 1
    else:
        max_sim = -1.0
        synsets_1 = wn.synsets(word_1)
        synsets_2 = wn.synsets(word_2)
        if synsets_1 and synsets_2:
            for synset_1, synset_2 in product(synsets_1, synsets_2):
                try:
                    sim = wn.res_similarity(synset_1, synset_2, info_content)
                    #sim = wn.wup_similarity(synset_1, synset_2)
                    if sim > max_sim:
                        max_sim = sim
                except:
                    continue

            return max_sim
        return max_sim


def mySim(text1, text2, sigma=0.85, w=0.3, corpus='ic-brown-resnik.dat'):
    # set stop words
    stopwords = nltk.corpus.stopwords.words('english')
    # set variables
    x = []
    y = []
    dic = {}
    info_content = wordnet_ic.ic(corpus)

    # clean raw text
    text1 = re.sub('[^a-zA-Z]', ' ', text1).lower()
    text2 = re.sub('[^a-zA-Z]', ' ', text2).lower()

    # tokenize inputs into vectors
    token_p = nltk.word_tokenize(text1, language='english')
    token_r = nltk.word_tokenize(text2, language='english')

    concept_p = [words for words in token_p if words not in stopwords]
    concept_r = [words for words in token_r if words not in stopwords]

    m = len(concept_p)
    n = len(concept_r)

    print(concept_p, concept_r)

    for w1, w2 in product(concept_p, concept_r):
        sim = get_sim_score(w1, w2, info_content)
        if sim >= sigma:
            x.append(w1)
            y.append(w2)
            dic[w1] = w2

    concept_x = [words for words in token_p if words in x]
    concept_y = [words for words in token_r if words in y]

    print(concept_x, concept_y)

    count = len(concept_x)
    total = 0

    if count == 0:
        sim_score_0 = 0

    elif count % 2 == 0 or count == 1:
        for position, word in enumerate(concept_x):
            total += abs(position - concept_y.index(dic[word]))

        sim_score_0 = 1 - 2 * total / count ** 2

    else:
        for position, word in enumerate(concept_x):
            total += abs(position - concept_y.index(dic[word]))

        sim_score_0 = 1 - 2 * total / (count ** 2 - 1)

    S = ((m + n) / (2 * m * n)) * (count * (1 - w * (1 - sim_score_0)))

    return S

if __name__ == '__main__':
    target = 'By 2030, eradicate extreme poverty for all people everywhere, currently measured as people living on less than $1.25 a day'
    test = 'By 2017, the proportion of severely poor individuals has dropped from 15.8% in 2009/10 to below 10%.'