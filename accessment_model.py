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
        max_sim = 0.0
        synsets_1 = wn.synsets(word_1)
        synsets_2 = wn.synsets(word_2)
        if synsets_1 and synsets_2:
            for synset_1, synset_2 in product(synsets_1, synsets_2):
                try:
                    sim = wn.lin_similarity(synset_1, synset_2, info_content)
                    # sim = wn.jcn_similarity(synset_1, synset_2, info_content)
                    # sim = wn.wup_similarity(synset_1, synset_2)
                    if sim > max_sim:
                        max_sim = sim
                except:
                    continue

            return max_sim
        return max_sim


def remove_dup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def mySim(text1, text2, sigma=0.85, w=0.3, corpus='ic-brown-resnik.dat'):
    import re
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

    # pos the tokens and n-grams
    pos_p = [word for word, tag in nltk.pos_tag(concept_p) if tag.startswith('NN') or tag.startswith('JJ')]
    pos_p.extend([' '.join(words).strip() for words in nltk.ngrams(pos_p, 2)])

    pos_r = [word for word, tag in nltk.pos_tag(concept_r) if tag.startswith('NN') or tag.startswith('JJ')]
    pos_r.extend([' '.join(words).strip() for words in nltk.ngrams(pos_r, 2)])

    # print(pos_p, pos_r)

    m = len(pos_p)
    n = len(pos_r)

    total_sim = 0
    total_count = 0

    for w1, w2 in product(pos_p, pos_r):
        sim = 0
        w1_set = set(w1.split())
        w2_set = set(w2.split())

        count = 0
        for t1, t2 in product(w1_set, w2_set):
            sim += get_sim_score(t1, t2, info_content)
            count += 1

        if sim / count >= sigma:
            total_sim += sim
            total_count += 1

            x.append(w1)
            y.append(w2)
            dic[w1] = w2
            # print(sim, '\t\t', w1_set, '|',  w2_set)

    l = len(x)
    #concept_x = [words for words in pos_p if words in x]
    #concept_y = [words for words in pos_r if words in y]


    if total_count:
        score = ((m + n) / (2 * m * n)) * (l * (total_sim / total_count))
        return score
    else:
        return 0


def concept_matching(concepts, sentence, sigma=0.85, w=0.3, corpus='ic-brown-resnik.dat'):
    # set stop words
    stopwords = nltk.corpus.stopwords.words('english')
    # set variables
    x = []
    y = []
    dic = {}
    info_content = wordnet_ic.ic(corpus)

    # clean raw text
    concepts = [re.sub('[^a-zA-Z]', ' ', word).lower() for word in concepts]
    context = re.sub('[^a-zA-Z]', ' ', sentence).lower()

    # tokenize inputs into vectors
    tokens = nltk.word_tokenize(context, language='english')

    # remove stopwords
    concept_p = [words for words in tokens if words not in stopwords]

    # pos the tokens and n-grams
    pos_p = [word for word, tag in nltk.pos_tag(concept_p) if tag.startswith('NN') or tag.startswith('JJ')]
    pos_p.extend([' '.join(words).strip() for words in nltk.ngrams(pos_p, 2)])

    # print(pos_p)

    count = 0
    total_sim = 0

    for w1, w2 in product(concepts, pos_p):
        sim = 0
        w1_set = set(w1.split())
        w2_set = set(w2.split())
        for t1, t2 in product(w1_set, w2_set):
            sim += get_sim_score(t1, t2, info_content)
            # print(sim, '\t\t', t1, t2, '\t\t', w1_set, '|',  w2_set)

        if sim >= sigma:
            count += 1
            total_sim += sim

            # x.append(w1)
            # y.append(w2)
            # dic[w1] = w2
            # print(sim, '\t\t', w1_set, '|',  w2_set)

    # concept_x = [words for words in concepts if words in x]
    # concept_y = [words for words in pos_p if words in y]

    # print(concept_x, concept_y)
    if count:
        return total_sim / count
    else:
        return 0

if __name__ == '__main__':
    print(mySim('Many consider Malin as the best player in PingPong history', 'Malin is one of the best PingPong players', 0.85))