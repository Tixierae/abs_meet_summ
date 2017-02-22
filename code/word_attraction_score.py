##################
#### packages ####
##################

from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances as ed

###################
#### functions ####
###################

# returns the vector of a word
def my_vector_getter(word, wv):
    try:
        # we use reshape because cosine similarity in sklearn now works only for multidimensional arrays
        word_array = wv[word].reshape(1,-1)
        return (word_array)
    except KeyError:
        print 'word: <', word, '> not in vocabulary!'

# returns euclidean distance between two word vectors
def my_euclidean_distance(word1, word2, wv):
    distance = ed(my_vector_getter(word1, wv),my_vector_getter(word2, wv)) 
    return (round(distance, 4))
	
def word_attraction_force(w1, w2, lotf, wv):
    '''
    compute the attraction force between two words using word embeddings
    based on this paper: Wang, R., Liu, W., & McDonald, C. (2014, November). Corpus-independent generic keyphrase extraction using word embedding vectors. In Software Engineering Research Conference (p. 39).
    ! see slide #7 here for a summary of the important concepts/formulaes: http://www.lix.polytechnique.fr/~anti5662/dascim_group_presentation_paper_review_tixier_10_14_16.pdf
    '''
    f1 = lotf.count(w1)
    f2 = lotf.count(w2)
    d = my_euclidean_distance(w1, w2, wv)
    waf = round(f1 * f2 / float(d * d), 5)
    return waf

##################
#### examples ####
##################

sentences = ['The wife of a former U.S. president Bill Clinton Hillary Clinton visited China last Monday','Hillary Clinton wanted to visit China last month but postponed her plans till Monday last week.','Hillary Clinton paid a visit to the People Republic of China on Monday.', 'Last week the Secretary of State Ms. Clinton visited Chinese officials.']

# we do not convert to lower case since Google News word vectors were learned on raw text
lists_of_tokens = [sent.split(' ') for sent in sentences]

lists_of_tokens_flatten = [item for sublist in lists_of_tokens for item in sublist]

# create empty word vectors for the words in vocabulary	
# we set size=300 to match dim of GNews word vectors
vectors = Word2Vec(size=3e2, min_count=1)
vectors.build_vocab(lists_of_tokens)

#vocab = [elt[0] for elt in vectors.vocab.items()]

# fill our empty word vectors with Google News ones
# they can be downloaded from here: https://code.google.com/archive/p/word2vec/ (! very big file)
path_to_wv = 'E:\\' # to fill

# we load only the Google word vectors corresponding to our vocabulary
# it takes quite some time, so it is better to compute the entire vocabulary for all sentences in all communities and load the vectors once at the beginning

vectors.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

# for vector normalization (in case)
#vectors.init_sims(replace=True)

#my_euclidean_distance('Clinton','Hillary', wv=vectors)

#my_euclidean_distance('Clinton','president', wv=vectors)

#my_euclidean_distance('China', 'visited', wv=vectors)

#my_euclidean_distance('Monday', 'People', wv=vectors)

# for instance:
word_attraction_force('China', 'visited', lotf=lists_of_tokens_flatten, wv=vectors)

word_attraction_force('Clinton', 'Hillary', lotf=lists_of_tokens_flatten, wv=vectors)