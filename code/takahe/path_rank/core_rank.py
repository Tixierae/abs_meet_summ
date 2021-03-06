import nltk
import string
import re
import itertools
import heapq
import operator
import igraph
import copy
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag


def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):

    punct = string.punctuation.replace('-', '')

    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    # strip extra white space
    text = re.sub(' +', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')
    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
                item[1] == 'NN' or
                item[1] == 'NNS' or
                item[1] == 'NNP' or
                item[1] == 'NNPS' or
                item[1] == 'JJ' or
                item[1] == 'JJS' or
                item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return(tokens)


def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of
    # fixed size 'w'

    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in xrange(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i - w + 1):(i + 1)]

        # edges to try
        candidate_edges = []
        for p in xrange(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit
                # weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertice weights
    # based on co-occurence within sliding window
    g.es['weight'] = from_to.values()
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return(g)


def core_dec(g, weighted=True):

    # work on clone of g to preserve g
    gg = copy.deepcopy(g)

    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs["name"], [0] * len(gg.vs["name"])))

    if weighted == True:
        # k-core decomposition for weighted graphs (generalized k-cores)
        # based on Batagelj and Zaversnik's (2002) algorithm #4

        # initialize min heap of degrees
        heap_g = zip(gg.vs["weight"], gg.vs["name"])
        heapq.heapify(heap_g)

        for i in range(11):

            top = heap_g[0][1]
            # find vertice index of heap top element
            index_top = gg.vs["name"].index(top)
            # save names of its neighbors
            neighbors_top = gg.vs[gg.neighbors(top)]["name"]
            # exclude self-edges
            neighbors_top = [elt for elt in neighbors_top if elt != top]
            # set core number of heap top element as its weighted degree
            cores_g[top] = gg.vs["weight"][index_top]
            # delete top vertice (weighted degrees are automatically updated)
            gg.delete_vertices(index_top)

            if len(neighbors_top) > 0:
                        # iterate over neighbors of top element
                for i, name_n in enumerate(neighbors_top):
                    index_n = gg.vs["name"].index(name_n)
                    max_n = max(cores_g[top], gg.strength(
                        weights=gg.es["weight"])[index_n])
                    gg.vs[index_n]["weight"] = max_n
                    # update heap
                    heap_g = zip(gg.vs["weight"], gg.vs["name"])
                    heapq.heapify(heap_g)
            else:
                # update heap
                heap_g = zip(gg.vs["weight"], gg.vs["name"])
                heapq.heapify(heap_g)

    else:
        # k-core decomposition for unweighted graphs
        # based on Batagelj and Zaversnik's (2002) algorithm #1
        cores_g = dict(zip(gg.vs["name"], g.coreness()))

    # sort vertices by decreasing core number
    sorted_cores_g = sorted(
        cores_g.items(), key=operator.itemgetter(1), reverse=True)

    return(sorted_cores_g)


def sum_numbers_neighbors(g, names_numbers):
    # if used with core numbers, implements CoreRank (Tixier et al. EMNLP 2016)
    # initialize dictionary that will contain the scores
    name_scores = dict(zip(g.vs['name'], [0] * len(g.vs['name'])))
    # iterate over the nodes
    for name_number in names_numbers:
        name = name_number[0]
        # find neighbor names
        neighbor_names = g.vs[g.neighbors(name)]['name']
        # sum up the scores of the neighbors
        neighbor_tuples = [
            elt for elt in names_numbers if elt[0] in neighbor_names]
        sum_numbers_neighbors = sum([elt[1] for elt in neighbor_tuples])
        # save result
        name_scores[name] = sum_numbers_neighbors

    # sort by decreasing score number
    sorted_name_scores = sorted(
        name_scores.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_name_scores


# examples

# read text
# example file can be found here:
# https://github.com/Tixierae/abs_meet_summ/blob/master/data/abstract_Hulth_2003.txt
# with open("abstract_Hulth_2003.txt", "r") as my_file:
#     text = my_file.read().splitlines()
# text = " ".join(text)

# # preprocess text
# all_terms = clean_text_simple(text)

# # get graph of terms
# g = terms_to_graph(all_terms, w=10)

# # get weighted core numbers
# sorted_cores_g = core_dec(g, weighted=True)

# # get CoreRank scores
# core_rank_scores = sum_numbers_neighbors(g, sorted_cores_g)

# print core_rank_scores
# print g.vs['weight']
# edge_weights = []
# for edge in g.es:
#     source = g.vs[edge.source]['name']
#     target = g.vs[edge.target]['name']
#     weight = edge['weight']
#     edge_weights.append([source, target, weight])
# print(edge_weights)
