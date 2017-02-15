#!/usr/bin/python
# -*- coding: utf-8 -*-

import zekun as compression
import string
from nltk import pos_tag

def separate_punct(token):
    punct_in_token = list(set([l for l in token if l in punct]))
    
    if punct_in_token:
    
        # separate punctuation marks from token
        new_tokens = []
        new_token = ''
        for l in token:
            if l in punct_in_token:
                new_tokens.append(new_token)
                new_tokens.append(l)
                new_token = ''
            else:
                new_token = new_token + l
    
    else:
        new_tokens = [token]
    
    # remove empty elements
    new_tokens = [elt for elt in new_tokens if len(elt)>0]
    
    return new_tokens

# do not consider dashes and apostrophes
punct = string.punctuation.replace('-', '').replace("'", '')

sentences=["Lonesome George, the world's last Pinta Island giant tortoise, has passed away","The giant tortoise known as Lonesome George died Sunday at the Galapagos National Park in Ecuador.", "He was only about a hundred years old, but the last known giant Pinta tortoise, Lonesome George, has passed away.", "Lonesome George, a giant tortoise believed to be the last of his kind, has died."]

tagged_sentences = []
for sentence in sentences:
    tagged_sentence = []
    tokens = sentence.split(' ')
    tokens_punct = [separate_punct(token) for token in tokens]
    tokens_punct = [item for sublist in tokens_punct for item in sublist]
    tagged_tokens = pos_tag(tokens_punct)
    for tuple in tagged_tokens:
        if tuple[1] in punct:
            tagged_sentence.append('/'.join([tuple[0],'PUNCT']))
        else:
            tagged_sentence.append('/'.join(tuple))
    tagged_sentences.append(' '.join(tagged_sentence))


<<<<<<< HEAD
print tagged_sentences
=======
>>>>>>> bf793d5e09edeab94c1411daacc53e4d8ac484e8
##########################################################################
# sentences = ["The/DT wife/NN of/IN a/DT former/JJ U.S./NNP president/NN \
# Bill/NNP Clinton/NNP Hillary/NNP Clinton/NNP visited/VBD China/NNP last/JJ \
# Monday/NNP ./PUNCT", "Hillary/NNP Clinton/NNP wanted/VBD to/TO visit/VB China/NNP \
# last/JJ month/NN but/CC postponed/VBD her/PRP$ plans/NNS till/IN Monday/NNP \
# last/JJ week/NN ./PUNCT", "Hillary/NNP Clinton/NNP paid/VBD a/DT visit/NN to/TO \
# the/DT People/NNP Republic/NNP of/IN China/NNP on/IN Monday/NNP ./PUNCT",
#              "Last/JJ week/NN the/DT Secretary/NNP of/IN State/NNP Ms./NNP Clinton/NNP \
# visited/VBD Chinese/JJ officials/NNS ./PUNCT"]
##########################################################################

# Create a word graph from the set of sentences with parameters :
# - minimal number of words in the compression : 6
# - language of the input sentences : en (english)
# - POS tag for punctuation marks : PUNCT
compresser = compression.word_graph(tagged_sentences,
                                    nb_words=6,
                                    lang='en',
                                    punct_tag="PUNCT")

# Write the word graph in the dot format
# compresser.write_dot('new.dot')

# Get the 50 best paths
candidates = compresser.get_compression(50)

print compresser.final_score(candidates)[:2]
# # 1. Rerank compressions by path length (Filippova's method)
# for cummulative_score, path in candidates:

#     # Normalize path score by path length
#     normalized_score = cummulative_score / len(path)

#     # Print normalized score and compression
#     print (round(normalized_score, 3), ' '.join([u[0] for u in path]))



# # 2. Rerank compressions by keyphrases (Boudin and Morin's method)
# reranker = compression.keyphrase_reranker(sentences,
#                                           candidates,
#                                           lang='en')

# reranked_candidates = reranker.rerank_nbest_compressions()

# # Loop over the best reranked candidates
# for score, path in reranked_candidates:

#     # Print the best reranked candidates
#     print (round(score, 3), ' '.join([u[0] for u in path]))
