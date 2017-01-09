# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:22:52 2017

@author: ding_wensi
"""

import core_rank as cr
##########################################################################
sentences = ["The/DT wife/NN of/IN a/DT former/JJ U.S./NNP president/NN \
Bill/NNP Clinton/NNP Hillary/NNP Clinton/NNP visited/VBD China/NNP last/JJ \
Monday/NNP ./PUNCT", "Hillary/NNP Clinton/NNP wanted/VBD to/TO visit/VB China/NNP \
last/JJ month/NN but/CC postponed/VBD her/PRP$ plans/NNS till/IN Monday/NNP \
last/JJ week/NN ./PUNCT", "Hillary/NNP Clinton/NNP paid/VBD a/DT visit/NN to/TO \
the/DT People/NNP Republic/NNP of/IN China/NNP on/IN Monday/NNP ./PUNCT",
             "Last/JJ week/NN the/DT Secretary/NNP of/IN State/NNP Ms./NNP Clinton/NNP \
visited/VBD Chinese/JJ officials/NNS ./PUNCT"]

def concat (sentences):
    sentences = ' '.join(sentences)
    words = sentences.split(' ')
    words = [word.split("/")[0] for word in words]
    sentences = ' '.join(words)
    return sentences
##########################################################################
text = concat(sentences)
all_terms = cr.clean_text_simple(text,pos_filtering=False, stemming=False)
# get graph of terms
    
g = cr.terms_to_graph(all_terms, w=10)

# get weighted core numbers
sorted_cores_g = cr.core_dec(g, weighted=True)

# get CoreRank scores
core_rank_scores = cr.sum_numbers_neighbors(g, sorted_cores_g)