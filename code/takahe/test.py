#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.chdir('C:\\Users\\mvazirg\\Documents\\abs_meet_summ\\code\\takahe')

import zekun as compression
import string
from nltk import pos_tag
# we use TweetTokenizer because unlike word_tokenize it does not split contractions (e.g., didn't-> did n't)
from nltk.tokenize import TweetTokenizer
from lan_model import language_model

my_tokenizer = TweetTokenizer()

# Create a language model
my_lm = language_model(model_path='C:\\Users\\mvazirg\\Documents\\en-70k-0.2.lm')

#my_lm = language_model(model_path='d:\\3A\\Projet3A\\project\\build_graph\\graph_build\\takahe\\en-70k-0.2.lm')

punct = string.punctuation

#### examples of clusters of well-formed sentences ####

sentences=["Lonesome George, the world's last Pinta Island giant tortoise, has passed away","The giant tortoise known as Lonesome George died Sunday at the Galapagos National Park in Ecuador.", "He was only about a hundred years old, but the last known giant Pinta tortoise, Lonesome George, has passed away.", "Lonesome George, a giant tortoise believed to be the last of his kind, has died."]

sentences = ['The wife of a former U.S. president Bill Clinton Hillary Clinton visited China last Monday','Hillary Clinton wanted to visit China last month but postponed her plans till Monday last week.','Hillary Clinton paid a visit to the People Republic of China on Monday.', 'Last week the Secretary of State Ms. Clinton visited Chinese officials.']

sentences = ['the meeting is about the design of a remote control','today, we will focus on the remote control','the production cost and price of the remote are two important parameters', 'design decisions will impact the price of the remote control',"today's meeting deals with designing the remote control",'the topic today is the remote control']

sentences = ['my favourite color is blue', 'do you like red?','I think red is a nice warm color', 'we need to decide about the colors',"choosing the colors won't be easy, but blue is quite nice"]


#### examples of clusters of utterances ####

path_to_comms = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ\\data\\datasets\\meeting_summarization\\ami_icsi\\communities\\ami\\'

with open(path_to_comms + 'IS1003a_comms.txt','r') as file:
    sentences_all_comms = file.read().splitlines()
	
# iterate through the sentences and do the splitting

# list of lists
comms = []
comm = []

for sentence in sentences_all_comms:
    if sentence != '':
        comm.append(sentence)
    else:
        comms.append(comm)
        comm = []

# retain communities with at least two sentences in them (for the others, compression is obviously not necessary)
big_comms = [comm for comm in comms if len(comm)>1]

sentences = big_comms[2]

#### put sentences in the right format ####

tagged_sentences = []
for sentence in sentences:
    tagged_sentence = []
    tokens = my_tokenizer.tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    for tuple in tagged_tokens:
        if tuple[1] in punct:
            tagged_sentence.append('/'.join([tuple[0],'PUNCT']))
        else:
            tagged_sentence.append('/'.join(tuple))
    tagged_sentences.append(' '.join(tagged_sentence))

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


##########################################################################

# Create a word graph from the set of sentences with parameters :
# - minimal number of words in the compression : 6
# - language of the input sentences : en (english)
# - POS tag for punctuation marks : PUNCT
compresser = compression.word_graph(tagged_sentences,
                                    nb_words=10,
                                    lang='en',
                                    punct_tag="PUNCT", model=my_lm)

# Write the word graph in the dot format
# compresser.write_dot('new.dot')

# Get the 50 best paths
candidates = compresser.get_compression(200)

final_paths = compresser.final_score(candidates)

for i in range(len(final_paths))[:10]:
    print final_paths[i][0], final_paths[i][1]
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
