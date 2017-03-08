
#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.chdir('D:\\3A\\Projet3A\\project\\abs_meet_summ\\code\\takahe')
# os.chdir('C:\\Users\\mvazirg\\Documents\\abs_meet_summ\\code\\takahe')
import time
import takahe_params_tuning as compression
import string
from nltk import pos_tag
# we use TweetTokenizer because unlike word_tokenize it does not split contractions (e.g., didn't-> did n't)
from nltk.tokenize import TweetTokenizer
from lan_model import language_model
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances as ed


my_tokenizer = TweetTokenizer()


# Create a language model
print "loading language model..."
start = time.time()
# my_lm = language_model(model_path='C:\\Users\\mvazirg\\Documents\\en-70k-0.2.lm')
my_lm = language_model(model_path='d:\\3A\\Projet3A\\project\\data\\en-70k-0.2.lm')
elapse = time.time() - start
print "finish loading language model, time_cost = %.2fs" % elapse

punct = string.punctuation

#### examples of clusters of well-formed sentences ####

sentences=["Lonesome George, the world's last Pinta Island giant tortoise, has passed away","The giant tortoise known as Lonesome George died Sunday at the Galapagos National Park in Ecuador.", "He was only about a hundred years old, but the last known giant Pinta tortoise, Lonesome George, has passed away.", "Lonesome George, a giant tortoise believed to be the last of his kind, has died."]

# sentences = ['The wife of a former U.S. president Bill Clinton Hillary Clinton visited China last Monday','Hillary Clinton wanted to visit China last month but postponed her plans till Monday last week.','Hillary Clinton paid a visit to the People Republic of China on Monday.', 'Last week the Secretary of State Ms. Clinton visited Chinese officials.']

# sentences = ['the meeting is about the design of a remote control','today, we will focus on the remote control','the production cost and price of the remote are two important parameters', 'design decisions will impact the price of the remote control',"today's meeting deals with designing the remote control",'the topic today is the remote control']

# sentences = ['my favourite color is blue', 'do you like red?','I think red is a nice warm color', 'we need to decide about the colors',"choosing the colors won't be easy, but blue is quite nice"]


#### examples of clusters of utterances ####

# path_to_comms = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ\\data\\datasets\\meeting_summarization\\ami_icsi\\communities\\ami\\'
path_to_comms = 'D:\\3A\\Projet3A\\project\\abs_meet_summ\\data\\datasets\\meeting_summarization\\ami_icsi\\communities\\ami\\'

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


lists_of_tokens = [sent.split(' ') for sent in sentences]

lists_of_tokens_flatten = [item for sublist in lists_of_tokens for item in sublist]

lotf=lists_of_tokens_flatten

vectors = Word2Vec(size=3e2, min_count=1)
vectors.build_vocab(lists_of_tokens)

path_to_wv = 'D:\\3A\\Projet3A\\project\\data\\' # to fill

print "loading GoogleNews..."
start = time.time()
vectors.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True) 
elapse = time.time() - start
print "finish loading GoogleNews, time_cost = %.2fs" % elapse
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

graph_type = [1,1,1]
word_attraction = [1,1,1]
keyphrase = [0,0,0]
fl_score = [1,0,0]
core_rank = [0,1,0]
word_embed = [0,0,1]

for i in range(len(graph_type)):
	print "graph_type=%.1f, word_attraction=%.1f, keyphrase=%.1f, fl_score=%.1f, core_rank=%.1f, word_embed=%.1f " % (graph_type[i], word_attraction[i], keyphrase[i], fl_score[i], core_rank[i], word_embed[i])

	compresser = compression.word_graph(tagged_sentences,model=my_lm, vectors=vectors, lotf=lotf,graph_type=graph_type[i], word_attraction=word_attraction[i],keyphrase=keyphrase[i],fl_score=fl_score[i],core_rank=core_rank[i],word_embed=word_embed[i],num_cluster=5, domain=True, nb_words=10,lang='en',punct_tag="PUNCT", pos_separator='/', cr_w = 10, cr_weighted = True, cr_pos_filtering = False, cr_stemming = False)

	# Write the word graph in the dot format
	# compresser.write_dot('new.dot')

	# Get the 50 best paths
	candidates = compresser.get_compression(200)

	final_paths = compresser.final_score(candidates,10)

	for i in range(len(final_paths))[:10]:
	    print final_paths[i][0], final_paths[i][1]
