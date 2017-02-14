import math
import pynlpl.lm.lm

#sentence='hey how are you'
#get_n_grams(sentence, n=3)
#[('hey',),
# ('hey', 'how'),
# ('hey', 'how', 'are'),
# ('how', 'are', 'you'),
# ('are', 'you'),
# ('you',)]
def get_n_grams(sentence, n):
    ss = sentence.split()
    if len(ss)<n:
        print 'order exceeds total number of words'
        n_grams = []
    else:
        n_grams, k = [], 0
        for i in range(len(ss)):
            if k >= n:
                break
            n_grams.append(tuple(ss[:i+1]))
            k += 1
        for i in range(1,len(ss)):
            n_grams.append(tuple(ss[i:i+n]))
    return n_grams

# ! ! ! note that the sentence scores are not always in [0,1]
# (because the score for a sentence is just the sum of the probabilities of all the n-grams it contains)
# so, you need to normalize the scores of the sentences in the community before plugging those scores into the overall ranking
# for instance, you can divide all scores by the max score


def get_sentence_score(sentence, my_model, n, unknownwordprob=0):
    score = 0
    n_grams = get_n_grams(sentence, n)
    if n_grams:
        for n_gram in n_grams:
            try:
                score += math.exp(my_model.ngrams.prob(n_gram))
            except KeyError:
                score += unknownwordprob
        return score      
    else:
        print 'order exceeds total number of words'
        return

#####################################

# download a pre-trained language model from here: https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/   
# you want to look at the files with .lm extension (standard ARPA format)
# I downloaded 'en-70k-0.2.lm.gz', but it takes about 6GB of RAM once loaded
# if you don't have enough memory you can try with the pruned version ('en-70k-0.2-pruned.lm.gz') but performance will be slightly inferior

# mode='simple' is faster but loads everything into memory
# if you have a C++ compiler installed, you can try mode='trie' (uses 35% less memory)
#path_to_file= # ! fill me
self.my_lm = pynlpl.lm.lm.ARPALanguageModel(filename='d:\\3A\\Projet3A\\project\\build_graph\\graph_build\\takahe\\en-70k-0.2.lm',mode='simple')

# this is a trigram language model so we don't want to exceed n=3

# good sentence
get_sentence_score(sentence='the meeting is about the design of a remote control', my_model=my_lm, n=3)
# should return 0.7

# bad sentence
get_sentence_score(sentence='important also the price and remote', my_model=my_lm, n=3)
# should return 0.025