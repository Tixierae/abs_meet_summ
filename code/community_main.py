#import math
import pandas as pd
import numpy as np
import string
import re
import igraph
import gensim
from matplotlib import pyplot as plt
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

my_vectorizer = TfidfVectorizer(stop_words=None)

nltk.download('stopwords')

# to be used within 'sort'
def getKey(item):
    return item[1]

def clean_doc(doc, punct):
    doc = doc.lower()
    # remove formatting
    doc =  re.sub('\s+', ' ', doc)
    # remove punctuation (preserving dashes)
    doc = ''.join(l for l in doc if l not in punct)
    # remove dashes that are not intra-word
    doc = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), doc)
    # strip extra white space
    doc = re.sub(' +',' ',doc)
    # strip leading and trailing white space
    doc = doc.strip()
    return doc

def clean_utterances(utterances_list, punct, stopwords_list):
    cleaned_utterances = []
    for element in utterances_list:
        utt = element[1]
        # convert to lower case
        utt = utt.lower()
        # remove punctuation (preserving dashes)
        utt = ''.join(l for l in utt if l not in punct)
#        # remove punctuation (preserving dashes)
#        utt = ''.join(l for l in utt if l not in punct)
#        # remove dashes that are not intra-word
#        utt = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), utt)
        # strip extra white space
        utt = re.sub(' +',' ',utt)
        # strip leading and trailing white space
        utt = utt.strip()
        # replace consecutive unigrams with a single instance
        utt = re.sub('\\b(\\w+)\\s+\\1\\b', '\\1',utt)
        # same for bigrams
        utt = re.sub('(\\b.+?\\b)\\1\\b', '\\1',utt)
        # tokenize
        tokens = utt.split(' ')
        # remove stopwords
        tokens = [token for token in tokens if token not in stopwords_list]
        # keep only tokens with at least two characters
        tokens = [token for token in tokens if len(token)>=2]
        # reconstruct utterance
        utt = ' '.join(tokens)
        cleaned_utterances.append((element[0],utt))
    # remove empty utterances
    cleaned_utterances = [elt for elt in cleaned_utterances if len(elt[1])>0]    
    return(cleaned_utterances)

def clean_utterance_final(utterance, filler_words):
        utt = utterance.lower()
        # replace consecutive unigrams with a single instance
        utt = re.sub('\\b(\\w+)\\s+\\1\\b', '\\1',utt)
        # same for bigrams
        utt = re.sub('(\\b.+?\\b)\\1\\b', '\\1',utt)
        # convert to lower case
        # strip extra white space
        utt = re.sub(' +',' ',utt)
        # strip leading and trailing white space
        utt = utt.strip()
        # remove filler words
        tokens = [token for token in utt.split(' ') if token not in filler_words]
        # reconstruct utterance
        utt = ' '.join(tokens)
        return utt
        
##################

path_root = 'C:\\Users\\mvazirg\\Documents\\text_stream_topic_detection_most_recent\\data\\'

#punct = string.punctuation.replace('-', '')
punct = string.punctuation
# regex to remove intra-word dashes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

# load Google News word vectors !!! Uses approx. 6GB of RAM
model = gensim.models.word2vec.Word2Vec.load_word2vec_format('E:\\GoogleNews-vectors-negative300.bin.gz', binary=True)  

##################

# traditional English stopwords
stpwds = nltk.corpus.stopwords.words("english")

### build custom list of stopwords ###

# read IDs of training set meetings from AMI corpus
with open(path_root + 'lists\\list.ami.train', 'r+') as txtfile:
    ami_train_ids = txtfile.read().splitlines()

full_text_ami = []
for counter in range(len(ami_train_ids)):
    asr_output = pd.read_csv(path_root + 'ami\\' + ami_train_ids[counter] + '.da-asr',
                             sep='\t', 
                             header=None, 
                             names = ['ID','start','end','letter','role','A','B','C','utt'])
    full_text_ami.append(' '.join(asr_output['utt'].tolist()))

# read IDs of training set meetings from ICSI corpus
with open(path_root + 'lists\\list.icsi.train', 'r+') as txtfile:
    icsi_train_ids = txtfile.read().splitlines()

full_text_icsi = []
for counter in range(len(icsi_train_ids)):
    asr_output = pd.read_csv(path_root + 'icsi\\' + icsi_train_ids[counter] + '.da-asr',
                             sep='\t', 
                             header=None, 
                             names = ['ID','start','end','letter','role','A','B','C','utt'])
    full_text_icsi.append(' '.join(asr_output['utt'].tolist()))


full_text = full_text_ami + full_text_icsi

full_text_cleaned = [clean_doc(elt, punct) for elt in full_text]

doc_term_matrix = my_vectorizer.fit_transform(full_text_cleaned)

# each column is a term, each entry is the tf-idf weight of the word in the doc corresponding to the row
doc_term_matrix.shape

terms_total_weights = doc_term_matrix.sum(axis=0)

# convert to list
terms_total_weights = np.array(terms_total_weights)[0].tolist()

# get actual terms (column names)
col_names = my_vectorizer.get_feature_names()

terms_total_weights = zip(col_names, terms_total_weights)

# sort by increasing order (lowest values first)
sorted_terms = sorted(terms_total_weights, key=lambda x: x[1], reverse=True)

# print top 500 (that are not standard stopwords) to file and clean manually
i = 0
with open(path_root + '\\communities\\stopwords\\stopwords.txt', 'w+') as txtfile:
    for tuple in sorted_terms:
        if tuple[0] not in stpwds:
            txtfile.write(tuple[0] + '\n')
            i+=1
            if i>=500:
                break

# load manually cleaned stopwords
with open(path_root + '\\communities\\stopwords\\stopwords.txt', 'r+') as txtfile:
    cus_stpwds = txtfile.read().splitlines()

# load filler words
with open(path_root + 'output\\filler_words.csv', 'r+') as txtfile:
    filler = txtfile.read().splitlines()

# merge, removing duplicates
stopwords = list(set(cus_stpwds + filler + stpwds))

##############################

counter = 10

# load ASR output
asr_output = pd.read_csv(path_root + 'ami\\' + ami_train_ids[counter] + '.da-asr',
                         sep='\t', 
                         header=None, 
                         names = ['ID','start','end','letter','role','A','B','C','utt'])

# add column containing duration of utterances
asr_output['duration'] = asr_output['end'] - asr_output['start']

# add column row indices
asr_output['index'] = range(asr_output.shape[0])

# inspect (.loc is label based, iloc index based)
asr_output.iloc[:6,8:11]

# retain utterances whose duration exceeds 0.85s
asr_output_cleaned = asr_output[asr_output['duration']>0.85]

# inspect
asr_output_cleaned.iloc[:6,8:11]

utterances = zip(asr_output_cleaned['index'].tolist(),asr_output_cleaned['utt'].tolist())

utterances_processed = clean_utterances(utterances, punct, stopwords)

# remove utterances with only one single word - optional
utterances_processed = [elt for elt in utterances_processed if len(elt[1].split(' '))>1]

# create complete graph of utterances
g = igraph.Graph.Full(n=len(utterances_processed), directed=False, loops=False)

# compute edge weights based on WMD
edge_weights = []
i = 0

for edge in g.es:
  source_id = edge.source
  target_id = edge.target
  
  utt_source = utterances_processed[source_id]
  utt_target = utterances_processed[target_id]
  
  text_source = utt_source[1].split(' ')
  text_target = utt_target[1].split(' ')
  
  if len(text_source) == len(text_target) == 1:
      if text_source[0] == text_target[0]:
          distance = 0
      else:
          distance = model.wmdistance(text_source,text_target)
  else:
      distance = model.wmdistance(text_source,text_target)
    
  edge_weights.append(distance)
  
  i+=1
  if i%1e4==0:
      print i, 'edges processed'

# find inf values (corresponding to single out-of-vocab words)
index_inf = [k for k in range(len(edge_weights)) if edge_weights[k] == float('inf')]

# replace those values with the max distance
edge_weights_wo_inf = [k for j,k in enumerate(edge_weights) if j not in index_inf]

max_distance = max(edge_weights_wo_inf)

for k in index_inf:
   edge_weights[k] = max_distance

# add a small quantity to avoid zero values
edge_weights = [elt + 1e-3 for elt in edge_weights]

# inspect
plt.hist(edge_weights)

# we could determine the WMD threshold ('my_threshold') based on the
# density of an undirected graph (max number of edges allowed)
# e.g., 2*len(g.es)/(len(g.vs)*(len(g.vs)-1.0))

# automatically set the threshold to some quantile
# (e.g., value below which only 0.8% of the observations lay)
# this is an important tuning parameter

my_step=6

my_q = 0.85

for my_q in np.arange(0.05,1.05,0.05):
    for my_step in range(2,9):
    
        g = igraph.Graph.Full(n=len(utterances_processed), directed=False, loops=False)
        
        edge_weights_np = np.array(edge_weights)
        my_threshold = np.percentile(edge_weights_np, my_q)
        
        # get index of edges associated with high distances (to be removed)
        edge_weights_weak_index = [j for j,k in enumerate(edge_weights) if k>my_threshold]
        
        # get weights of the strong edges
        edge_weights_strong = [elt for elt in edge_weights if elt<=my_threshold]
        
        # sanity check
    #    if len(edge_weights_weak_index)+len(edge_weights_strong)-len(edge_weights) == 0:
    #        print '1st sanity check passed'
    #    else:
    #        print '1st sanity check failed'
        
        # delete weak edges
        g.delete_edges(edge_weights_weak_index)
        
        # we invert edge weights because the WMD is a distance (the smaller the 'better')
        # whereas for edge weights, it should be: the greater the 'better'
        # edge_weights_strong_inverted = [math.log(1+1/elt) for elt in edge_weights_strong]
        
        edge_weights_strong_inverted = [1/elt for elt in edge_weights_strong]
        
        # assign weights to the edges kept
        g.es['weight'] = edge_weights_strong_inverted
        
        # edge_weights_inverted = [math.log(1+1/elt) for elt in edge_weights]
        # clustering = g.community_fastgreedy(weights=g.es['weight'])
        # clustering = g.community_walktrap(weights=g.es['weight'], steps=4)
        # clustering = g.community_infomap(edge_weights=g.es['weight'], vertex_weights=None, trials=10)
        # multilevel is also called Louvain algorithm
        #clustering = g.community_multilevel(weights=g.es['weight'])
        
        # detect communities
        dendogram = g.community_walktrap(weights=g.es['weight'], steps=my_step)
        # convert it into a flat clustering
        clustering = dendogram.as_clustering()
        #clustering = g.community_infomap(edge_weights=g.es['weight'], vertex_weights=None, trials=10)
        # get the membership vector
        membership = clustering.membership
        
        c = Counter(membership)
        
        # c.items() contains the communities and their size (number of vertices)
        
        # sanity check
    #    if sum([elt[1] for elt in c.items()]) == len(utterances_processed):
    #        print '2nd sanity check passed'
    #    else:
    #        print '2nd sanity check failed'
        
        # select the communities containing more than one element
        c_items_big = [elt for elt in c.items() if elt[1]>=1]
        
        c_items_big = sorted(c_items_big, key=getKey, reverse=True)
        
        print 'n steps=', my_step
        print 'quantile=', my_q
        print 'single/total:', len([elt for elt in c_items_big if elt[1]==1])/float(len(c_items_big))

for k in range(len(c_items_big)):
    comm_object = c_items_big[k]
    comm_id = comm_object[0]
    index_comm = [l for l in range(len(membership)) if membership[l]==comm_id]
    print '\n' + str([utterances_processed[l] for l in index_comm])
    
# go back to the original utterances
# write the communities to file one row per utterance - two rows between each comm

# use the loop below:
with open(path_root + '\\communities\\ami\\' + ami_train_ids[counter] + '_comms.txt', 'w+') as txtfile:
    for k in range(len(c_items_big)):
        comm_object = c_items_big[k]
        comm_id = comm_object[0]
        index_comm = [l for l in range(len(membership)) if membership[l]==comm_id]
        for m in index_comm:
            index_utterance = utterances_processed[m][0]
            to_write = [elt[1] for elt in utterances if elt[0]==index_utterance][0]
            to_write = clean_utterance_final(to_write,filler_words=filler)
            # one utterance per line
            txtfile.write(to_write + '\n')
        # separate communities by white line
        txtfile.write('\n')
        
        
# TODO: add in, on, one to lists of stopwords
# pre (or post?) processing can be improved: the utterances that do not contain
# any keyword can be removed...
        
# or maybe some sort of a community ranking...
        
# TODO: change directory (even just with example files) to github `abs_meet_summ`