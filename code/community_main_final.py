import pandas as pd
import numpy as np
import string
import re
import igraph
import gensim
from collections import Counter
import nltk

#################
### FUNCTIONS ###
#################

# to be used within 'sort'
def getKey(item):
    return item[1]

def clean_utterances(utterances_list, punct, stopwords_list):
    cleaned_utterances = []
    for element in utterances_list:
        utt = element[1]
        # convert to lower case
        utt = utt.lower()
        # remove punctuation
        utt = ''.join(l for l in utt if l not in punct)
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

def create_communities(utt_tuples, remove_single=True, my_p=0.85, n_steps=6, big_comm=False):
    # utt_tuples: list of tuples (utterance_index, utterance)
    # remove_single: should utterancse with only one word be removed?
    # my_p: the my_p % edges with lowest WMD will be kept. If 1, all edges are kept
    # n_steps: length of the random walks in walktrap community detection algorithm
    # big_comm: should communities with only one single node (i.e., utterance) be removed?
    
    if remove_single:
        # remove utterances with only one single word
        utt_tuples = [elt for elt in utt_tuples if len(elt[1].split(' '))>1]
    
    # create complete graph of utterances
    g = igraph.Graph.Full(n=len(utt_tuples), directed=False, loops=False)
    
    # compute edge weights based on WMD
    edge_weights = []
    i = 0
    
    for edge in g.es:
    	source_id = edge.source
    	target_id = edge.target
    
    	utt_source = utt_tuples[source_id]
    	utt_target = utt_tuples[target_id]
    
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
    
    edge_weights_np = np.array(edge_weights)
    
    # value below which my_p % of the weights fall
    my_threshold = np.percentile(edge_weights_np, my_p)
            
    # get index of edges associated with large distances (to be removed)
    edge_weights_weak_index = [j for j,k in enumerate(edge_weights) if k>my_threshold]
            
    # get weights of the strong edges
    edge_weights_strong = [elt for elt in edge_weights if elt<=my_threshold]
            
    # sanity check
    if not len(edge_weights_weak_index)+len(edge_weights_strong)-len(edge_weights) == 0:
        print '1st sanity check failed'
        return
            
    # delete weak edges
    g.delete_edges(edge_weights_weak_index)
    
    # we invert edge weights because the WMD is a distance (the smaller the 'better')
    # whereas for edge weights, it is: the greater the 'better'
        
    edge_weights_strong_inverted = [1/elt for elt in edge_weights_strong]
    
    # assign weights to the edges kept
    g.es['weight'] = edge_weights_strong_inverted
            
    # detect communities
    dendogram = g.community_walktrap(weights=g.es['weight'], steps=n_steps)
    
    # convert it into a flat clustering
    clustering = dendogram.as_clustering()
    
    # get the membership vector
    membership = clustering.membership
    
    # c.items() contains the communities and their size (# of vertices, i.e., # of utt)
    c = Counter(membership)
        
    #sanity check
    if not sum([elt[1] for elt in c.items()]) == len(utt_tuples):
        print '2nd sanity check failed'
        return
    
    if big_comm:
         # select the communities containing more than one utterance
        c_items = [elt for elt in c.items() if elt[1]>1]
    else:
        c_items = c.items()
    
    c_items = sorted(c_items, key=getKey, reverse=True)
            
    return c_items, membership

####################
### DATA LOADING ###
####################

path_root = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ'

path_to_data = path_root + '\\data\\datasets\\meeting_summarization\\ami_icsi'

# read IDs of training set meetings from AMI corpus
with open(path_to_data + '\\lists\\list.ami.train', 'r+') as txtfile:
    ami_train_ids = txtfile.read().splitlines()

# read IDs of training set meetings from ICSI corpus
with open(path_to_data + '\\lists\\list.icsi.train', 'r+') as txtfile:
    icsi_train_ids = txtfile.read().splitlines()

# load Google News word vectors 
# ! Uses approx. 6GB of RAM
model = gensim.models.word2vec.Word2Vec.load_word2vec_format('E:\\GoogleNews-vectors-negative300.bin.gz', binary=True)  

# traditional stopwords
#nltk.download('stopwords')
stpwds = nltk.corpus.stopwords.words("english")

# custom stopwords
with open(path_to_data + '\\communities\\stopwords\\stopwords.txt', 'r+') as txtfile:
    cus_stpwds = txtfile.read().splitlines()

# filler words
with open(path_to_data + '\\communities\\stopwords\\filler_words.csv', 'r+') as txtfile:
    filler = txtfile.read().splitlines()

# merge, removing duplicates
stopwords = list(set(cus_stpwds + filler + stpwds))

punct = string.punctuation

##########################
### COMMUNITY CREATION ###
##########################

ami_or_icsi = 'icsi'

if ami_or_icsi=='ami':
    my_ids = ami_train_ids
elif ami_or_icsi=='icsi':
    my_ids = icsi_train_ids

for kk in range(len(my_ids)):

    # load ASR output
    asr_output = pd.read_csv(path_to_data + '\\' + ami_or_icsi + '\\' + my_ids[kk] + '.da-asr',
                             sep='\t', 
                             header=None, 
                             names = ['ID','start','end','letter','role','A','B','C','utt'])
    
    if asr_output.shape[0]==0:
        print 'empty file'
        continue
    
    # add column containing duration of utterances
    asr_output['duration'] = asr_output['end'] - asr_output['start']
    
    # add column row indices
    asr_output['index'] = range(asr_output.shape[0])
    
    # retain utterances whose duration exceeds 0.85s
    asr_output_cleaned = asr_output[asr_output['duration']>0.85]
    
    utterances = zip(asr_output_cleaned['index'].tolist(),asr_output_cleaned['utt'].tolist())
    
    utterances_processed = clean_utterances(utterances, punct, stopwords)
    
    c_items, membership = create_communities(utterances_processed)
    
    # go back to the original utterances
    # write the communities to file
    # one row per utterance - one blank row between each community
    
    with open(path_to_data + '\\communities\\' + ami_or_icsi + '\\' + my_ids[kk] + '_comms.txt', 'w+') as txtfile:
        for k in range(len(c_items)):
            comm_object = c_items[k]
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
            
    print kk+1, 'file(s) done'