import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import string
import re 
import itertools
import copy
import igraph
import nltk
from collections import Counter
import operator
import heapq
# requires nltk 3.2.1
#from nltk import pos_tag

#################
### FUNCTIONS ###
#################

def clean_utterances(utterances_list, punct, my_regex,  stopwords_list):
    cleaned_utterances = []
    all_tokens = []
    for element in utterances_list:
        utt = element[1]
        #utt = element
        # convert to lower case
        utt = utt.lower()
        # remove punctuation
        utt = ''.join(l for l in utt if l not in punct)
        # remove dashes and apostrophes that are not intra-word
        utt = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), utt)
        # strip extra white space
        utt = re.sub(' +',' ',utt)
        # strip leading and trailing white space
        utt = utt.strip()
        # tokenize
        tokens = utt.split(' ')
        # remove stopwords
        tokens = [token for token in tokens if token not in stopwords_list]
        # keep only tokens with at least two characters
        tokens = [token for token in tokens if len(token)>=2]
        stemmer = nltk.stem.PorterStemmer()
        # stem
        tokens = [stemmer.stem(token) for token in tokens]
        # reconstruct utterance
        utt = ' '.join(tokens)
        cleaned_utterances.append((element[0],utt))
        all_tokens.append(tokens)
    # remove empty utterances
    cleaned_utterances = [elt for elt in cleaned_utterances if len(elt[1])>0]
    # flatten tokens
    all_tokens = [item for sublist in all_tokens for item in sublist]
    return all_tokens, cleaned_utterances

def cluster_utterances(utterances_processed, remove_single, n_comms, max_elt):
    '''
    remove_single: whether to remove utterances with only one word
    n_comms: desired number of communities
    max_elt: maximum number of utterances allowed per community
    '''
    utt_tuples = utterances_processed
    
    if remove_single:
        # remove utterances with only one single word
        utt_tuples = [elt for elt in utt_tuples if len(elt[1].split(' '))>1]
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=None,ngram_range=(1,3))
        
    tdm = tfidf_vectorizer.fit_transform([elt[1] for elt in utt_tuples])
        
    membership = KMeans(n_clusters=n_comms, n_init=50).fit_predict(tdm)
    
    c = dict(Counter(membership))
    
    # get the IDs of the comms with more than 'max_elt' elements    
    big_comms_ids = [k for k,v in c.iteritems() if v>=max_elt]
    
    n_comm_split_total = 0
    
    while big_comms_ids:
            
        max_c_idx = max(c.keys())
        
        # iterate through the big comms
        for id in big_comms_ids:
            # get idx of the utterances belonging to the big comm
            idxs = [idx for idx, label in enumerate(membership) if label == id]
            
            # sanity check
            if not len(idxs) == sum([v for k,v in c.iteritems() if k == id]):
                print '2nd sanity check failed!'
                #return
            
            # collect corresponding utterances
            utt_tuple_comm = [tuple for idx, tuple in enumerate(utt_tuples) if idx in idxs]
            
            # decide into how many pieces the big comm should be clustered
            n_comm_split = int(math.ceil(len(idxs)/float(max_elt)) + 1)
            n_comm_split_total += n_comm_split
            print 'splitting into', n_comm_split, 'pieces'
            
            tdm_comm = tfidf_vectorizer.fit_transform([elt[1] for elt in utt_tuple_comm])
            
            membership_comm = KMeans(n_clusters=n_comm_split, random_state=0, n_init=50).fit_predict(tdm_comm)
            
            # make sure not to use already existing comm labels	
            to_add_mbshp = [elt + max_c_idx + 1 for elt in membership_comm]
            
            for pos, idx in enumerate(idxs):
                membership[idx] = to_add_mbshp[pos]
            
            max_c_idx += n_comm_split 
            
        c = dict(Counter(membership))
        
        big_comms_ids = [k for k,v in c.iteritems() if v>=max_elt]
        
        print len(big_comms_ids), 'big communities remaining'
    
    return c, membership, utt_tuples

def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'
    
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
        terms_temp = terms[(i-w+1):(i+1)]
        
        # edges to try
        candidate_edges = []
        for p in xrange(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
            
        for try_edge in candidate_edges:
        
            # if not self-edge
            if try_edge[1] != try_edge[0]:
                
                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1
                
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    from_to[try_edge] = 1
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertice weights
    g.es['weight'] = from_to.values() # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values()) # weighted degree
    
    return(g)

def core_dec(g, weighted = True):
    
    # work on clone of g to preserve g 
    gg = copy.deepcopy(g)    
    
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs["name"],[0]*len(gg.vs["name"])))
    
    if weighted == True:
	    # k-core decomposition for weighted graphs (generalized k-cores)
		# based on Batagelj and Zaversnik's (2002) algorithm #4
        
		# initialize min heap of degrees
        heap_g = zip(gg.vs["weight"],gg.vs["name"])
        heapq.heapify(heap_g)
        
        while len(heap_g)>0:
            			
            top = heap_g[0][1]
            # find vertice index of heap top element
            index_top = gg.vs["name"].index(top)
            # save names of its neighbors
            neighbors_top = gg.vs[gg.neighbors(top)]["name"]
            # exclude self-edges
            neighbors_top = [elt for elt in neighbors_top if elt!=top]
            # set core number of heap top element as its weighted degree
            cores_g[top] = gg.vs["weight"][index_top]
            # delete top vertice (weighted degrees are automatically updated)
            gg.delete_vertices(index_top)
            
            if len(neighbors_top)>0:
                # iterate over neighbors of top element
                for i, name_n in enumerate(neighbors_top):
                    index_n = gg.vs["name"].index(name_n)
                    max_n = max(cores_g[top],gg.strength(weights=gg.es["weight"])[index_n])
                    gg.vs[index_n]["weight"] = max_n
                    # update heap
                    heap_g = zip(gg.vs["weight"],gg.vs["name"])
                    heapq.heapify(heap_g)
            else:
                # update heap
                heap_g = zip(gg.vs["weight"],gg.vs["name"])
                heapq.heapify(heap_g)
            
    else:
        # k-core decomposition for unweighted graphs
        # based on Batagelj and Zaversnik's (2002) algorithm #1
        cores_g = dict(zip(gg.vs["name"],g.coreness()))
    
    # sort vertices by decreasing core number
    sorted_cores_g = sorted(cores_g.items(), key=operator.itemgetter(1), reverse=True)
    
    return(sorted_cores_g)
    
def sum_numbers_neighbors(g, names_numbers):
    # if used with core numbers, implements CoreRank (Tixier et al. EMNLP 2016)
    # initialize dictionary that will contain the scores
    name_scores = dict(zip(g.vs['name'],[0]*len(g.vs['name'])))  
    # iterate over the nodes
    for name_number in names_numbers:
        name = name_number[0]
        # find neighbor names
        neighbor_names = g.vs[g.neighbors(name)]['name']
        # sum up the scores of the neighbors
        neighbor_tuples = [elt for elt in names_numbers if elt[0] in neighbor_names]
        sum_numbers_neighbors = sum([elt[1] for elt in neighbor_tuples])
        # save result
        name_scores[name] = sum_numbers_neighbors
    
    # sort by decreasing score number
    sorted_name_scores = sorted(name_scores.items(), key=operator.itemgetter(1), reverse=True)
        
    return dict(sorted_name_scores)
	
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

# to be used within 'sorted'
def getKey(item):
    return item[1]
	
####################
### DATA LOADING ###
####################

path_root = 'd:\\3A\\Projet3A\\project\\abs_meet_summ'

# path_root = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ'

path_to_data = path_root + '\\data\\datasets\\meeting_summarization\\ami_icsi'

# read IDs of training set meetings from AMI corpus
with open(path_to_data + '\\lists\\list.ami.train', 'r+') as txtfile:
    ami_train_ids = txtfile.read().splitlines()

# read IDs of training set meetings from ICSI corpus
with open(path_to_data + '\\lists\\list.icsi.train', 'r+') as txtfile:
    icsi_train_ids = txtfile.read().splitlines()

# traditional stopwords
stpwds = nltk.corpus.stopwords.words("english")

# custom stopwords
with open(path_to_data + '\\communities\\stopwords\\stopwords.txt', 'r+') as txtfile:
    cus_stpwds = txtfile.read().splitlines()

# filler words
with open(path_to_data + '\\communities\\stopwords\\filler_words.csv', 'r+') as txtfile:
    filler = txtfile.read().splitlines()

# merge, removing duplicates
stopwords = list(set(cus_stpwds + filler + stpwds))

# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '')

# regex to match intra-word dashes only
my_regex = re.compile(r"(\b[-]\b)|[\W_]")

# ##########################
# ### COMMUNITY CREATION ###
# ##########################

ami_or_icsi = 'ami'
n_comms = 15
w = 5

if ami_or_icsi=='ami':
    my_ids = ami_train_ids
elif ami_or_icsi=='icsi':
    my_ids = icsi_train_ids

for kk in range(len(my_ids)):
    
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
    
    # retain utterances whose duration exceeds min_dur
    min_dur = 0.85
    asr_output_cleaned = asr_output[asr_output['duration']>min_dur]
    
    utterances = zip(asr_output_cleaned['index'].tolist(),asr_output_cleaned['utt'].tolist())
    
    tokens, utterances_processed = clean_utterances(utterances, punct, my_regex, stopwords)
        
    c, membership, utt_tuples = cluster_utterances(utterances_processed, 
                                                       remove_single=True, 
                                                       n_comms = n_comms,
                                                       max_elt = 10)

    # sanity check
    if not sum([elt[1] for elt in c.items()]) == len(utt_tuples):
        print '3rd sanity check failed!'
    
    g = terms_to_graph(tokens, w=w)
    
    # get weighted core numbers
    sorted_cores_g = core_dec(g, weighted=True)
    
    # get CoreRank scores
    core_rank_scores = sum_numbers_neighbors(g, sorted_cores_g)    
    #sorted(core_rank_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    # assign scores to each utterance based on the scores of the words they contain, divided by the number of words (measure of average informativeness)
    
    utt_scores = []
    for my_tuple in utt_tuples:
        utt = my_tuple[1]
        words = utt.split(' ')
        utt_scores.append(round(sum([core_rank_scores[word] for word in words])/float(len(words)),2))
    	
    # sort communities according to the average score of the utterances they contain
    
    comm_labels = list(set(membership))
    
    comm_scores = []
    for label in comm_labels:
        # get the index of all the utterances belonging to the comm
        utt_indexes = [idx for idx, value in enumerate(membership) if value==label]
        comm_scores.append(round(sum([utt_scores[idx] for idx in utt_indexes])/float(len(utt_indexes)),2))
        
    # get sorted index of elements of comm_scores
    std_idx = sorted(range(len(comm_scores)), key=lambda x: comm_scores[x], reverse=True)
    
    std_comm_labels = [comm_labels[idx] for idx in std_idx]
#    
#    for label in std_comm_labels[:n_comms]:
#        # utterances whose membership is equal to label
#        print [sent[1] for id,sent in enumerate(utt_tuples) if membership[id] == label]
    
    with open(path_to_data + '\\communities\\' + ami_or_icsi + '\\' + my_ids[kk] + '_comms.txt', 'w+') as txtfile:
        for label in std_comm_labels[:n_comms]:
            for my_label in [sent[0] for id,sent in enumerate(utt_tuples) if membership[id] == label]:
                to_write = [elt[1] for elt in utterances if elt[0]==my_label][0]
                to_write = clean_utterance_final(to_write,filler_words=filler)
                # one utterance per line
                txtfile.write(to_write + '\n')
            # separate communities by white line
            txtfile.write('\n')
    
    print kk+1, 'file(s) done'

##########################################
### EXAMPLE FOR A TRADITIONAL DOCUMENT ###
##########################################

import nltk.data
punct = string.punctuation.replace('-', '')
# regex to match intra-word dashes only
my_regex = re.compile(r"(\b[-]\b)|[\W_]")

path_root = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ'
path_to_data = path_root + '\\data\\datasets\\meeting_summarization\\ami_icsi'

# traditional stopwords
stpwds = nltk.corpus.stopwords.words("english")

# custom stopwords
with open(path_to_data + '\\communities\\stopwords\\stopwords.txt', 'r+') as txtfile:
    cus_stpwds = txtfile.read().splitlines()

# filler words
with open(path_to_data + '\\communities\\stopwords\\filler_words.csv', 'r+') as txtfile:
    filler = txtfile.read().splitlines()

# merge, removing duplicates
stopwords = list(set(cus_stpwds + filler + stpwds))


text = ''' Stack Overflow U.S. is a privately held website, the flagship USA site of the Stack Exchange Network, created in 2008 by Jeff Atwood and Joel Spolsky.
It is commonly topped with a selection U.S.A. of meats, vegetables and condiments. 
It features questions and answers on a wide range of topics in computer programming. 
Several similar dishes are prepared from ingredients commonly used in pizza preparation, such as calzone and stromboli. It is a popular fast food item.
The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. 
Pizza is a flatbread generally topped with tomato sauce and cheese and baked in an oven. 
Users of Stack Overflow can earn reputation points and badges; for example, a person is awarded reputation points for receiving an up vote on an answer given to a question, and can receive badges for their valued contributions, which represents a kind of gamification of the traditional Q&A site or forum. All user-generated content is licensed under a Creative Commons Attribute-ShareAlike license. 
Closing questions is a main differentiation from Yahoo! 
The modern pizza was invented in Naples, Italy, and the dish and its variants have since become popular in many areas of the world. 
In 2009, upon Italys request, Neapolitan pizza was safeguarded in the European Union as a Traditional Speciality Guaranteed dish. 
Answers and a way to prevent low quality questions. 
It promotes and protects the true Neapolitan pizza. 
The mechanism was overhauled in 2013; questions edited after being put on hold now appear in a review queue. Jeff Atwood stated in 2010 that duplicate questions are not seen as a problem but rather they constitute an advantage if such additional questions drive extra traffic to the site by multiplying relevant keyword hits in search engines. 
The Associazione Verace Pizza Napoletana (the True Neapolitan Pizza Association) is a non-profit organization founded in 1984 with headquarters in Naples. 
As of April 2014, Stack Overflow has over , , registered users and more than , , questions, with , , questions celebrated in late August 2015. 
Modern pizza evolved from similar flatbread dishes in Naples, Italy in the 18th or early 19th century. Until about 1830, pizza was sold from open-air stands and out of pizza bakeries, and pizzerias keep this old tradition alive today. Antica Pizzeria PortAlba in Naples is widely regarded as the first pizzeria. 
It was created to be a more open alternative to earlier Q&A sites such as Experts-Exchange. 
Pizza is sold fresh, frozen or in portions. Various types of ovens are used to cook them and many varieties exist. 
The name for the website was chosen by voting in April 2008 by readers of Coding Horror, Atwoods popular programming blog. 
A popular contemporary legend holds that the archetypal pizza, pizza Margherita, was invented in 1889, when the Royal Palace of Capodimonte commissioned the Neapolitan pizzaiolo (pizza maker) Raffaele Esposito to create a pizza in honor of the visiting Queen Margherita. Of the three different pizzas he created, the Queen strongly preferred a pie swathed in the colors of the Italian flag: red (tomato), green (basil), and white (mozzarella). 
Stack Exchange is a network of question and answer Web sites on topics in varied fields, each site covering a specific topic, where questions, answers, and users are subject to a reputation award process. 
Prior to that time, flatbread was often topped with ingredients such as garlic, salt, lard, cheese, and basil. It is uncertain when tomatoes were first added and there are many conflicting claims. 
The sites are modeled after Stack Overflow, a Q&A site for computer programming questions that was the original site in this network. The reputation system allows the sites to be self-moderating.
The term was first recorded in the 10th century, in a Latin manuscript from Gaeta in Central Italy. 
User contributions are licensed under Creative Commons Attribution-ShareAlike . Unported. 
Supposedly, this kind of pizza was then named after the Queen as Pizza Margherita, although recent research casts doubt on this legend. 
Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: Java, JavaScript, C#, PHP, Android, jQuery, Python and HTML. 
A popular variant of pizza in Italy is Sicilian pizza (locally called sfincione or sfinciuni), a thick-crust or deep-dish pizza originating during the 17th century in Sicily: it is essentially a focaccia that is typically topped with tomato sauce and other ingredients. 
Stack Exchange uses IIS, SQL Server, and the ASP.NET framework, all from a single code base for every Stack Exchange site (except Area , which runs off a fork of the Stack Overflow code base citation needed ). 
The code is primarily written in C# ASP.NET MVC using the Razor View Engine. The preferred IDE is Visual Studio and the data layers uses Dapper for data access. 
Until the 1860s, sfincione was the type of pizza usually consumed in Sicily, especially in the Western portion of the island. 
Blogs formerly used WordPress, but they were updated to run Jekyll. citation needed The team also notably uses Redis, HAProxy and Elasticsearch. 
Other variations of pizzas are also found in other regions of Italy, for example pizza al padellino or pizza al tegamino, a small-sized, thick-crust and deep-dish pizza typically served in Turin, Piedmont. 
Stack Exchange tries to stay up to date with the newest technologies from Microsoft, usually using the latest releases of any given framework.
'''

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(text.replace('\n', ' '))
utterances = zip(range(len(sentences)), sentences)

<<<<<<< HEAD
tokens, utterances_processed = clean_utterances(utterances_processed, punct, my_regex, stopwords)
	
=======
tokens, utterances_processed = clean_utterances(utterances, punct, my_regex, stopwords)

>>>>>>> cdb899820613e9e14401e2286c8ff792ce854a12
c, membership, utt_tuples = cluster_utterances(utterances_processed, 
												   remove_single=True, 
												   n_comms = 15,
												   max_elt = 10)

# sanity check
if not sum([elt[1] for elt in c.items()]) == len(utt_tuples):
	print '3rd sanity check failed!'

g = terms_to_graph(tokens, w=5)

# get weighted core numbers
sorted_cores_g = core_dec(g, weighted=True)

# get CoreRank scores
core_rank_scores = sum_numbers_neighbors(g, sorted_cores_g)    
#sorted(core_rank_scores.items(), key=operator.itemgetter(1), reverse=True)

# assign scores to each utterance based on the scores of the words they contain, divided by the number of words (measure of average informativeness)

utt_scores = []
for my_tuple in utt_tuples:
	utt = my_tuple[1]
	words = utt.split(' ')
	utt_scores.append(round(sum([core_rank_scores[word] for word in words])/float(len(words)),2))
	
# sort communities according to the average score of the utterances they contain

comm_labels = list(set(membership))

comm_scores = []
for label in comm_labels:
	# get the index of all the utterances belonging to the comm
	utt_indexes = [idx for idx, value in enumerate(membership) if value==label]
	comm_scores.append(round(sum([utt_scores[idx] for idx in utt_indexes])/float(len(utt_indexes)),2))
	
# get sorted index of elements of comm_scores
std_idx = sorted(range(len(comm_scores)), key=lambda x: comm_scores[x], reverse=True)

std_comm_labels = [comm_labels[idx] for idx in std_idx]
#    
#    for label in std_comm_labels[:n_comms]:
#        # utterances whose membership is equal to label
#        print [sent[1] for id,sent in enumerate(utt_tuples) if membership[id] == label]

with open(path_to_data + '//test_document.txt', 'w+') as txtfile:
<<<<<<< HEAD
	for label in std_comm_labels[:n_comms]:
		for label in [sent[0] for id,sent in enumerate(utt_tuples) if membership[id] == label]:
			to_write = [elt[1] for elt in utterances_processed if elt[0]==label][0]
			to_write = clean_utterance_final(to_write,filler_words=filler)
=======
	for label in std_comm_labels[:15]:
		for my_label in [sent[0] for id,sent in enumerate(utt_tuples) if membership[id] == label]:
			to_write = [elt[1] for elt in utterances if elt[0]==my_label][0]
			#to_write = clean_utterance_final(to_write,filler_words=filler)
>>>>>>> cdb899820613e9e14401e2286c8ff792ce854a12
			# one utterance per line
			txtfile.write(to_write + '\n')
		# separate communities by white line
		txtfile.write('\n')
