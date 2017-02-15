#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
:Name:
    takahe

:Authors:
    Florian Boudin (florian.boudin@univ-nantes.fr)

:Version:
    0.4

:Date:
    Mar. 2013

:Description:
    takahe is a multi-sentence compression module. Given a set of redundant 
    sentences, a word-graph is constructed by iteratively adding sentences to 
    it. The best compression is obtained by finding the shortest path in the
    word graph. The original algorithm was published and described in
    [filippova:2010:COLING]_. A keyphrase-based reranking method, described in
    [boudin-morin:2013:NAACL]_ can be applied to generate more informative 
    compressions.

    .. [filippova:2010:COLING] Katja Filippova, Multi-Sentence Compression: 
       Finding Shortest Paths in Word Graphs, *Proceedings of the 23rd 
       International Conference on Computational Linguistics (Coling 2010)*, 
       pages 322-330, 2010.
    .. [boudin-morin:2013:NAACL] Florian Boudin and Emmanuel Morin, Keyphrase 
       Extraction for N-best Reranking in Multi-Sentence Compression, 
       *Proceedings of the 2013 Conference of the North American Chapter of the
       Association for Computational Linguistics: Human Language Technologies 
       (NAACL-HLT 2013)*, 2013.


:History:
    Development history of the takahe module:
        - 0.4 (Mar. 2013) adding the keyphrase-based nbest reranking algorithm
        - 0.33 (Feb. 2013), bug fixes and better code documentation
        - 0.32 (Jun. 2012), Punctuation marks are now considered within the 
          graph, compressions are then punctuated
        - 0.31 (Nov. 2011), modified context function (uses the left and right 
          contexts), improved docstring documentation, bug fixes
        - 0.3 (Oct. 2011), improved K-shortest paths algorithm including 
          verb/size constraints and ordered lists for performance
        - 0.2 (Dec. 2010), removed dependencies from nltk (i.e. POS-tagging, 
          tokenization and stopwords removal)
        - 0.1 (Nov. 2010), first version

:Dependencies:
    The following Python modules are required:
        - `networkx <http://networkx.github.com/>`_ for the graph construction
          (v1.2+)

:Usage:
    A typical usage of this module is::
    
        import takahe
        
        # A list of tokenized and POS-tagged sentences
        sentences = ['Hillary/NNP Clinton/NNP wanted/VBD to/stop visit/VB ...']
        
        # Create a word graph from the set of sentences with parameters :
        # - minimal number of words in the compression : 6
        # - language of the input sentences : en (english)
        # - POS tag for punctuation marks : PUNCT
        compresser = takahe.word_graph( sentences, 
                                        nb_words = 6, 
                                        lang = 'en', 
                                        punct_tag = "PUNCT" )

        # Get the 50 best paths
        candidates = compresser.get_compression(50)

        # 1. Rerank compressions by path length (Filippova's method)
        for cummulative_score, path in candidates:

            # Normalize path score by path length
            normalized_score = cummulative_score / len(path)

            # Print normalized score and compression
            print round(normalized_score, 3), ' '.join([u[0] for u in path])

        # Write the word graph in the dot format
        compresser.write_dot('test.dot')

        # 2. Rerank compressions by keyphrases (Boudin and Morin's method)
        reranker = takahe.keyphrase_reranker( sentences,  
                                              candidates, 
                                              lang = 'en' )

        reranked_candidates = reranker.rerank_nbest_compressions()

        # Loop over the best reranked candidates
        for score, path in reranked_candidates:
            
            # Print the best reranked candidates
            print round(score, 3), ' '.join([u[0] for u in path])

:Misc:
    The Takahe is a flightless bird indigenous to New Zealand. It was thought to
    be extinct after the last four known specimens were taken in 1898. However, 
    after a carefully planned search effort the bird was rediscovered by on 
    November 20, 1948. (Wikipedia, http://en.wikipedia.org/wiki/takahe)  
"""

import math
import codecs
import os
import re
import sys
import bisect
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import core_rank as cr
from nltk.corpus import wordnet as wn 
import pynlpl.lm.lm
import numpy as np
#import matplotlib.pyplot as plt

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# [ Class word_graph
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class word_graph:
    """
    The word_graph class constructs a word graph from the set of sentences given
    as input. The set of sentences is a list of strings, sentences are tokenized
    and words are POS-tagged (e.g. ``"Saturn/NNP is/VBZ the/DT sixth/JJ 
    planet/NN from/IN the/DT Sun/NNP in/IN the/DT Solar/NNP System/NNP"``). 
    Four optional parameters can be specified:

    - nb_words is is the minimal number of words for the best compression 
      (default value is 8).
    - lang is the language parameter and is used for selecting the correct 
      stopwords list (default is "en" for english, stopword lists are localized 
      in /resources/ directory).
    - punct_tag is the punctuation mark tag used during graph construction 
      (default is PUNCT).
    """

    #-T-----------------------------------------------------------------------T-
    def __init__(self, sentence_list, nb_words=8, lang="en", punct_tag="PUNCT", pos_separator='/'):

        self.sentence = list(sentence_list)
        """ A list of sentences provided by the user. """

        self.length = len(sentence_list)
        """ The number of sentences given for fusion. """
        
        self.nb_words = nb_words
        """ The minimal number of words in the compression. """

        self.resources = os.path.dirname(__file__) + '\\resources\\'
        """ The path of the resources folder. """

        #self.stopword_path = self.resources+'stopwords.'+lang+'.dat'
        self.stopword_path = 'C:\\Users\\mvazirg\\Documents\\abs_meet_summ\\code\\takahe\\resources\\stopwords.en.dat'
        """ The path of the stopword list, e.g. stopwords.[lang].dat. """

        self.stopwords = self.load_stopwords(self.stopword_path)
        """ The set of stopwords loaded from stopwords.[lang].dat. """

        self.punct_tag = punct_tag
        """ The stopword tag used in the graph. """

        self.pos_separator = pos_separator
        """ The character (or string) used to separate a word and its Part of Speech tag """

        self.graph = nx.DiGraph()
        """ The directed graph used for fusion. """
    
        self.start = '-start-'
        """ The start token in the graph. """

        self.stop = '-end-'
        """ The end token in the graph. """

        self.sep = '/-/'
        """ The separator used between a word and its POS in the graph. """
        
        self.term_freq = {}
        """ The frequency of a given term. """
        
        self.verbs = set(['VB', 'VBD', 'VBP', 'VBZ', 'VH', 'VHD', 'VHP', 'VBZ', 
        'VV', 'VVD', 'VVP', 'VVZ'])
        """
        The list of verb POS tags required in the compression. At least *one* 
        verb must occur in the candidate compressions.
        """

        # Replacing default values for French
        if lang == "fr":
            self.verbs = set(['V', 'VPP', 'VINF'])

        #**************************************************************************
        # initialize a graph for core rank scores
        #**************************************************************************
        self.core_rank_scores = self.core_rank()
        #**************************************************************************
        # END       initialize a graph for core rank scores
        #**************************************************************************

        #**************************************************************************
        # initialize lan model
        #**************************************************************************
        print 'loading language model'
        self.my_lm = pynlpl.lm.lm.ARPALanguageModel(filename='C:\\Users\\mvazirg\\Documents\\en-70k-0.2.lm',mode='simple')

        #**************************************************************************
        # initialize mapping to build edges
        #**************************************************************************
        self.mapping = []


        # 1. Pre-process the sentences
        self.pre_process_sentences()

        # 2. Compute term statistics
        self.compute_statistics()

        # 3. Build the word graph
        self.build_graph()

        # 4. Path selection
        # compression = self.get_compression()
        # final_score = self.final_score(compression)
        # print(final_score)
    #-B-----------------------------------------------------------------------B-

    #**************************************************************************
    # initialize a graph for core rank scores
    #**************************************************************************
    def concat (self, sentences):
        sentences = ' '.join(sentences)
        words = sentences.split(' ')
        words = [word.split("/")[0] for word in words]
        sentences = ' '.join(words)
        return sentences

    def core_rank(self):
        text = self.concat(self.sentence)
        all_terms = cr.clean_text_simple(text,pos_filtering=False, stemming=False)
        # get graph of terms    
        g = cr.terms_to_graph(all_terms, w=10)
        # get weighted core numbers
        sorted_cores_g = cr.core_dec(g, weighted=True)
        # get CoreRank scores
        core_rank_scores = dict(cr.sum_numbers_neighbors(g, sorted_cores_g))
        return core_rank_scores
    #**************************************************************************
    # END       initialize a graph for core rank scores
    #**************************************************************************

    #-T-----------------------------------------------------------------------T-
    def pre_process_sentences(self):
        """
        Pre-process the list of sentences given as input. Split sentences using 
        whitespaces and convert each sentence to a list of (word, POS) tuples.
        """

        for i in range(self.length):
        
            # Normalise extra white spaces
            self.sentence[i] = re.sub(' +', ' ', self.sentence[i])
            self.sentence[i] = self.sentence[i].strip()
            
            # Tokenize the current sentence in word/POS
            sentence = self.sentence[i].split(' ')

            # Creating an empty container for the cleaned up sentence
            container = [(self.start, self.start)]

            # Looping over the words
            for w in sentence:
                
                # Splitting word, POS
                pos_separator_re = re.escape(self.pos_separator)
                m = re.match("^(.+)" +pos_separator_re +"(.+)$", w)
                
                # Extract the word information
                token, POS = m.group(1), m.group(2)

                # Add the token/POS to the sentence container
                container.append((token.lower(), POS))
                    
            # Add the stop token at the end of the container
            container.append((self.stop, self.stop))

            # Recopy the container into the current sentence
            self.sentence[i] = container
    #-B-----------------------------------------------------------------------B-
    
    
    #-T-----------------------------------------------------------------------T-
    def build_graph(self):
        """
        Constructs a directed word graph from the list of input sentences. Each
        sentence is iteratively added to the directed graph according to the 
        following algorithm:

        - Word mapping/creation is done in four steps:

            1. non-stopwords (same, syn, hyp, entail)

            2. stopwords

            3. punctuation marks

        For the last three groups of words where mapping is ambiguous we check 
        the immediate context (the preceding and following words in the sentence 
        and the neighboring nodes in the graph) and select the candidate which 
        has larger overlap in the context, or the one with a greater frequency 
        (i.e. the one which has more words mapped onto it). Stopwords are mapped 
        only if there is some overlap in non-stopwords neighbors, otherwise a 
        new node is created. Punctuation marks are mapped only if the preceding 
        and following words in the sentence and the neighboring nodes are the
        same.

        - Edges are then computed and added between mapped words.
        
        Each node in the graph is represented as a tuple ('word/POS', id) and 
        possesses an info list containing (sentence_id, position_in_sentence)
        tuples.
        """     

        # Iteratively add each sentence in the graph --------------------------
        for i in range(self.length):

            # Compute the sentence length
            sentence_len = len(self.sentence[i])

            # Create the mapping container
            self.mapping.append([0] * sentence_len) 

            #-------------------------------------------------------------------
            # 1. non-stopwords 
            #    same_nodes, synonyme_nodes, hypernyme_nodes, 
            #    common_hypernym_nodes, entail_nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Treat '-start-/-/-start-' and '-end-/-/-end-'
                if (i==0) and (j==0):
                    self.graph.add_node( ('-start-/-/-start-', 0), info=[(i, j)],
                                         label='-start-' )
                    self.mapping[i][j] = ('-start-/-/-start-',0)
                    continue
                elif j==0:
                    self.graph.node[('-start-/-/-start-',0)]['info'].append((i,j))
                    self.mapping[i][j] = ('-start-/-/-start-',0)
                    continue
                elif (i==0) and (j==(sentence_len-1)):
                    self.graph.add_node( ('-end-/-/-end-', 0), info=[(i, j)],
                                         label='-end-' )
                    self.mapping[i][j] = ('-end-/-/-end-',0)
                    continue
                elif j==(sentence_len-1):
                    self.graph.node[('-end-/-/-end-',0)]['info'].append((i,j))
                    self.mapping[i][j] = ('-end-/-/-end-',0)
                    continue

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue
            
                # Create the node identifier
                node = token.lower() + self.sep + POS

                #-------------------------------------------------------------------
                # Find the number of ambiguous nodes(same nodes) in the graph
                #-------------------------------------------------------------------
                k = self.ambiguous_nodes(node)
                same_candidates = self.same_nodes(node)
                syn_candidates = self.synonym_nodes(node)
                hyp_candidates = self.hypernym_nodes(node)
                common_hyp_candidates = self.common_hypernym_nodes(node)
                entail_candidates = self.entail_nodes(node)

                #-------------------------------------------------------------------
                # Filter candidates, remove node of same sentence
                #-------------------------------------------------------------------
                same_candidates = self.filter_cand(same_candidates, i)
                syn_candidates = self.filter_cand(syn_candidates, i)
                hyp_candidates = self.filter_cand(hyp_candidates, i)
                common_hyp_candidates = self.filter_cand_common_hyp(common_hyp_candidates, i)
                entail_candidates = self.filter_cand(entail_candidates, i)
                #-------------------------------------------------------------------
                # If there is no ambiguous node in the graph, check for synonyme candidates
                # else, choose the node with max core-rank score
                #-------------------------------------------------------------------
                if(same_candidates != []):
                    node_to_append = self.best_candidate_context(same_candidates,i,j)
                    self.graph.node[node_to_append]['info'].append((i,j))
                    self.mapping[i][j] = node_to_append

                elif (syn_candidates != []):
                    node_to_replace, max_score = self.best_candidate_coreRank(syn_candidates, token)
                    if max_score < self.core_rank_scores[token]:
                        # Update the node in the graph
                        self.update_nodes(node_to_replace, node, i, j)
                    else:
                        # Append to the node in the graph
                        self.graph.node[node_to_replace]['info'].append((i,j))
                        # Mark the word to node-to-replace
                        self.mapping[i][j] = node_to_replace

                elif (hyp_candidates != []):
                    node_to_replace, max_score = self.best_candidate_coreRank(hyp_candidates, token)
                    if max_score < self.core_rank_scores[token]:
                        # Update the node in the graph
                        self.update_nodes(node_to_replace, i, j)
                    else:
                        self.graph.node[node_to_replace]['info'].append((i,j))
                        # Mark the word to node-to-replace
                        self.mapping[i][j] = node_to_replace


                elif (common_hyp_candidates != []):
                    # Use path_similarity to Find the nearest common hypernyme
                    node_to_replace, common_hyp, max_score = \
                        self.best_candidate_similarity(common_hyp_candidates, node)
                    # Update CoreRank scores
                    self.core_rank_scores.update({common_hyp.lemmas()[0].name() : max_score})
                    # Update the node in the graph
                    self.update_nodes_common_hyp(node_to_replace, common_hyp, i, j)


                elif (entail_candidates != []):
                    node_to_replace, max_score = self.best_candidate_coreRank(syn_candidates, token)
                    if max_score < self.core_rank_scores[token]:
                        # Update the node in the graph
                        self.update_nodes(node_to_replace, i, j)
                    else:
                        self.graph.node[node_to_replace]['info'].append((i,j))
                        # Mark the word to node-to-replace
                        self.mapping[i][j] = node_to_replace

                else:
                    self.graph.add_node((node, k),
                                        info=[(i,j)],
                                        label=token.lower())
                    self.mapping[i][j] = (node, k)
                #-------------------------------------------------------------------
           

            #-------------------------------------------------------------------
            # 2. map the stopwords to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* stopword, continues
                if not token in self.stopwords :
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token.lower() )

                    # Mark the word as mapped to k
                    self.mapping[i][j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes, the
                        # boolean indicates to consider only non stopwords
                        l_context = self.get_directed_context((node, l), 'left',\
                                    True)
                        r_context = self.get_directed_context((node, l), 'right',\
                                    True)
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 0:
                    # if i not in ids and \
                    # (ambinode_overlap[selected] > 1 and POS==self.punct_tag) or\
                    # (ambinode_overlap[selected] > 0 and POS!=self.punct_tag) :

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        self.mapping[i][j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k) , info=[(i, j)],
                                             label=token.lower() )

                        # Mark the word as mapped to k
                        self.mapping[i][j] = (node, k)

            #-------------------------------------------------------------------
            # 3. At last map the punctuation marks to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* punctuation mark, continues
                if not re.search('(?u)^\W$', token):
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token.lower() )

                    # Mark the word as mapped to k
                    self.mapping[i][j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context((node, l), 'left')
                        r_context = self.get_directed_context((node, l), 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 1:

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        self.mapping[i][j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k), info=[(i, j)], 
                                             label=token.lower() )

                        # Mark the word as mapped to k
                        self.mapping[i][j] = (node, k)

        #-------------------------------------------------------------------
        # 4. Connects the mapped words with directed edges
        #    We need to finish all sentences, so taht mapping is the latest one
        #-------------------------------------------------------------------
        for i in range(self.length):
            for j in range(1, len(self.mapping[i])):
                self.graph.add_edge(self.mapping[i][j-1], self.mapping[i][j])


        # Assigns a weight to each node in the graph ---------------------------
        for node1, node2 in self.graph.edges_iter():
            edge_weight = self.get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)
    #-B-----------------------------------------------------------------------B-

 
    #-T-----------------------------------------------------------------------T-
    def ambiguous_nodes(self, node):
        """
        Takes a node in parameter and returns the number of possible candidate 
        (ambiguous) nodes in the graph.
        """
        k = 0
        while(self.graph.has_node((node, k))):
            k += 1
        return k
    #-B-----------------------------------------------------------------------B-


    #**************************************************************************
    # filter candidates list
    #**************************************************************************
    def filter_cand(self, candidates, i):
        res = []
        if candidates == []:
            return res
        # filter out candidate node in the same sentence
        for cand in candidates:
            for sid, pos_s in self.graph.node[cand]['info']:
                if sid!=i:
                    res.append(cand)
        return res

    #**************************************************************************
    # filter common hyps' candidates list
    #**************************************************************************
    def filter_cand_common_hyp(self, candidates, i):
        # print('candidates:' , candidates)
        res = []
        if candidates == ([], ''):
            return res
        # filter out candidate node in the same sentence
        for gnode, hyp in candidates:
            for sid, pos_s in self.graph.node[gnode]['info']:
                if sid!=i:
                    res.append((gnode, hyp))
        return res

    #**************************************************************************
    # best candidate according to core-rank-scores
    #**************************************************************************
    def best_candidate_coreRank(self, candidates, word):
        node_to_replace = None
        max_score = 0
        for tmp_node in candidates:
            tmp_word, tmp_pos = tmp_node[0].split(self.sep)
            tmp_score = self.core_rank_scores[tmp_word]
            if tmp_score >= max_score:
                node_to_replace = tmp_node
                max_score = tmp_score  
        return node_to_replace, max_score

    #**************************************************************************
    # best candidate according to context
    #**************************************************************************
    def best_candidate_context(self, candidate_nodes, i, j):
        # Create the neighboring nodes identifiers
        prev_token, prev_POS = self.sentence[i][j-1]
        next_token, next_POS = self.sentence[i][j+1]
        prev_node = prev_token.lower() + self.sep + prev_POS
        next_node = next_token.lower() + self.sep + next_POS
        
        # Find the number of ambiguous nodes in the graph
        k = len(candidate_nodes)
        
        # Search for the ambiguous node with the larger overlap in
        # context or the greater frequency.
        ambinode_overlap = []
        ambinode_frequency = []

        # For each ambiguous node
        for l in range(k):

            # Get the immediate context words of the nodes
            l_context = self.get_directed_context(candidate_nodes[l],'left')
            r_context = self.get_directed_context(candidate_nodes[l],'right')
            
            # Compute the (directed) context sum
            val = l_context.count(prev_node) 
            val += r_context.count(next_node)

            # Add the count of the overlapping words
            ambinode_overlap.append(val)

            # Add the frequency of the ambiguous node
            ambinode_frequency.append(
                len( self.graph.node[candidate_nodes[l]]['info'] )
            )
        # Select the ambiguous node
        selected = self.max_index(ambinode_overlap)
        if ambinode_overlap[selected] == 0:
            selected = self.max_index(ambinode_frequency)
        
        return candidate_nodes[selected]

    #**************************************************************************
    # best candidate for common hyps according to path_similarity
    # word-to-add's tagging should be transformed to wordnet's tagging
    #**************************************************************************
    def best_candidate_similarity(self, candidates, node_to_add):
        # change word_to_add tagging
        word, tag = node_to_add.split(self.sep)
        pos = self.tagging(tag)
        syn_to_add = wn.synsets(word, pos)[0]

        node_to_replace = None
        word_to_replace = None
        common_hyp = None
        max_score = 0
        for tmp_node, tmp_hyp in candidates:
            tmp_word, tmp_tag = tmp_node[0].split(self.sep)
            tmp_pos = self.tagging(tmp_tag)        
            tmp_syn = wn.synsets(tmp_word, pos=tmp_pos)[0]

            # score of common-hyp = similarity(common-hyp, node-to-add)
            #                     * similarity(common-hyp, node-to-replace) 
            tmp_score = wn.path_similarity(tmp_hyp, syn_to_add) * \
                        wn.path_similarity(tmp_hyp, tmp_syn)
            if tmp_score >= max_score:
                node_to_replace = tmp_node
                word_to_replace = tmp_word
                common_hyp = tmp_hyp
                max_score = tmp_score 

        return node_to_replace, common_hyp, max(self.core_rank_scores[word], 
                                                self.core_rank_scores[word_to_replace])


    #**************************************************************************
    # update best-candidate-node with new node
    #**************************************************************************
    def update_nodes(self, node_to_replace, node, i, j):
        token, pos = node.split(self.sep)
        new_id = self.ambiguous_nodes(node)
        new_node = (node, new_id)

        old_info = self.graph.node[node_to_replace]['info']

        self.graph.add_node(new_node,
                            info=[],
                            label=token.lower())
        self.graph.node[new_node]['info'] += old_info
        self.graph.node[new_node]['info'].append((i,j))

        self.update_mapping(new_node, node_to_replace, i, j)

        self.graph.remove_node(node_to_replace)
        return

    def update_mapping(self, new_node, node_to_replace, i, j):
        # change old mapping
        for k in range(len(self.mapping)):
            for l in range(len(self.mapping[k])):
                if self.mapping[k][l] == node_to_replace:
                    self.mapping[k][l] = new_node

        # add new mapping
        self.mapping[i][j] = new_node
        return

    #**************************************************************************
    # update best-candidate-node with new node of common-hyp
    #**************************************************************************
    def update_nodes_common_hyp(self, node_to_replace, common_hyp,  i, j):
        
        word_common = common_hyp.lemmas()[0].name()
        pos_common = node_to_replace[0].split(self.sep)[1]
        
        new_id = self.ambiguous_nodes(word_common + self.sep + pos_common)
        node_common = (word_common + self.sep + pos_common, new_id)

        old_info = self.graph.node[node_to_replace]['info']

        self.graph.add_node(node_common,
                            info=[],
                            label=word_common.lower())

        self.graph.node[node_common]['info'] += old_info
        self.graph.node[node_common]['info'].append((i,j))

        self.update_mapping(node_common, node_to_replace, i, j)

        self.graph.remove_node(node_to_replace)

        # print 'node_to_replace: ', self.graph.node[node_to_replace]
        return


    #**************************************************************************
    # replace tag with wordnet's tag
    #**************************************************************************  
    def tagging (self, tag):
        if tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('R'):
            return wn.ADV
        else :
            return ''
    #**************************************************************************
    # check for same, synonyme, hypernyme, entailment nodes
    #**************************************************************************
    def same_nodes(self, node):
        same_nodes = []
        word,tag = node.split(self.sep)
        pos = self.tagging(tag)
        for gnode in self.graph.nodes():
            ref_word, ref_tag = gnode[0].split(self.sep)
            ref_pos = self.tagging(ref_tag)
            if (ref_pos == pos) and (ref_word == word):
                same_nodes.append(gnode)
        return same_nodes
        
    def synonym_nodes(self, node):
        syn_nodes = []
        syns = []
        word,tag = node.split(self.sep)
        pos = self.tagging(tag)
        if pos == '':
            return syn_nodes
        for syn in wn.synsets(word,pos=pos):
            syn_candidate = syn.name().split('.')[0]
            if syn_candidate not in syns:
                syns.append(syn_candidate)
        for gnode in self.graph.nodes():
            ref_word, ref_tag = gnode[0].split(self.sep)
            ref_pos = self.tagging(ref_tag)
            if (ref_pos == pos) and (ref_word != word):
                if ref_word in syns:
                    syn_nodes.append(gnode)
        return syn_nodes

    # Case 1 for hypernyme: word to add is hypernyme of existing node 
    def hypernym_nodes(self,node):
        hyp_nodes = []
        hyps = []
        word,tag = node.split(self.sep)
        pos = self.tagging(tag)
        if pos == '':
            return hyp_nodes
        for syn in wn.synsets(word, pos=pos):
            for hyp in syn.hypernyms():
                hyp_candidate = hyp.name().split('.')[0]
                if hyp_candidate not in hyps:
                    hyps.append(hyp_candidate)
        for gnode in self.graph.nodes():
            ref_word, ref_tag = gnode[0].split(self.sep)
            ref_pos = self.tagging(ref_tag)
            if (ref_word in hyps) and (ref_pos == pos) and (ref_word != word):
                hyp_nodes.append(gnode)
        return hyp_nodes


    # Case 2 for hypernyme: word to add has common hypernyme with existing node
    def common_hypernym_nodes(self, node):
        hyps_nodes = []  # return [[node1, common_hyp1],[node2, common_hyp2],...]
        hyps_word = []   # hypernymes of this word
        word, tag = node.split(self.sep)
        pos = self.tagging(tag) 
        if pos == '':
            return hyps_nodes, ''
        # All hyps of node-to-add
        for syn in wn.synsets(word, pos=pos):
            for hyp in syn.hypernyms():
                hyps_word.append(hyp)
        # For each existing node, compare its hyps with hyps
        for gnode in self.graph.nodes():
            ref_word, ref_tag = gnode[0].split(self.sep)
            ref_pos = self.tagging(ref_tag)
            for gnode_syn in wn.synsets(ref_word, pos=ref_pos):
                for gnode_hyp in gnode_syn.hypernyms():
                    if (gnode_hyp in hyps_word) and ([gnode, hyp] not in hyps_nodes):
                        hyps_nodes.append((gnode, hyp))
        return hyps_nodes 

    def entail_nodes(self,node):
        ent_nodes = []
        ents = []
        word,tag = node.split(self.sep)
        pos = self.tagging(tag)
        if pos == '':
            return ent_nodes
        for syn in wn.synsets(word,pos=pos):
            for ent in syn.entailments():
                ent_candidate = ent.name().split('.')[0]
                if ent_candidate not in ents:
                    ents.append(ent_candidate)
        for gnode in self.graph.nodes():
            ref_word, ref_tag = gnode[0].split(self.sep)
            ref_pos = self.tagging(ref_tag)
            if (ref_pos == pos) & (ref_word != word):
                if ref_word in ents:
                    ent_nodes.append(gnode)
        return ent_nodes    

    #**************************************************************************
    # END  check for synonyme, hypernyme, entailment nodes
    #************************************************************************** 


    #**************************************************************************
    # Path selection with core-rank-score
    #**************************************************************************
    #-T-----------------------------------------------------------------------T-
    def get_directed_context(self, node, dir='all', non_pos=False):
        """
        Returns the directed context of a given node, i.e. a list of word/POS of
        the left or right neighboring nodes in the graph. The function takes 
        four parameters :

        - node is the word/POS tuple
        - k is the node identifier used when multiple nodes refer to the same 
          word/POS (e.g. k=0 for (the/DET, 0), k=1 for (the/DET, 1), etc.)
        - dir is the parameter that controls the directed context calculation, 
          it can be set to left, right or all (default)
        - non_pos is a boolean allowing to remove stopwords from the context 
          (default is false)
        """

        # Define the context containers
        l_context = []
        r_context = []

        # For all the sentence/position tuples
        for sid, off in self.graph.node[node]['info']:
            
            prev = self.sentence[sid][off-1][0].lower() + self.sep +\
                   self.sentence[sid][off-1][1]
                   
            next = self.sentence[sid][off+1][0].lower() + self.sep +\
                   self.sentence[sid][off+1][1]
                   
            if non_pos:
                if self.sentence[sid][off-1][0] not in self.stopwords:
                    l_context.append(prev)
                if self.sentence[sid][off+1][0] not in self.stopwords:
                    r_context.append(next)
            else:
                l_context.append(prev)
                r_context.append(next)

        # Returns the left (previous) context
        if dir == 'left':
            return l_context
        # Returns the right (next) context
        elif dir == 'right':
            return r_context
        # Returns the whole context
        else:
            l_context.extend(r_context)
            return l_context
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_edge_weight(self, node1, node2):
        """
        Compute the weight of an edge *e* between nodes *node1* and *node2*. It 
        is computed as e_ij = (A / B) / C with:
        
        - A = freq(i) + freq(j), 
        - B = Sum (s in S) 1 / diff(s, i, j)
        - C = freq(i) * freq(j)
        
        A node is a tuple of ('word/POS', unique_id).
        """

        # Get the list of (sentence_id, pos_in_sentence) for node1
        info1 = self.graph.node[node1]['info']
        
        # Get the list of (sentence_id, pos_in_sentence) for node2
        info2 = self.graph.node[node2]['info']
        
        # Get the frequency of node1 in the graph
        # freq1 = self.graph.degree(node1)
        freq1 = len(info1)
        
        # Get the frequency of node2 in cluster
        # freq2 = self.graph.degree(node2)
        freq2 = len(info2)

        # Initializing the diff function list container
        diff = []

        # For each sentence of the cluster (for s in S)
        for s in range(self.length):
        
            # Compute diff(s, i, j) which is calculated as
            # pos(s, i) - pos(s, j) if pos(s, i) < pos(s, j)
            # O otherwise
    
            # Get the positions of i and j in s, named pos(s, i) and pos(s, j)
            # As a word can appear at multiple positions in a sentence, a list
            # of positions is used
            pos_i_in_s = []
            pos_j_in_s = []
            
            # For each (sentence_id, pos_in_sentence) of node1
            for sentence_id, pos_in_sentence in info1:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_i_in_s.append(pos_in_sentence)
            
            # For each (sentence_id, pos_in_sentence) of node2
            for sentence_id, pos_in_sentence in info2:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_j_in_s.append(pos_in_sentence)
                    
            # Container for all the diff(s, i, j) for i and j
            all_diff_pos_i_j = []
            
            # Loop over all the i, j couples
            for x in range(len(pos_i_in_s)):
                for y in range(len(pos_j_in_s)):
                    diff_i_j = pos_i_in_s[x] - pos_j_in_s[y]
                    # Test if word i appears *BEFORE* word j in s
                    if diff_i_j < 0:
                        all_diff_pos_i_j.append(-1.0*diff_i_j)
                        
            # Add the mininum distance to diff (i.e. in case of multiple 
            # occurrencies of i or/and j in sentence s), 0 otherwise.
            if len(all_diff_pos_i_j) > 0:
                diff.append(1.0/min(all_diff_pos_i_j))
            else:
                diff.append(0.0)
                
        weight1 = freq1
        weight2 = freq2

        return ( (freq1 + freq2) / sum(diff) ) / (weight1 * weight2)
    #-B-----------------------------------------------------------------------B-
   
   
    #-T-----------------------------------------------------------------------T-
    def k_shortest_paths(self, start, end, k=10):
        """
        Simple implementation of a k-shortest paths algorithms. Takes three
        parameters: the starting node, the ending node and the number of 
        shortest paths desired. Returns a list of k tuples (path, weight).
        """

        # Initialize the list of shortest paths
        kshortestpaths = []

        # Initializing the label container 
        orderedX = []
        orderedX.append((0, start, 0))
        
        # Initializing the path container
        paths = {}
        paths[(0, start, 0)] = [start]
        
        # Initialize the visited container
        visited = {}
        visited[start] = 0

        # Initialize the sentence container that will be used to remove 
        # duplicate sentences passing throught different nodes
        sentence_container = {}
    
        # While the number of shortest paths isn't reached or all paths explored
        while len(kshortestpaths) < k and len(orderedX) > 0:
        
            # Searching for the shortest distance in orderedX
            shortest = orderedX.pop(0)
            shortestpath = paths[shortest]
            
            # Removing the shortest node from X and paths
            del paths[shortest]
    
            # Iterating over the accessible nodes
            for node in self.graph.neighbors(shortest[1]):
            
                # To avoid loops
                if node in shortestpath:
                    continue
            
                # Compute the weight to node
                w = shortest[0] + self.graph[shortest[1]][node]['weight']
            
                # If found the end, adds to k-shortest paths 
                if node == end:

                    #-T-------------------------------------------------------T-
                    # --- Constraints on the shortest paths

                    # 1. Check if path contains at least one werb
                    # 2. Check the length of the shortest path, without 
                    #    considering punctuation marks and starting node (-1 in
                    #    the range loop, because nodes are reversed)
                    # 3. Check the paired parentheses and quotation marks
                    # 4. Check if sentence is not redundant

                    nb_verbs = 0
                    length = 0
                    paired_parentheses = 0
                    quotation_mark_number = 0
                    raw_sentence = ''

                    for i in range(len(shortestpath) - 1):
                        word, tag = shortestpath[i][0].split(self.sep)
                        # 1.
                        if tag in self.verbs:
                            nb_verbs += 1
                        # 2.
                        if not re.search('(?u)^\W$', word):
                            length += 1
                        # 3.
                        else:
                            if word == '(':
                                paired_parentheses -= 1
                            elif word == ')':
                                paired_parentheses += 1
                            elif word == '"':
                                quotation_mark_number += 1
                        # 4.
                        raw_sentence += word + ' '
                    
                    # Remove extra space from sentence
                    raw_sentence = raw_sentence.strip()

                    if nb_verbs >0 and \
                        length >= self.nb_words and \
                        paired_parentheses == 0 and \
                        (quotation_mark_number%2) == 0 \
                        and not sentence_container.has_key(raw_sentence):
                        path = [node]
                        path.extend(shortestpath)
                        path.reverse()
                        weight = float(w) #/ float(length)
                        kshortestpaths.append((path, weight))
                        sentence_container[raw_sentence] = 1

                    #-B-------------------------------------------------------B-

                else:
            
                    # test if node has already been visited
                    if visited.has_key(node):
                        visited[node] += 1
                    else:
                        visited[node] = 0
                    id = visited[node]

                    # Add the node to orderedX
                    bisect.insort(orderedX, (w, node, id))
                    
                    # Add the node to paths
                    paths[(w, node, id)] = [node]
                    paths[(w, node, id)].extend(shortestpath)
    
        # Returns the list of shortest paths
        return kshortestpaths
    #-B-----------------------------------------------------------------------B-
    
    #-T-----------------------------------------------------------------------T-
    def get_compression(self, nb_candidates=50):
        """
        Searches all possible paths from **start** to **end** in the word graph,
        removes paths containing no verb or shorter than *n* words. Returns an
        ordered list (smaller first) of nb (default value is 50) (cummulative 
        score, path) tuples. The score is not normalized with the sentence 
        length.
        """

        # Search for the k-shortest paths in the graph
        self.paths = self.k_shortest_paths((self.start+self.sep+self.start, 0),
                                           (self.stop+self.sep+self.stop, 0),
                                            nb_candidates)

        # Initialize the fusion container
        fusions = []
        
        # Test if there are some paths
        if len(self.paths) > 0:
        
            # For nb candidates 
            for i in range(min(nb_candidates, len(self.paths))):
                nodes = self.paths[i][0]
                sentence = []
                
                for j in range(1, len(nodes)-1):
                    word, tag = nodes[j][0].split(self.sep)
                    sentence.append((word, tag))

                bisect.insort(fusions, (self.paths[i][1], sentence))

        return fusions
    #-B-----------------------------------------------------------------------B-

    #**************************************************************************
    # Path selection with core-rank-score
    #**************************************************************************

    def get_n_grams(self, sentence, n):
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

    def get_sentence_score(self, sentence, my_model, n, unknownwordprob=0):
        score = 0
        n_grams = self.get_n_grams(sentence, n)
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


    def sentence_core_rank_score(self, nbest_compressions):
        ll = len(nbest_compressions)
        scores = np.zeros(ll)
        for i in range(ll):
            sentence = nbest_compressions[i][1]
            sentence = " ".join([word[0] for word in sentence])
            sentence = cr.clean_text_simple(sentence,pos_filtering=False, stemming=False)
            #print sentence
            for j in range(len(sentence)):
                scores[i] += self.core_rank_scores[sentence[j]]
        return scores

    def fluency_score(self, nbest_compressions):
        all_scores = []
        for w, sentence in nbest_compressions:
            sentence_clean = " ".join([word[0] for word in sentence])
            all_scores.append(self.get_sentence_score(sentence=sentence_clean, my_model=self.my_lm, n=3))
        return all_scores
        
    def final_score(self, nbest_compressions):
        ll = len(nbest_compressions)
        scores = []
        cr_score = self.sentence_core_rank_score(nbest_compressions)
        fl_score = self.fluency_score(nbest_compressions)
        for i in range(ll):
            sentence_len = len(nbest_compressions[i][1])
            score = nbest_compressions[i][0]/fl_score[i]/(sentence_len*cr_score[i])
            bisect.insort(scores, (score , nbest_compressions[i][1]))

        return scores
    #**************************************************************************
    # END Path selection with core-rank-score
    #**************************************************************************




    #-T-----------------------------------------------------------------------T-
    def max_index(self, l):
        """ Returns the index of the maximum value of a given list. """

        ll = len(l)
        if ll < 0:
            return None
        elif ll == 1:
            return 0
        max_val = l[0]
        max_ind = 0
        for z in range(1, ll):
            if l[z] > max_val:
                max_val = l[z]
                max_ind = z
        return max_ind
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def compute_statistics(self):
        """
        This function iterates over the cluster's sentences and computes the
        following statistics about each word:
        
        - term frequency (self.term_freq)
        """

        # Structure for containing the list of sentences in which a term occurs
        terms = {}

        # Loop over the sentences
        for i in range(self.length):
        
            # For each tuple (token, POS) of sentence i
            for token, POS in self.sentence[i]:
            
                # generate the word/POS token
                node = token.lower() + self.sep + POS
                
                # Add the token to the terms list
                if not terms.has_key(node):
                    terms[node] = [i]
                else:
                    terms[node].append(i)

        # Loop over the terms
        for w in terms:

            # Compute the term frequency
            self.term_freq[w] = len(terms[w])
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def load_stopwords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip().lower())

        # Return the set of stopwords
        return stopwords
    #-B-----------------------------------------------------------------------B-
    

    #-T-----------------------------------------------------------------------T-
    def write_dot(self, dotfile):
        """ Outputs the word graph in dot format in the specified file. """
        write_dot(self.graph, dotfile)
    #-B-----------------------------------------------------------------------B-

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ] Ending word_graph class
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
