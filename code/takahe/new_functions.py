#**************************************************************************
# filter candidates list
#**************************************************************************
def filter_cand(candidates, i):
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
def filter_cand_common_hyp(candidates, i):
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
def best_candidate_coreRank(candidates, word):
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
        ambinode_frequency.append(len( self.graph.node[candidate_nodes[l]]['info'] ))

    # Select the ambiguous node
    selected = self.max_index(ambinode_overlap)
    if ambinode_overlap[selected] == 0:
        selected = self.max_index(ambinode_frequency)
    
    return candidate_nodes[selected]

#**************************************************************************
# best candidate for common hyps according to path_similarity
# word-to-add's tagging should be transformed to wordnet's tagging
#**************************************************************************
def best_candidate_similarity(candidates, node_to_add):
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
def update_nodes(node_to_replace, node, i, j):
    token, pos = node.split(self.sep)
    new_id = ambiguous_nodes(node)
    new_node = (node, new_id)

    self.graph.add_node(new_node,
                        info=self.graph.node[node_to_replace]['info'],
                        label=token.lower())
    self.graph.node[new_node]['info'].append((i,j))

    update_mapping(new_node, node_to_replace)

    self.graph.remove_node(node_to_replace)
    return

def update_mapping(new_node, node_to_replace):
    for mapping_sentence in self.mapping:
        for mapping_node in mapping_sentence:
            if mapping_node == node_to_replace:
                mapping_node = new_node
    return

#**************************************************************************
# update best-candidate-node with new node of common-hyp
#**************************************************************************
def update_nodes_common_hyp(node_to_replace, common_hyp,  i, j):
    
    word_common = common_hyp.lemmas()[0].name()
    pos_common = node_to_replace.split(self.sep)[1]
    
    new_id = ambiguous_nodes(word_common + self.sep + pos_common)
    node_common = (word_common + self.sep + pos_common, new_id)


    self.graph.add_node(node_common,
                        info=self.graph.node[node_to_replace]['info'],
                        label=word_common.lower())
    self.graph.node[node_common]['info'].append((i,j))

    update_mapping(node_common, node_to_replace)

    self.graph.remove_node(node_to_replace)
    return

#**************************************************************************
# transform sentence to ensemble of words
#**************************************************************************
def concat (sentences):
    sentences = ' '.join(sentences)
    words = sentences.split(' ')
    words = [word.split("/")[0] for word in words]
    sentences = ' '.join(words)
    return sentences


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
    hyps_nodes = []  # return [(node1, common_hyp1),(node2, common_hyp2),...]
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


