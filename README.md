* Relevant papers can be found in the corresponding folder

* Some useful links:

  * [Boudin and Morin 2013 repo with graph-building code](https://github.com/boudinfl/takahe)
  * [using WordNet in Python](http://www.nltk.org/howto/wordnet.html)
  * language modeling in Python (to be used for the *path ranking and selection* step):
    * [1](http://www.nltk.org/api/nltk.model.html)
	* [2](https://github.com/senarvi/theanolm)
  
  
* Next steps:

  * Zekun & Wensi: 
   *(1) fix what remains to be fixed in the graph building code
   *(2) test that it works as expected on the Microsoft (English) and LINA (French, just added) sentence compression data sets
   *(3) implement path ranking function based on original Filipova edge weights, keyphrase scores (for coverage) and language model (for fluency)
  * Antoine: perform community detection (grouping of related sentences) and community cleaning (elimination of redundant and non-informative sentences) in an *unsupervised* way
  * next face-to-face progress meeting to be determined before Christmas break. I am always available via email for questions.

  
* Word embeddings resources:

 * note that the Word Mover's Distance paper and the CoreRank (EMNLP 2016) paper are already in the 'papers' folder.
 * [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library in Python for word2vec
 * [training visualization demo (for Chrome)](https://ronxin.github.io/wevi/)
 * [interesting blogpost](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
 * [introduction slides with links to the important papers](http://www.lix.polytechnique.fr/~anti5662/word_embeddings_intro_tixier.pdf)