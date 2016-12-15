* Relevant papers can be found in the corresponding folder

* Some useful links:
  * list of stopwords (English) from the SMART information retrieval system can be found [here](http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop)
  * [Boudin and Morin 2013 repo with graph-building code](https://github.com/boudinfl/takahe)
  * [using WordNet in Python](http://www.nltk.org/howto/wordnet.html)
  * language modeling in Python (to be used for the *path ranking and selection* step):
    * [1](http://www.nltk.org/api/nltk.model.html)
	* [2](https://github.com/senarvi/theanolm)
  * [Microsoft Concept Graph](https://concept.research.microsoft.com/) (as a possible alternative to WordNet for hypo/hypernyms)
  
  
* Next steps:

  * Zekun & Wensi:
  
   *(1) fix what remains to be fixed in the graph building code (syns, hyper/hypo, stopwords...)
   *(2) test that it works as expected on examples manually built (either from English Google News or from AMI/ICSI corpora)
   *(3) implement path scoring and ranking module as we discussed
  * Antoine: perform community detection (grouping of related sentences) and community cleaning (elimination of redundant and non-informative sentences) in an *unsupervised* way. Send some example of clusters to Zekun & Wensi
  * next face-to-face progress meeting to be determined after Christmas break. I am always available via email for questions, even during the break (it just might take me more time to respond).

  
* Word embeddings resources:

 * note that the Word Mover's Distance paper and the CoreRank (EMNLP 2016) paper are already in the 'papers' folder.
 * [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library in Python for word2vec
 * [training visualization demo (for Chrome)](https://ronxin.github.io/wevi/)
 * [interesting blogpost](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
 * [introduction slides with links to the important papers](http://www.lix.polytechnique.fr/~anti5662/word_embeddings_intro_tixier.pdf)