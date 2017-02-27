* Relevant papers can be found in the corresponding folder

* Some useful links:
  * list of stopwords (English) from the SMART information retrieval system can be found [here](http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop)
  * [Boudin and Morin 2013 repo with graph-building code](https://github.com/boudinfl/takahe)
  * [using WordNet in Python](http://www.nltk.org/howto/wordnet.html)
  * language modeling in Python (to be used for the *path ranking and selection* step):
    * [Python NLTK language models](http://www.nltk.org/api/nltk.model.html)
	* [Language Modeling with Python's Theano](https://github.com/senarvi/theanolm)
  * [Microsoft Concept Graph](https://concept.research.microsoft.com/) (as a possible alternative to WordNet for hypo/hypernyms)
  
  
* Next steps:
  
   *(1) integrate word attraction force
   *(2) try embedding-based clustering of keywords
   *(3) re-organize the code so that all tuning parameters are easily accessible
   *(4) run experiments for traditional documents and meetings

* Word embeddings resources:

 * note that the Word Mover's Distance paper and the CoreRank (EMNLP 2016) paper are already in the 'papers' folder.
 * [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library in Python for word2vec
 * [training visualization demo (for Chrome)](https://ronxin.github.io/wevi/)
 * [interesting blogpost](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
 * [introduction slides with links to the important papers](http://www.lix.polytechnique.fr/~anti5662/word_embeddings_intro_tixier.pdf)