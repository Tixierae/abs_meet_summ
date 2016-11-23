This document describes the contents of the MSR Abstractive Text Compression Dataset Release

This is the dataset described in the paper "A dataset and evaluation metrics for abstractive sentence and paragraph compression" [1].

The release contains the following data:

1. Original sentences and short paragraphs (texts) with corresponding crowd-sourced compressed versions and crowd-sourced ratings of each versions

The directory RawData contains the texts in the training, validation, and test sets, with file names  train.tsv , valid.tsv, and test.tsv

The format of each file is as follows:

On each line, we have information for one source text and corresponding compressed versions.

The format is SourceInfo[ ||| CompressionInfo]+

SourceInfo is information on the soure text and has the following fields (tab-separated):

SourceID \t Domain \t SourceText

The SourceID has one or more integers connected with _, for exmaple "15" or "101_102". There is one integer per sentence in the source text.

CompressionInfo is information about a compression for the SourceText.

CompressionInfo has the following fields [tab-separated]:

CompressedText \t JudgeId \t numRatings[\t Rating]^numRatings

This is the compressed text (un-tokenized), the JudgeId (the anonymized ids of one or more crowd-workers that proposed this compression ), and indication of how many ratings we have and the sequence of ratings.

Each Rating has the format:

CombinedRatingValue \t Meaning_quality_string \t Grammar_quality_string

The CombinedRatingValue is sufficient to indicate the meaning and grammaticality qualitiy values assigned by a single rater, and the subseqeunt string values are not strictly necessary.

The different numeric values of CombinedRatingValue have the following meanings:

6	Most important meaning Flawless language      (3 on meaning and 3 on grammar as per the paper's terminology)
7	Most important meaning Minor errors           (3 on meaning and 2 on grammar)
9	Most important meaning Disfluent or incomprehensible (3 on meaning and 1 on grammar)
11	Much meaning Flawless language                (2 on meaning and 3 on grammar)
12	Much meaning Minor errors                     (2 on meaning and 2 on grammar)
14	Much meaning Disfluent or incomprehensible    (2 on meaning and 1 on grammar)
21	Little or none meaning Flawless language      (1 on meaning and 3 on grammar)
22	Little or none meaning Minor errors           (1 on meaning and 2 on grammar)
24	Little or none meaning Disfluent or incomprehensible (1 on meaning and 1 on grammar)

2. Processed original texts with corresponding crowd-sourced compressions

For convenince, we also include processed versions of the texts in 1 above, where processing includes sentence-breaking, tokenization, dependency and constituency parsing, and word alignment from compressed to original texts.

The files in Processed\train.tsv Processed\valid.tsv and Processed\test.tsv contain information corresponding to the data in RawData

The format of the processed files is as follows:

On every line, we have:

ProccessedSourceInfo[ ||| ProcessedCompressionInfo]+

The order of the lines and the order of the compressions for each source is the same as in the rawdata.

ProccessedSourceInfo contains the following tab-separated fields: 

SourceID \t TokenizedSourceText \t SourceDependencyTrees \t SourceConstituencyTrees

The SourceIDs align with the SourceIDs in the RawData
TokenizedSourceText has been obtained using the tokenizer from the Stanford CoreNLP library. Sentence boundaries are indicated via <eos> tokens.
SourceDependencyTrees are obtained using the Stanford parser version 3.4.1, and constituency trees are obtained using the same parser. In case of multiple sentences in TokenizedSourceText, there will be mutliple dependency and consttituency trees separated by <eos>

ProcessedCompressionInfo contains the following tab-separated fields:

TokenizedCompressedText \t CompressionDependencyTrees \t CompressionConstituencyTrees \t WordAlignment

the processed compressions appear in the same order as the original un-processed compressions in the RawData
The Stanford tokenizer and parsers have been used here as well, and <eos> is used as a sentence separator.

The WordAlignment indicates a single source text token corresponding to each compression token. Thsi was obtained using Jacana and post-processing to align the null-aligned tokens.

3. Editting history for the generation of the text compressions.

As mentioned in the paper, we collected the history of edits that a crowd worker performed to create the compression. Some workers copied and pasted the original text and performed deletions. Others typed the compressed text from scratch. We recorded the status of the output field on every click, paste, or type action. 

This information is available in the file compressionhistory.tsv

The format of the file is as follows (tab-separated):

sourceID \t Domain \t SourceText \t CompressedText \t judgeID \t AverageMeaningQuality \t AverageGrammarQuality \t TimespentOnTask \t EditHistory

Here the sourceIDs are aligned to the ones used in 1. and 2. above. The Source and Compressed texts are un-processed (not tokenized). The judgeID is the anonymized id of the crowd worker that performed the task (these ids were also used in 1.). The average quality is obtained by averaging the numerical values of mtuliple rating judgements on meaning preservation and grammaticality.

The edit history shows a sequence of snapshots of the output text field after each action. The different snapshots are separated by a sequence of two spaces, so the delimiting is not entirely unambiguous.


4. The gudelines provided to the crowd workers are available in the Documents direcory. A screenshot of the interface can be seen there as well. There were separate gudelines for shortening a single sentence versus a short paragraph. The rating gudelines were the same for all rating tasks.

5. We also provide the output of the four automatic compression systems included in the study in [1], together with quality judgements of their meaning and grammaticality.

The format of the files is the same as the format of the crowd compressions in 1., except that the judgeIds are now the ids of the systems. We provide the four system outputs in separate files.

The four systems with corresponding rated output files, are, under the sub-directory RawData:
 
 test_output.t3.tsv is from the T3 system described in [2] and available from the first authors' website.
 test_outut.ilp.tsv is from the ILP model of [3]  with implementation avaialble at [4].
 test_output.seq2seq.tsv is from our re-implementation of [5].
 test_output.namas.tsv is from the open-source implementation [6] of [7]. 


 
 
 References
 
 1. Kristina Toutanova, Chris Brockett, Ke M. Tran, and Saleema Amershi. A dataset and evaluation metrics for abstractive sentence and paragraph compression. In Proceedings of EMNLP 2016.
 2. Trevor Cohn and Mirella Lapata. Sentence compression beyond word deletion. In Proceedings of COLING 2008.
 3. James Clarke and Mirella Lapata. Global inference for sentence compression: An integer linear programming approach. JAIR 2008.
 4. https://github.com/cnap/sentence-compression
 5. Katja Filippova, Enrique Alfonseca, Carlos A Colmenares, Lukasz Kaiser, and Oriol Vinyals. Sentence  compression  by  deletion  with  LSTMs. In Proceedings of EMNLP 2015.
 6. https://github.com/facebook/NAMAS
 7. Alexander M. Rush, Sumit Chopra, and  Jason  Weston.A  neural attention model for abstractive sentence summarization. In Proceedings of EMNLP 2015.
 
 
