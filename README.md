# swe-ma-thesis
[LTR project](https://github.com/Turtilla/ltr-project)

## REPOSITORY STRUCTURE:
* `code`: Contains all the code (and some of the data) for the project. Large models are not included.
  * `bert`: Contains the parameters trained models for [BERT](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification) as well as their training, testing, eval data and some other related scripts from the aforementioned link.
  * `marmot`: Contains data needed for training, evaluating, and testing a [Marmot](https://github.com/muelletm/cistern/blob/wiki/marmotTraining.md) model.  
  * `preproc_bert.py`: A script used for preprocessing files for BERT to remove unnecessary tokens (that describe the range of agglutinations of tokens).
  * `functions.py`: A Python file containing functions that are reused across multiple Jupyter Notebooks (also for the sake of keeping the notebooks shorter).
  * `requirements.txt`: A list of necessary libraries/modules (not complete).
  * `morfeusz_concraft_preannotation.ipynb`: A notebook that was used to generate the CoNLL-U files with lemma, UPOS, XPOS information for manual annotation.
  * `NKJP-vocabulary-check.ipynb`: A notebook where the vocabulary from the memoir and the test set are tested against the NKJP.
  * `bigram-and-trigram-statistics.ipynb`: A notebook where bigram and trigram counts for UPOS and XPOS tags for the test set and the memoir are obtained.
  * `korba_data_extraction.ipynb`: A notebook where Korba data is extracted - **this data is not used due to a different tagset!**
  * **`*.ipynb`: Other notebooks detailing the evaluation of different taggers and lemmatizers.**
* `data`: Contains the majority of the data used in the project.
  * `mistakes`: Contains the outputs of the Jupyter Notebooks with mistakes made by specific taggers or lemmatizers.
  * `ngram_stats`: Contains the statistics for the ngrams.
  * *`ud-treebanks`: Contains the UD treebanks out of which PDB is used - not included in this repository due to size.*
  * *`korba_corpus`: Contains the [Korba Corpus](https://korba.edu.pl/download), a corpus of 17th and 16th century Polish. Not included in the repository due to size and not used due to a different tagset.*
  * `memoirs.txt`: The text of the memoir, with 1 line corresponding to 1 paragraph.
  * `memoirs_annotated*.txt`: Different files containing different stages and/or amounts of annotation. Refer to `memoirs_annotated_10k.txt` for the most recent version of UPOS annotation in that format.
  * `memoirs*.conllu`: Different files containing preannotation or annotation in a CoNLL-U format. Refer to `memoirs_3k_corrected.conllu` for the fully corrected version (UPOS, XPOS, lemma), and `memoirs_10k_corrected.conllu` for a version with 10k tokens with manual UPOS annotation, but not all XPOS and lemma annotation has been reviewed in that file.
  * `korba*.txt`: Two files with selected Korba data that was not used due to differences in the tagset.
* `thesis-plan`: Contains different versions of the thesis plan and its presentation.

#### TODO tagging:
+ ~~Tag 10k tokens with basic tags.~~
+ ~~Decide how many tokens to tag with advanced tags + lemmas.~~
+ ~~Pre-tag with the above.~~
+ ~~Fix issues with reconstructing and spaces after t.p..~~
+ ~~Finish advanced tagging.~~
+ ~~Make sure that the ranges are correct.~~
+ ~~Review the tagging.~~

#### TODO taggers:
+ ~~Decide on a method to use BERT for tagging ([existing architecture?](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification))~~ 
+ ~~(Potentially) train Marmot on the desired data.~~
+ ~~Figure out how to access the UD one.~~
+ ~~Figure out how to use Morfeusz.~~
+ ~~Figure out how to use the lemmatizer.~~

#### TODO eval:
+ ~~Run evaluation of all the taggers/lemmatizers.~~
+ ~~Stanza~~
+ ~~BERT~~ ~~need to process the errors!~~
+ ~~Marmot~~
+ ~~UD~~
+ ~~Morfeusz~~
+ ~~Extract and review the errors.~~
+ ~~Annotate errors.~~
+ ~~Display errors, get error statistics.~~
+ ~~Find patterns.~~

#### TODO syntax:
+ ~~Get bigram & trigram stats for corpora and the text for different kinds of annotation.~~
+ ~~Compare differences.~~

#### TODO NKJP:
+ ~~Figure out the pipeline for retrieving items.~~
+ ~~Decide on potential search conditions.~~
+ ~~Test by lemma+**.~~
+ ~~Test by token.~~

#### TODO writing
+ ~~Set up the template, fix potential issues.~~
+ Set up references.
+ Abstract.
+ Introduction.
+ Background, prior research.
+ ~~Tools and methods.~~
+ ~~Results.~~
+ ~~Discussion.~~
+ Ethical concerns **in progress**
+ Limitations.
+ Conclusions.
+ Future work.
+ Proofreading.

