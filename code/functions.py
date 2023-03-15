import conllu
import stanza
import sklearn.metrics
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm

def extract_conllu_data(filename: str, feature: str, sentences: bool = True, combined: bool = False, fulltext: bool = True):
    '''A function that allows for the extraction of the desired data from a conllu file, structured into sentences or not.
    
    Args:
        filename (str): The name of the .conllu file to be read.
        feature (str): The name of the desired conllu format feature.
        sentences (bool): Whether or not the output should be a list of lists of strings representing words in separate sentences.
        combined (bool): Whether or not the tokens and tags should be returned in one list of space-separated elements.
        fulltext (bool): Whether or not to extract and return the metadata sentences.
        
    Returns:
        A list of the original tokens (tokenized sentences), a list of the corresponding features, and a list of full original 
        sentences.
    '''
    #checking the validity of the feature argument
    possible_features = ['lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
    if feature not in possible_features:
        print('Please specify a valid feature type.')
        return
    
    # specifying lists
    tokens_features = []
    tokens = []
    features = []
    data = []
    
    # opening the file
    with open(filename) as f:
        text = f.read()
    
    # parsing the file
    sents = conllu.parse(text)
    
    # selecting the relevant data and adding it to relevant lists
    for sentence in sents:
        if fulltext:
            data.append(sentence.metadata['text'])
        sent_tokens_features = []
        sent_tokens = []
        sent_features = []
        for entry in sentence:
            token = entry['form']
            feat = entry[feature]
            
            sent_tokens.append(token)
            sent_features.append(feat)
            
            if combined:  # this will return a different data structure
                if feat == None:
                    feat = '_'
                sent_tokens_features.append(' '.join([token, feat]))
            
            
        tokens.append(sent_tokens)
        features.append(sent_features)
        
        if combined:
            tokens_features.append(sent_tokens_features)
    
    # unravelling the sentence-level lists if needed
    if not sentences:
        tokens = [x for sentence in tokens for x in sentence]
        features = [x for sentence in features for x in sentence]
        if combined:
            tokens_features = [x for sentence in tokens_features for x in sentence]
            
    if combined:
        if fulltext:
            return tokens_features, data
        else:
            return tokens_features
    else:
        if fulltext:        
            return tokens, features, data
        else:
            return tokens, features

def get_measures(gold_standard: list, predictions: list, labels: list = [], matrix: bool = False, details: bool = False):
    '''A function intended for retrieving a selection of evaluation measures for comparing the gold standard and the tagger
    annotations. The measures are printed out and include accuracy, Matthew's Correlation Coefficient, per-class precision 
    and recall, as well as a confusion matrix, which, in addition, get saved locally. These measures are calculated using 
    functions from sklearn and pyplot.
    
    Args:
        gold_standard (list[str]): A list of gold standard labels.
        predictions (list[str]): A list of predicted labels.
        labels (list[str]): A list of labels (if it needs to be specified).
        matrix (bool): Whether or not to produce a confusion matrix.
    '''
    
    if labels == []:  # setting up a list of labels based on the training data
        labels = sorted(list(set(gold_standard)))

    # printing out the measures
    print('MEASURES:')
    print(f'Accuracy: {"{:.2%}".format(sklearn.metrics.accuracy_score(gold_standard, predictions))}')
    print(f'Precision (weighted): {"{:.2%}".format(sklearn.metrics.precision_score(gold_standard, predictions, average="weighted", zero_division=0))}')
    print(f'Recall (weighted): {"{:.2%}".format(sklearn.metrics.recall_score(gold_standard, predictions, average="weighted", zero_division=0))}')
    print(f'F1 (weighted): {"{:.2%}".format(sklearn.metrics.f1_score(gold_standard, predictions, average="weighted", zero_division=0))}')
    print(f'Matthew\'s Correlation Coefficient: {"{:.2%}".format(sklearn.metrics.matthews_corrcoef(gold_standard, predictions))}')
    if details:
        print()
        print('MEASURES PER CLASS:')
        precision = sklearn.metrics.precision_score(gold_standard, predictions, average=None, labels=labels, zero_division=0)
        print('Precision:')
        for i in range(0,len(labels)):
            print(f'\t{labels[i]}: {"{:.2%}".format(precision[i])}')
        recall = sklearn.metrics.recall_score(gold_standard, predictions, average=None, labels=labels, zero_division=0)
        print('Recall:')
        for i in range(0,len(labels)):
            print(f'\t{labels[i]}: {"{:.2%}".format(recall[i])}')
        print()
    
    # printing out and saving the confusion matrix
    if matrix:
        print('Confusion matrix:')
        cm = sklearn.metrics.confusion_matrix(gold_standard, predictions)
        matrix = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12,12))
        matrix.plot(ax=ax)
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(timestr + "confusion_matrix.jpg")

def get_comparison(standard: list, predictions: list, tokens: list):
    '''A function that returns a comparison of where mistakes were made during annotation.
    
    Args:
        standard (list): A list of lists of gold standard annotations.
        predictions (list): A list of lists of predicted annotations.
        tokens (list): A list of original tokens corresponding to the tags.
    
    Returns:
        A Pandas dataframe containing the mismatched annotations, their context and tokens.
    '''
    
    problematic = []
    for i, ann in enumerate(predictions):
        if standard[i] != ann:
            if i != 0:
                preceding = tokens[i-1]
            else:
                preceding = ''
                
            if i != len(tokens)-1:
                succeeding = tokens[i+1]
            else:
                succeeding = ''

            problematic.append((tokens[i], ' '.join([preceding, tokens[i], succeeding]), standard[i], predictions[i]))
            
    problematic_frame = pd.DataFrame(problematic, columns=['Token', 'Context', 'Gold Standard', 'Prediction'])
    
    return problematic_frame

def get_confidence_comparison(problematic_frame: pd.DataFrame, processed_annotations: list):
    '''A function intended for enrichening of a regular comparison frame with confidences returned by the tagger.
    
    Args:
        problematic_frame (pd.DataFrame): An existing dataframe containing information about tokens, context, standard, and 
        predictions.
        processed_annotations (list[list]): A list of lists where every element represents information about the annotation as
            obtained from the tagger.
        
    Returns:
        A DataFrame containing the original token, context, gold standard, prediction, and the confidence of that prediction 
        for every mismatched prediction and gold standard.
    '''
    confidences = []
    for i, ann in enumerate(processed_annotations):
        confidences.append(ann[2])
            
    problematic_frame['Confidence'] = confidences
    
    return problematic_frame

def split_tags_and_tokens(tags: list):
    '''A function that splits every entry in a list by whitespace and into two separate lists.
    
    Args:
        tags (list): A list where every entry is a string containing whitespace.
        
    Returns:
        Two lists, containing the first and the second element of every entry from the original list.
    '''
    tokens = [x.strip().split()[0] for x in tags if len(x.strip()) > 0]
    tags = [x.strip().split()[1] for x in tags if len(x.strip()) > 0]

    return tokens, tags

def best_interpretation(dag_disamb: list):
    '''A function that allows for the selection of only the best possible morphosyntactic interpretation of a sentence as
    returned by Morfeusz2 + Concraft.
    
    Args:
        dag_disamb (list): A list of possible interpretations returned by Concraft based on Morfeusz2's analysis.
        
    Returns:
        A list containing only the highest probability interpretations for every token.    
    '''
    
    while("" in dag_disamb):
        dag_disamb.remove("")
    
    best_inter = []
    
    for item in dag_disamb:
        if item[0] == len(best_inter):
            best_inter.append(item)
        else:
            if item[3] > best_inter[-1][3]:
                best_inter[-1] = item
        
            
    return best_inter

class ConlluWord():
    '''A class intended to represent a line in a conll file (1 word, respectively).
    
    Attributes:
        ID (str): The index number of the word.
        FORM (str): The word as it appears in the sentence.
        LEMMA (str): The lemma of the word.
        UPOS (str): Universal Dependencies' universal part of speach tag.
        XPOS (str): Language-specific part of speech tag.
        FEATS (str): Features. Empty by default.
        HEAD (str): Signifies which word is the head of this one. Empty by default.
        DEPREL (str): Represents the dependency relations. Empty by default.
        DEPS (str): The dependencies of this word. Empty by default.
        MISC (str): Miscellaneous information. Empty by default.
        
    '''
    def __init__(
        self, ID: str, FORM: str, LEMMA: str, UPOS: str, XPOS: str, FEATS: str = '_', 
        HEAD: str = '_', DEPREL: str = '_', DEPS: str = '_', MISC: str = '_'
    ):
        '''The __init__ method of the class. Assigns the values to the ConLLu tags.
        
        Args:
            ID (str): The index number of the word.
            FORM (str): The word as it appears in the sentence.
            LEMMA (str): The lemma of the word.
            UPOS (str): Universal Dependencies' universal part of speach tag.
            XPOS (str): Language-specific part of speech tag.
            FEATS (str): Features. Empty by default.
            HEAD (str): Signifies which word is the head of this one. Empty by default.
            DEPREL (str): Represents the dependency relations. Empty by default.
            DEPS (str): The dependencies of this word. Empty by default.
            MISC (str): Miscellaneous information. Empty by default.
        '''
        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.UPOS = UPOS
        self.XPOS = XPOS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.DEPS = DEPS
        self.MISC = MISC
        
    def return_line(self):
        '''A method of the class that returns all the tags in the form of a tab-separated string, as per the ConLLu format.
        '''
        elements = [
            self.ID, self.FORM, self.LEMMA, self.UPOS, self.XPOS, self.FEATS, self.HEAD, 
            self.DEPREL, self.DEPS, self.MISC]
        line = "\t".join(elements)
        return line
    
class ConlluSentence():
    '''A class intended to represent a sentence entry in a conll file.
    
    Attributes:
        sent_id (str): The ID of the sentence.
        sent (str): The sentence itself.
        words (list[ConlluWord]): A list of ConlluWord objects representing constituent words and their annotation.
    '''
    def __init__(self, sent_id: str, sent: str, words: list):
        '''The __init__ method of the class. Assigns the arguments to internal attributes.
        Args:
            sent_id (str): The ID of the sentence.
            sent (str): The sentence itself.
            words (list[ConlluWord]): A list of ConlluWord objects representing constituent words and their annotation.
        ''' 
        self.sent_id = sent_id
        self.sent = sent
        self.words = words
        
    def return_sent(self):
        '''A method of the class that returns a sentence entry.
        '''
        whole_sent = '\n'.join([self.sent_id, self.sent] + [x.return_line() for x in self.words])
        return whole_sent
    
    def __len__(self):
        '''The __len__ magic method of the class.
            
        Returns:
            The length of all the elements in all sentences.
        '''
        return len(self.words)
    
    def __getitem__(self, index: int):
        '''The __getitem__ magic method of the class.
            
        Args:
            index (int): The index signifying the desired element.
            
        Returns:
            The ConlluWord object at the desired index.
        '''
                    
        return self.words[index]
    
class ConlluFormatter():
    '''A class intended to create a representation of the input text in ConLLu format using pre-annotated tags as well as
    annotation from Morfeusz and Concraft.
    
    Attributes:
        all_conll_sents (list[list]): A list of lists representing the sentences with their annotation stored.        
    '''
    def __init__(
        self, 
        annotations, 
        morfeusz, 
        concraft
    ):
        '''The __init__ method of the class.
        Creates a list for every sentence in the input that contains annotations for every word using the ConlluWord class.
        
        Args:
            sents (list[str]): A list of sentences as strings.
            tokenized (list[list[str]]): A list of tokenized sentences (in the same order as in sents).
            anns (list[list[str]]): A list of annotations corresponding to the words in tokenized.
            morfeusz (Morfeusz): a Morfeusz object that will be used for morphological analysis of the sentences.
            concraft (Concraft): a Concraft object that will be used for morphological disambiguation and annotation.
        '''
        self.all_conll_sents = []
        
        # retrieving the data sentence by sentence
        for i, sent in enumerate(tqdm(annotations.sentences)):
            conll_sent = []
            # getting the Morfeusz + Concraft info
            dag = morfeusz.analyse(sent)
            dag_disamb = concraft.disamb(dag)
            best_inter = best_interpretation(dag_disamb)
            # retrieving the manual annotations as well as tokens corresponding to the annotation
            ann = annotations.simple_gold_standard_tokenized[i]
            tokens = annotations.simple_sentences_tokenized[i]
            # setting up the offset that will be used for situations where tokens that were split in manual annotation were
            # not split in the machine one
            offset = 0
            j_offset = 0
            
            # retrieving the data word by word
            for j, inter in enumerate(best_inter):
                # defining the index, retrieving the word as detected by Morfeusz
                idx = str(j + j_offset + 1)
                form = inter[2][0]
                # excluding mistakenly detected ś tokens (that were not even split from the preceding word)    
                if form == "ś":
                    if best_inter[j-1][2][0].endswith('ś'):
                        offset += -1
                        j_offset += -1
                        continue
                # retrieving the lemma
                if len(inter[2][1]) > 1:
                    lemma = inter[2][1].split(':')[0]
                else:  # for when the lemma is just ':'
                    lemma = inter[2][1]
                # retrieving the UPOS tag for the word from the manual annotation, updating the offset accordingly
                try:
                    if tokens[j+offset] == form:
                        upos = ann[j+offset]      
                    elif tokens[j+offset] + tokens[j+offset+1] == form:
                        upos = ann[j+offset]
                        offset += 1
                    elif tokens[j+offset] + tokens[j+offset+1] + tokens[j+offset+2] == form:
                        upos = ann[j+offset]
                        offset += 2
                    else:    
                        upos = '_'
                except IndexError:
                    continue
                
                # retrieving the XPOS tag
                xpos = inter[2][2]
                
                # lowercasing the lemmas to match the UD standard
                if upos != 'PROPN':
                    lemma = lemma.lower()
                
                # creating a ConlluWord object to store the retrieved information, appending it to a temporary sentence list
                word = ConlluWord(idx, form, lemma, upos, xpos)
                conll_sent.append(word)
            
            # handling of compounded elements (only the ones marked with 'aglt' in XPOS are displayed this way by UD)
            tracker = []
            for j, word in enumerate(conll_sent):
                if word.XPOS.startswith('aglt') and word.UPOS == 'AUX':
                    if conll_sent[j+1].XPOS.startswith('aglt') and word.UPOS == 'AUX':
                        tracker.append(
                            (j-1, 
                             str(j)+'-'+str(j+2), 
                             conll_sent[j-1].FORM+word.FORM+conll_sent[j+1].FORM)
                        ) 
                    else:  # only 2 words connected
                        tracker.append((j-1, str(j)+'-'+str(j+1), conll_sent[j-1].FORM+word.FORM))
            # adding the additional entries
            for j, entry in reversed(list(enumerate(tracker))):
                word = ConlluWord(entry[1], entry[2], '_', '_', '_')
                conll_sent.insert(entry[0], word)
            
            # creating a ConlluSentence object, appending it to the internal list of all sentences
            full_sent = ConlluSentence('# sent_id = ' + str(i+1), '# text = ' + sent, conll_sent)
            self.all_conll_sents.append(full_sent)
            
    def __len__(self):
        '''A method of the class that returns the length of the internal storage of ConLLu sentences.
        '''
        return len(self.all_conll_sents)
    
    def __getitem__(self, index: int):
        '''A method of the class that returns the transformed sentence at a given index.
        
        Args:
            index (int): The index of the desired element.
        '''
        return self.all_conll_sents[index]
    
    def print_item(self, index: int):
        '''A method of the class that prints out the sentence at a given index.
        
        Args:
            index (int): The index of the desired element.
        '''
        print(self.all_conll_sents[index].return_sent())
    
    def print_conllu(self):
        '''A method of the class indended for printing out all of the annotation in the ConLLu format.
        '''
        for sentence in self.all_conll_sents:
            print(sentence.return_sent())
            print('\n')
        
    def write_conllu_2_file(self, filename: str):
        '''A method of the class indended for displaying saving all of the annotation in the ConLLu format.
        
        Args:
            filename (str): The name of the file the data should be saved to.
        '''      
        with open(filename, 'w') as f:
            for sentence in self.all_conll_sents:
                f.write(sentence.return_sent() + '\n\n')

    def retrieve_anns(self, feature):
        '''A method of the class indended for retrieving lists of annotations of a specific category (lemma, upos, or xpos).
        
        Args:
            feature (str): The name of the desired feature.
        '''    
        possible_features = ['lemma', 'upos', 'xpos']
        if feature not in possible_features:
            print('Please specify a valid feature type (lemma, upos, xpos).')
            return
        
        anns = []
        forms = []
        
        for i, sent in enumerate(self.all_conll_sents):
            ann = []
            form = []
            for j, word in enumerate(sent):
                if '-' in word.ID:
                    continue
                elif feature == 'lemma':
                    ann.append(word.LEMMA)
                    form.append(word.FORM)
                elif feature == 'upos':
                    ann.append(word.UPOS)
                    form.append(word.FORM)
                elif feature == 'xpos':
                    ann.append(word.XPOS)
                    form.append(word.FORM)
            anns.append(ann)
            forms.append(form)

        return anns, forms       
    
def get_lemma_measures(standard: list, predictions: list, lowercase: bool = False):
    '''A function that calculates and prints out the accuracy of the lemmatization.
    
    Args:
        standard (list): A list of lists of gold standard lemmas.
        predictions (list): A list of lists of predicted lemmas.
        lowercase (bool): Whether or not the data should be lowercased for comparison.
    '''
    if lowercase:
        standard = [x.lower() for sentence in standard for x in sentence]
        predictions = [x.lower() for sentence in predictions for x in sentence]
    else:
        standard = [x for sentence in standard for x in sentence]
        predictions = [x for sentence in predictions for x in sentence]
    print(f'Accuracy: {"{:.2%}".format(sklearn.metrics.accuracy_score(standard, predictions))}')

def get_lemma_comparison(standard: list, predictions: list, tokens: list, lowercase: bool = False):
    '''A function that calculates and prints out the accuracy of the lemmatization.
    
    Args:
        standard (list): A list of lists of gold standard lemmas.
        predictions (list): A list of lists of predicted lemmas.
        lowercase (bool): Whether or not the data should be lowercased for comparison.
    
    Returns:
        A Pandas dataframe containing the mismatched lemmas.
    '''
    tokens = [x.lower() for sentence in tokens for x in sentence]
    
    if lowercase:
        standard = [x.lower() for sentence in standard for x in sentence]
        predictions = [x.lower() for sentence in predictions for x in sentence]
    else:
        standard = [x for sentence in standard for x in sentence]
        predictions = [x for sentence in predictions for x in sentence]
            
    problematic_frame = get_comparison(standard, predictions, tokens)
    
    return problematic_frame