import conllu
import stanza
import sklearn.metrics
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
from preproc_bert import remove_ranges

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
    
    if isinstance(gold_standard[0], list):
        gold_standard = [x for sentence in gold_standard for x in sentence]
    if isinstance(predictions[0], list):
        predictions = [x for sentence in predictions for x in sentence]

    if labels == []:  # setting up a list of labels based on the training data
        labels = sorted(list(set(gold_standard)))
    else:
        if isinstance(labels[0], list):
            labels = [x for sentence in labels for x in sentence]

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

def get_comparison(standard: list, predictions: list, tokens: list, confidence=[]):
    '''A function that returns a comparison of where mistakes were made during annotation.
    
    Args:
        standard (list): A list of gold standard annotations.
        predictions (list): A list of predicted annotations.
        tokens (list): A list of original tokens corresponding to the tags.
    
    Returns:
        A Pandas dataframe containing the mismatched annotations, their context and tokens.
    '''

    if isinstance(standard[0], list):
        standard = [x for sentence in standard for x in sentence]
    if isinstance(predictions[0], list):
        predictions = [x for sentence in predictions for x in sentence]
    if isinstance(tokens[0], list):
        tokens = [x for sentence in tokens for x in sentence]
    
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
            
            if not confidence:
                problematic.append((tokens[i], ' '.join([preceding, tokens[i], succeeding]), standard[i], predictions[i]))
            else:
                if isinstance(confidence[0], list):
                    confidence = [x for sentence in confidence for x in sentence]
                problematic.append((tokens[i], ' '.join([preceding, tokens[i], succeeding]), standard[i], predictions[i], confidence[i]))
    if not confidence:        
        problematic_frame = pd.DataFrame(problematic, columns=['Token', 'Context', 'Gold Standard', 'Prediction'])
    else:
        problematic_frame = pd.DataFrame(problematic, columns=['Token', 'Context', 'Gold Standard', 'Prediction', 'Confidence'])
    
    return problematic_frame

def split_tags_and_tokens(tags: list):
    '''A function that splits every entry in a list by whitespace and into two separate lists.
    
    Args:
        tags (list): A list where every entry is a string containing whitespace.
        
    Returns:
        Two lists, containing the first and the second element of every entry from the original list.
    '''
    tokens = [x.strip().split()[0] for x in tags if len(x.strip()) > 1]
    tags = [(' ').join(x.strip().split()[1:]) for x in tags if len(x.strip()) > 1]

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

def get_lemma_measures(standard: list, predictions: list, lowercase: bool = False):
    '''A function that calculates and prints out the accuracy of the lemmatization.
    
    Args:
        standard (list): A list of lists of gold standard lemmas.
        predictions (list): A list of lists of predicted lemmas.
        lowercase (bool): Whether or not both of the lists should be lowercased.
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
        tokens (list): A list of lists of the tokens to be lemmatized.
        lowercase (bool): Whether or not standard and predictions should be lowercased.
    
    Returns:
        A Pandas dataframe containing the mismatched lemmas.
    '''
    tokens = [x for sentence in tokens for x in sentence]

    if lowercase:
        standard = [x.lower() for sentence in standard for x in sentence]
        predictions = [x.lower() for sentence in predictions for x in sentence]
    else:
        standard = [x for sentence in standard for x in sentence]
        predictions = [x for sentence in predictions for x in sentence]
            
    problematic_frame = get_comparison(standard, predictions, tokens)
    
    return problematic_frame

def make_tagger_friendly(tokens_tags):
    '''A function allowing for the use of split_tags_and_tokens and remove_ranges on nested lists.
    
    Arguments:
        token_tags (list[list]): A list of lists representing sentences with annotations.
        
    Returns:
        Two separate lists of lists representing sentences and their annotations respectively.'''
    tokens = []
    tags = []
    for element in tokens_tags:
        mini_tokens, mini_tags = split_tags_and_tokens(remove_ranges(element))
        tokens.append(mini_tokens)
        tags.append(mini_tags)
        
    return tokens, tags

def get_korba_data(filename: str):
    '''A function used for retrieving of the preprocessed Korba tags and annotations.
    
    Arguments:
        filename (str): The name of the file that the annotations should be retrieved from.
        
    Returns:
        Three lists of lists representing the tokens, lemmas, and xpos tag per original corpus file.'''
    with open(filename) as f:
        lines = f.readlines()
    
    tokens_list = []
    lemmas_list = []
    xpos_list = []

    tokens = []
    lemmas = []
    xpos = []
    for line in tqdm([x.strip() for x in lines], desc='Processing the annotations...'):
        if len(tokens) != 0 and len(line) == 0:
            tokens_list.append(tokens)
            lemmas_list.append(lemmas)
            xpos_list.append(xpos)
            tokens = []
            lemmas = []
            xpos = []

        elif len(line) > 0:
            tokens.append(line.split()[0])
            lemmas.append(line.split()[1])
            xpos.append(line.split()[2])
            
    tokens_list.append(tokens)
    lemmas_list.append(lemmas)
    xpos_list.append(xpos) 
            
    return tokens_list, lemmas_list, xpos_list

def get_full_table(standard: list, predictions: list, tokens: list, confidence=[], lowercase: bool = False):
    '''A function that returns a list of all the tokens with their predictions, gold standard, and context.
    
    Args:
        standard (list): A list of gold standard annotations.
        predictions (list): A list of predicted annotations.
        tokens (list): A list of original tokens corresponding to the tags.
        confidence (list): A list of prediction confidences, if available; empty by default.
        lowercase (bool): Whether or not standard and predictions should be lowercased.
    
    Returns:
        A Pandas dataframe containing the mismatched annotations, their context and tokens.
    '''

    if isinstance(standard[0], list):
        standard = [x for sentence in standard for x in sentence]
    if isinstance(predictions[0], list):
        predictions = [x for sentence in predictions for x in sentence]
    if isinstance(tokens[0], list):
        tokens = [x for sentence in tokens for x in sentence]

    if lowercase:
        standard = [x.lower() for x in standard]
        predictions = [x.lower() for x in predictions]
    
    all_entries = []
    for i, ann in enumerate(predictions):
        if i != 0:
            preceding = tokens[i-1]
        else:
            preceding = ''
                
        if i != len(tokens)-1:
            succeeding = tokens[i+1]
        else:
            succeeding = ''
            
        if not confidence:
            all_entries.append((tokens[i], ' '.join([preceding, tokens[i], succeeding]), standard[i], predictions[i]))
        else:
            if isinstance(confidence[0], list):
                confidence = [x for sentence in confidence for x in sentence]
            all_entries.append((tokens[i], ' '.join([preceding, tokens[i], succeeding]), standard[i], predictions[i], confidence[i]))
    if not confidence:        
        problematic_frame = pd.DataFrame(all_entries, columns=['Token', 'Context', 'Gold Standard', 'Prediction'])
    else:
        problematic_frame = pd.DataFrame(all_entries, columns=['Token', 'Context', 'Gold Standard', 'Prediction', 'Confidence'])
    
    return problematic_frame

def get_labels(filename: str):
    '''A function that extracts labels/tags from a .txt file.
    
    Args:
        filename (str): The name of the file.
    
    Returns:
        A list of tags.
    '''
    with open(filename) as f:
        labels = f.readlines()[1:]
        labels = [x.strip() for x in labels]
        
    return labels