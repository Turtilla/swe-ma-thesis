import conllu
import stanza
import sklearn.metrics
import matplotlib.pyplot as plt
import time
import pandas as pd

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