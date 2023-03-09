import conllu
import stanza
import sklearn.metrics
import matplotlib.pyplot as plt
import time

def extract_conllu_data(filename: str, feature: str, sentences: bool = True):
    '''A function that allows for the extraction of the desired data from a conllu file, structured into sentences or not.
    
    Args:
        filename (str): The name of the .conllu file to be read.
        feature (str): The name of the desired conllu format feature.
        sentences (bool): Whether or not the output should be a list of lists of strings representing words in separate sentences.
        
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
    tokens = []
    features = []
    data = []
    
    with open(filename) as f:
        text = f.read()
    
    sents = conllu.parse(text)
    
    for sentence in sents:
        data.append(sentence.metadata['text'])
        sent_tokens = []
        sent_features = []
        for entry in sentence:
            token = entry['form']
            feat = entry[feature]
            
            sent_tokens.append(token)
            sent_features.append(feat)
            
            
        tokens.append(sent_tokens)
        features.append(sent_features)
            
    if not sentences:
        tokens = [x for sentence in tokens for x in sentence]
        features = [x for sentence in features for x in sentence]
        
    return tokens, features, data

def get_measures(gold_standard: list, predictions: list, labels: list = [], matrix: bool = False):
    '''A function intended for retrieving a selection of evaluation measures for comparing the gold standard and the tagger
    annotations. The measures are printed out and include accuracy, Matthew's Correlation Coefficient, per-class precision 
    and recall, as well as a confusion matrix, which, in addition, get saved locally. These measures are calculated using 
    functions from sklearn and pyplot.
    
    Args:
        gold_standard (list[str]): A list of gold standard labels.
        predictions (list[str]): A list of predicted labels.
    '''
    if labels == []:
        labels = sorted(list(set(gold_standard)))

    print('MEASURES:')
    print(f'Accuracy: {"{:.2%}".format(sklearn.metrics.accuracy_score(gold_standard, predictions))}')
    print(f'Matthew\'s Correlation Coefficient: {"{:.2%}".format(sklearn.metrics.matthews_corrcoef(gold_standard, predictions))}')
    print()
    print('MEASURES PER CLASS:')
    precision = sklearn.metrics.precision_score(gold_standard, predictions, average=None, labels=labels)
    print('Precision:')
    for i in range(0,len(labels)):
        print(f'\t{labels[i]}: {"{:.2%}".format(precision[i])}')
    recall = sklearn.metrics.recall_score(gold_standard, predictions, average=None, labels=labels)
    print('Recall:')
    for i in range(0,len(labels)):
        print(f'\t{labels[i]}: {"{:.2%}".format(recall[i])}')
    print()

    if matrix:
        print('Confusion matrix:')
        cm = sklearn.metrics.confusion_matrix(gold_standard, predictions)
        matrix = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12,12))
        matrix.plot(ax=ax)
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(timestr + "confusion_matrix.jpg")