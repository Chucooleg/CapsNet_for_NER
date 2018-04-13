import numpy as np


def convert_raw_y_pred(raw_y_pred):
    """
    Arguments:
        raw_y_pred : raw predictions generated from model.predict() method. \
        this is a 2D matrix of shape (?, number of NER classes). Each row correspond to one 1-hot NER vector
    
    Returns:
        an array of shape (?,). Each value correspond to the predicted ner tag for a word.\
        You can convert this array back to NER tags by vocabData.ner_vocab.ids_to_words()
    """
    return np.argmax(raw_y_pred, axis=1) + 3


def get_precision(y_true, y_pred):
    """
    compute precision = correct entities / predicted entities
    
    Arguments:
        y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
        y_pred : model prediction. same shape and format as y_true
    
    Returns:
        precision : real number
    """
    model_entities_filter = (y_pred != 3).astype("int") # of the words our model say has a NER class
    precision_correct_entities = (y_pred[np.where(model_entities_filter)] == y_true[np.where(model_entities_filter)]).astype("int")
    precision = np.sum(precision_correct_entities)/np.sum(model_entities_filter)   
    return precision


def get_recall(y_true, y_pred):
    """
    compute recall = correct entities / all gold entities
    
    Arguments:
        y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
        y_pred : model prediction. same shape and format as y_true
    
    Returns:
        recall : real number
    """
    true_entities_filter = (y_true != 3).astype("int") # of the words that truly has a NER class
    recall_correct_entities = (y_pred[np.where(true_entities_filter)] == y_true[np.where(true_entities_filter)]).astype("int")
    recall = np.sum(recall_correct_entities)/np.sum(true_entities_filter)
    return recall


def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1 = 2*(precision*recall)/(precision + recall)
    return (precision, recall, f1)

# !!
assert(get_f1(y_true=np.array([5,7,3,4]), y_pred=convert_raw_y_pred(np.array([[1,0,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,1,0,0,0]]))) == (1/2,1/3,0.4))


# ------------------------- IGNORE BELOW FOR NOW -------------------------


# our model hallucinated that there's a NER class
def get_ner_hallucination_idx(y_true, y_pred):
    """
    extract indices of rows which the model predicted to have a NER class but gold label has no NER class
    
    Arguments:
        y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
        y_pred : model prediction. same shape and format as y_true
    
    Returns:
        hallucination_idx : an array. values correspond to indices on the training input
    """
    true_O = (y_true==3).astype("int")
    predicted_ners = (y_pred!=3).astype("int")
    hallucination_filter = np.all([true_O, predicted_ners], axis=0).astype("int")
    hallucination_idx = np.nonzero(hallucination_filter)[0]
    return hallucination_idx


# our model missed a NER class
def get_missed_ner_idx(y_true, raw_y_pred):
    """
    extract indices of rows which the model predicted NOT to have a NER class but gold label has NER class
    
    Arguments:
        y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
        y_pred : model prediction. same shape and format as y_true
    
    Returns:
        missed_ner_idx : an array. values correspond to indices on the training input
    """
    predicted_O = (y_pred==3).astype("int")
    true_ners = (y_true!=3).astype("int")
    missed_ner_filter = np.all([predicted_O, true_ners], axis=0).astype("int")
    missed_ner_idx = np.nonzero(missed_ner_filter)[0]
    return missed_ner_idx


# ner exist in both model pred and gold 
# model got right/wrong
def get_matching_ner_idx(y_true, raw_y_pred):
    """
    extract indices of rows which both the model prediction and gold shows NER class. 
    
    Arguments:
        y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
        y_pred : model prediction. same shape and format as y_true
       
    Returns:
        correct_ner_idx: an array. values correspind to indices on the training input
        incorrect_ner_idx: same type as above
    """
    non_Os = np.all([(y_true!=3), (y_pred!=3)], axis=0).astype("int")
    matches = (y_true == y_pred).astype("int")
    mismatches = (y_true != y_pred).astype("int")
    match_ner_idx = np.nonzero(np.all([non_Os, matches],axis=0).astype("int"))[0]
    mismatch_ner_idx = np.nonzero(np.all([non_Os, mismatches],axis=0).astype("int"))[0]
    return match_ner_idx, mismatch_ner_idx