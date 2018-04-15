import numpy as np
import collections
from collections import Counter, defaultdict
import loadutils

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


def compare_models_by_f1(modelName_list, y_true, return_results=False):
    """
    rank and compare models by f1 score
    print the ranked results
    
    Arguments:
        modelName_list: a list/tuple of strings. each string is a modelName used while training in model_training_tmpl.ipynb
        y_true : devY, loaded from conll2003Data.formatWindowedData(). see that documentation
    """
    models = defaultdict(int)
    for modelName in modelName_list:
        models[modelName] = get_f1_by_modelName(modelName, y_true)
    
    sorted_models = sorted(models.items(), key=lambda x:x[1], reverse=True) 
    for (i,(mod, f1)) in enumerate(sorted_models):
        print ("\nrank {}".format(i+1))
        print ("modelName:",mod)
        print ("f1=",f1)
        
    if return_results:
        return sorted_models
    
def get_f1_by_modelName(modelName, y_true):
    """
    access f1 score of a model directly by modelName
    
    Arguments:
        modelName : string, the modelName used while training in model_training_tmpl.ipynb
        y_true : devY, loaded from conll2003Data.formatWindowedData(). see that documentation
    
    Returns:
        float f1 score
    """
    _, _, y_pred = loadutils.loadDevPredictionsData(modelName)
    
    return get_f1(y_true, y_pred)


def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1 = 2*(precision*recall)/(precision + recall)
    return  f1

# !!
assert(get_f1(y_true=np.array([5,7,3,4]), y_pred=convert_raw_y_pred(np.array([[1,0,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,1,0,0,0]]))) == (0.4))


# ------------------------- WIP below but still useful -------------------------

class EvalDev_Report(object):
    """
    Construct a report object for a trained model from the dev set predictions and gold labels
    Make sure to load predictions from code/dev_Predictions/...npy before initializing this object
    To load predictions, see *loadutils.loadDevPredictionsData()
    """
    
    def __init__(self, y_true, raw_y_pred=np.empty(0), y_pred=np.empty(0)):
        """
        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            raw_y_pred : raw predictions generated from model.predict() method. \
            this is a 2D matrix of shape (?, number of NER classes). Each row correspond to one 1-hot NER vector }
            y_pred : model prediction. same shape and format as y_true. if this is None then will be constructed from raw_y_pred 
            """
        if not raw_y_pred.any() and not y_pred.any():
            raise ValueError("raw_y_pred and y_pred are both empty arrays. at least one of them must be provided \nprovide raw_y_pred as array of shape (?, embed_dim) or y_pred as array of shape (?,)")
        
        self.y_true = y_true
        self.raw_y_pred = raw_y_pred
        self.y_pred = self.convert_raw_y_pred(self.raw_y_pred) if (not y_pred.any()) else y_pred  
        
        self.gold_cts, self.pred_cts = self.count_ner_labels(self.y_true, self.y_pred)
        
        self.precision = self.get_precision(self.y_true, self.y_pred)
        self.recall = self.get_recall(self.y_true, self.y_pred)
        self.f1 = self.get_f1(self.y_true, self.y_pred)
        
        self.hallucination_idx = self.get_ner_hallucination_idx(self.y_true, self.y_pred)
        self.missed_ner_idx = self.get_missed_ner_idx(self.y_true, self.y_pred)
        self.match_ner_idx, self.mismatch_ner_idx = self.get_matching_ner_idx(self.y_true, self.y_pred)
        self.gold_pred_idx_dict, self.gold_pred_ct_dict = self.get_gold_pred_idx_dict(self.y_true, self.y_pred)
        
        self.vocab = None  # connect to vocabData.vocab
        self.posTags = None
        self.nerTags = None
        self.capitalTags = None
        
    def connect_to_dataClass(self, dataClass):
        """
        connect to vocabData.vocab objects
        
        Argument:
            dataClass : object constructed from loadutils.conll2003Data(). Typically it is `vocabData`
        """
        self.dataClass = dataClass
        self.vocab = dataClass.vocab
        self.posTags = dataClass.posTags
        self.nerTags = dataClass.nerTags
        self.capitalTags = dataClass.capitalTags
        
        
    def convert_raw_y_pred(self, raw_y_pred):
        """
        convert raw model predictions 
        
        Arguments:
            raw_y_pred : raw predictions generated from model.predict() method. \
            this is a 2D matrix of shape (?, number of NER classes). Each row correspond to one 1-hot NER vector } 
            
        Returns:
            an array of shape (?,). Each value correspond to the predicted ner tag for a word.\
            You can convert this array back to NER tags by vocabData.ner_vocab.ids_to_words()
        """
        return np.argmax(raw_y_pred, axis=1) + 3

    
    def count_ner_labels(self, y_true, y_pred):
        """
        report straight forward distribution of ner classes from gold and pred labels

        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            y_pred : model prediction. same shape and format as y_true  

        Returns:
            counter of y_true
            counter of y_pred
        """
        return Counter(y_true), Counter(y_pred)
    
        
    def get_recall(self, y_true, y_pred):
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
    
    
    def get_precision(self, y_true, y_pred):
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
    
    
    def get_f1(self, y_true, y_pred):
        """
        compute f1 score from precision and recall 

        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            y_pred : model prediction. same shape and format as y_true        
        
        Returns:
            f1 : float
        """
        precision = self.get_precision(y_true, y_pred)
        recall = self.get_recall(y_true, y_pred)
        f1 = 2*(precision*recall)/(precision + recall)
        return f1 
    
    
    def get_ner_hallucination_idx(self, y_true, y_pred):
        """
        our model hallucinated that there's a NER label ("O")
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

    
    def get_missed_ner_idx(self, y_true, y_pred):
        """
        our model missed a NER label (not "O")
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
    

    def get_matching_ner_idx(self, y_true, y_pred):
        """
        extract indices of matching and mismatching rows which both the model prediction and gold shows NER class (not "O")

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
    

    def get_gold_pred_idx_dict(self, y_true, y_pred):
        """
        for each gold label, look at what the model predicted
        for each gold label, count prediction class

        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            y_pred : model prediction. same shape and format as y_true    

        Returns:
            gold_pred_idx_dict : {gold3_idx:{pred3:[idx, idx...], pred4:[idx, idx...]}, gold4_idx:{...}}
            gold_pred_ct_dict :  {gold3_idx:{pred3:int, pred4:int}, gold4_idx:{...}}
        """
        gold_pred_idx_dict = defaultdict(lambda: defaultdict(list))
        gold_pred_ct_dict = defaultdict(lambda: defaultdict(int)) 

        for gold_idx in range(3,11):
            gold_filter = (y_true == gold_idx).astype("int") # 1/0 all rows with that gold_idx
            for pred_idx in range(3,11):
                pred_filter = (y_pred == pred_idx).astype("int") # 1/0 all rows with that ner_idx
                match_ner_idx = np.nonzero(np.all([gold_filter, pred_filter],axis=0).astype("int"))[0]
                gold_pred_idx_dict[gold_idx][pred_idx] = match_ner_idx  
                gold_pred_ct_dict[gold_idx][pred_idx] = match_ner_idx.shape[0]              

        return gold_pred_idx_dict, gold_pred_ct_dict   
    
    
    
    
    
    def calc_cross_entropy_per_row(y_true_row, y_pred_row):
        pass
    
    def print_report():
        pass
    
    # sample fro idx
    # look up A0
    def worst_by_cross_entropy(self):
        pass