import numpy as np
import collections
from collections import Counter, defaultdict
import loadutils
from keras.utils import to_categorical

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
    
    def __init__(self, modelName, y_true, raw_y_pred, y_pred=np.empty(0)):
        """
        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            raw_y_pred : raw predictions generated from model.predict() method. \
            this is a 2D matrix of shape (?, number of NER classes). Each row correspond to one 1-hot NER vector }
            y_pred : model prediction. same shape and format as y_true. if this is None then will be constructed from raw_y_pred 
            """
        if not y_pred.any():
            print ("since y_pred is not provided, converting 1-hot raw_y_pred of shape (?, number of NER classes)into categorical index y_pred array of shape(?,)")
        
        self.modelName = modelName
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
        self.gold_idx_dict = self.get_gold_idx_dict(self.y_true, self.y_pred)
        
        self.vocab = None  # connect to vocabData.vocab
        self.posTags = None
        self.nerTags = None
        self.capitalTags = None
        
        self.devX = None
        self.devX_pos = None
        self.devX_capitals = None
        self.devY = None
        self.devY_cat = None
        
        self.CE_list = None  # array of KL divergence values
        
        
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
 

    def connect_to_devData(self, devData):
        """
        connect to Dev data
        
        Argument:
            devData : (devX, devX_pos, devX_capitals, devY) loaded from loadutils.conll2003Data.formatWindowedData()
                      Notebook Example (model_train_impl.ipynb)
                      ```
                      devSents = vocabData.readFile( DEV_FILE)
                      (devX, devX_pos, devX_capitals, devY) = vocabData.formatWindowedData(devSents, windowLength=?)
                      ```              
        """
        self.devX = devData[0]
        self.devX_pos = devData[1]
        self.devX_capitals = devData[2]
        self.devY = devData[3]
        
        num_classes = self.raw_y_pred.shape[1] + 3
        self.devY_cat = to_categorical(self.devY.astype('float32'), num_classes=num_classes)
        self.devY_cat = np.array(list(map( lambda i: np.array(i[3:], dtype=np.float), self.devY_cat)), dtype=np.float)

        
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
    

    def get_gold_idx_dict(self, y_true, y_pred):
        """
        for each gold label, retrieve idx
        
        Arguments:
            y_true : trainY/devY/testY. an array of shape(?,). Each value correspond to the ner tag for a word
            y_pred : model prediction. same shape and format as y_true              
        
        Returns:
            gold_idx_dict : {gold3_idx:[idx, idx...], gold4_idx:[idx, idx...]}
        """
        gold_idx_dict = defaultdict(list)
        for gold in self.gold_pred_idx_dict:
            gold_idx_dict[gold] = np.hstack(self.gold_pred_idx_dict[gold].values())
        
        return gold_idx_dict
    
    
    def print_idxlist_to_textlists(self, idx_list, worst=True, k=None, devData=None, y_pred=None, \
                                   print_window=True, dataClass=None, return_indices=False):
        """
        with array of indices, print the training window sentences 
        can be applied to 

        Argument: 
            y_pred :    y_pred : model prediction. shape (?, ). each value refer to a NER class
            devData :   a list/tuple (devX, devX_pos, devX_capitals, devY) loaded from conll2003Data.formatWindowedData()
            idx_list :  array of indices each indicating one training example row. For example pass in one of the followings:
                        evaluation_helper.EvalDev_Report.hallucination_idx
                        evaluation_helper.EvalDev_Report.missed_ner_idx
                        evaluation_helper.EvalDev_Report.match_ner_idx
                        evaluation_helper.EvalDev_Report.mismatch_ner_idx
                        evaluation_helper.EvalDev_Report.mismatch_ner_idx
            worst :     True -- rank by worst examples. False -- rank by best examples
            k :         int. to look at only k examples
            dataClass : loadutils.conll2003Data() object
            vocab_type: one of the followings: 
                        "words" for english words 
                        "ner" for NER tags
                        "pos" for PoS tags 
                        "capitals" for capitalization tags

        Returns:
            (print out the in text format)
            txt_list : same list as idx_list, except that all indices are replaced by text
        """   
        print ("indices counts =", idx_list.shape[0])
        boo = "worst" if worst else "best"
        print ("ranked by {} cross-entropy loss".format(boo))
        
        idx_list = [idx for (idx,ce) in self.rank_predictions(idx_selected=idx_list, worst=worst) ]
        ce_list = [ce for (idx,ce) in self.rank_predictions(idx_selected=idx_list, worst=worst) ]
        if k is not None:
            print ("top {} results".format(k))
            idx_list = idx_list[:k]
            ce_list = ce_list[:k]            
        
        devData = (self.devX, self.devX_pos, self.devX_capitals, self.devY) if (devData is None) else devData
        y_pred = self.y_pred if (y_pred is None) else y_pred
        dataClass = self.dataClass if (dataClass is None) else dataClass
        
        word_windows = list(map(dataClass.vocab.ids_to_words, devData[0][idx_list]))
        pos_windows =  list(map(dataClass.posTags.ids_to_words, devData[1][idx_list]))
        capital_windows = list(map(dataClass.capitalTags.ids_to_words, devData[2][idx_list])) 
        gold_ner_class = [dataClass.nerTags.ids_to_words([tag]) for tag in devData[3][idx_list]]
        pred_ner_class = [dataClass.nerTags.ids_to_words([tag]) for tag in y_pred[idx_list]]    

        if word_windows:
            cen = len(word_windows[0])//2 
            for i in range(len(word_windows)):
                print ("\nID {}".format(idx_list[i]))
                if worst: print ("KL divergence {}".format(ce_list[i]))
                print ("FEATURES:   \"{}\", {}, {}".format(word_windows[i][cen], pos_windows[i][cen], \
                                                     capital_windows[i][cen]))
                print ("Gold NER    {}".format(gold_ner_class[i]))
                print ("Pred NER    {}".format(pred_ner_class[i]))
                print ("Text window {}".format(word_windows[i]))
                print ("PoS window  {}".format(pos_windows[i]))
                print ("Caps window {}".format(capital_windows[i]))
        else:
            print ("empty -- no predictions were made")

        if return_indices:
            return idx_list    

        
    def print_brief_summary(self):
        """
        print quick summary of precision, recall, f1, gold label and pred label counts
        """
        print ("Model     {}".format(self.modelName))
        print ("Precision {}".format(self.precision))
        print ("Recall    {}".format(self.recall))
        print ("f1 score  {}".format(self.f1))
        
        # work here
        print ("\nGold NER label counts:")
        for ner in self.gold_cts.keys():
            print ("{} : {} (tag{})".format(self.gold_cts[ner], self.nerTags.ids_to_words([ner]), ner))
        print ("\nPredicted NER label counts:")
        for ner in self.pred_cts.keys():
            print ("{} : {} (tag{})".format(self.pred_cts[ner], self.nerTags.ids_to_words([ner]), ner))            

       
    def print_gold_to_pred_counts(self, return_dict=False):
        """
        for each gold label, look at the distribution of predicted labels
        """
        for gold_label in self.gold_pred_ct_dict.keys():
            print ("\nGold label \"{}\" (tag{}), prediction label counts:".format(self.nerTags.ids_to_words([gold_label]),\
                                                                                  gold_label))
            sum_ct = sum(self.gold_pred_ct_dict[gold_label].values())
            for pred_label in self.gold_pred_ct_dict[gold_label].keys():
                ct = self.gold_pred_ct_dict[gold_label][pred_label]
                percent = round(float(ct)/sum_ct, 4) if (sum_ct!=0) else 0
                print ("{} ({}%): \"{}\" (tag{})".format(ct, percent, self.nerTags.ids_to_words([pred_label]), pred_label))
                
        if return_dict:
            return self.gold_pred_ct_dict
            
            
    def CE(self, p_true, p_model):
        """
        compute cross entropy between true and model label distributions. KL(P||Q) = CE(P,Q) - H(P)
        because true distribution is 1-hot (H(P) = 0). this CE is also equivalent to KL divergence
        
        Arguments:
            p_true : a single vector of length = number of classes. 1-hot from self.devY_cat in our case
            p_model : same shape as p_true. continuous float values.
        
        Returns : float
        """
        return np.sum(-np.array(p_true)*np.log2(np.array(p_model)))  
    

    def extract_all_CE(self):
        """
        compute all cross-entropy values on dev set predictions
        
        Returns : an array of shape (?,). float values referring to cross-entropy per example
        """
        return np.array([self.CE(p_true, p_model) for (p_true, p_model) in list(zip(self.devY_cat, self.raw_y_pred))])


    def rank_predictions(self, idx_selected=None, worst=True):        
        """
        rank the predictions by cross-entropy
        
        Arguments:
            worst : True -- rank by worst cross-entropy. False -- rank by best cross-entropy.
            idx_selected : an optional array if only these indices are used in ranking
            
        Returns:
            worst_list : an array of (idx, CE) tuples of the worst predictions
        """
        self.CE_list = self.extract_all_CE()
        worst_list = sorted([(idx,CE) for (idx,CE) in enumerate(self.CE_list)], key=lambda x:x[1], reverse=worst)
        if idx_selected is not None:
            worst_list = [(idx, CE) for (idx, CE) in worst_list if idx in idx_selected]
        
        return worst_list

    
    def print_overall_rank(self, worst=True, k=None, print_window=True):
        """
        print ranked overall predictions
        
        Arguments :
            worst : True -- rank by worst cross-entropy. False -- rank by best cross-entropy.
            k : top k results
        """
        self.print_idxlist_to_textlists(idx_list=np.arange(self.devY.shape[0]), \
                                        worst=worst, k=k, print_window=print_window,\
                                       return_indices=False)
        pass
    
    def print_rank_per_gold_label(self, worst=True, k=None, print_window=True):
        
        for gold_label in self.gold_idx_dict.keys():
            print ("----------------------------\nLabel {} (tag{})".format(self.nerTags.ids_to_words([gold_label])[0], gold_label))
            self.print_idxlist_to_textlists(idx_list=self.gold_idx_dict[gold_label], \
                                        worst=worst, k=k, print_window=print_window,\
                                       return_indices=False)
                   
           
    def print_whole_report(self, to_file=False):
        pass
    