
import sys
import json
import buildCapsModel as caps
from loadutils import retrieve_model, loadProcessedData, saveDevPredictionsData
from evaluation_helper import convert_raw_y_pred, get_f1, get_precision, get_recall

def printUsage():
    print("USAGE:\n\ntrain a capsnet model")
    print("All training data must have already been saved with loadutils.saveProcessedData()")
    print("<model name> <hyper parameters file (JSON)> ")


def main():
    """
    train a capsnet model
    command line arguments:
    <model name> <hyper parameters file (JSON)> 
    """
    
    if len(sys.argv) < 3:
        printUsage()
        return -1
    
    modelName = sys.argv[1]
    
    with open(sys.argv[2]) as fp:
        hypers = json.load( fp)
    
    trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat, \
           devX_pos_cat, trainY_cat, devY_cat, embedding_matrix, train_decoderY, dev_decoderY = loadProcessedData()
    
    # contruct training dicts
    trainX_dict = {'x':trainX}
    devX_list_arrayS = [devX]
    trainY_dict = {'out_pred':trainY_cat}
    devY_list_arrayS = [devY_cat]
    
    # for final prediction
    devX_dict = {'x':devX} #for model_eval only
    
    if hypers["use_pos_tags"]:
        trainX_dict["x_pos"] = trainX_pos_cat
        devX_list_arrayS += [devX_pos_cat]
        devX_dict["x_pos"] = devX_pos_cat #for model_eval only
    
    if hypers['use_capitalization_info']:
        trainX_dict["x_capital"] = trainX_capitals_cat
        devX_list_arrayS += [devX_capitals_cat]
        devX_dict["x_capital"] = devX_capitals_cat  #for model_eval only
    
    if hypers['use_decoder']:
        trainX_dict["decoder_y_cat"] = trainY_cat
        devX_list_arrayS += [devY_cat]
        trainY_dict["train_decoder_output"] = train_decoderY
        devY_list_arrayS += [dev_decoderY]
    
    if hypers['use_decoder']:
        model, model_eval = caps.draw_capsnet_model( hyper_param=hypers, embedding_matrix=embedding_matrix, verbose=True)

    else:
        model = caps.draw_capsnet_model( hyper_param=hypers, embedding_matrix=embedding_matrix, verbose=True)
    
    model = caps.compile_caps_model( hypers, model)   
    
    
    print( "\nTraining Model:", modelName)
    caps.fit_model( hypers, model, modelName, trainX_dict, devX_list_arrayS, trainY_dict, devY_list_arrayS)
    
    # making dev predictions from here downwards
    if hypers['use_decoder']:
        print ("making model prediction on dev set... \nusing model_eval because decoder is enabled.") 
        raw_y_pred, raw_y_pred_decoder_embeddings = model_eval.predict(devX_dict, verbose=1) 
    else:
        print ("making model prediction on dev set... \nusing trained model as is because decoder is DISabled.")
        raw_y_pred = model.predict(devX_dict, verbose=1) 
    
    print ("prediction on dev set finished. raw 1-hot prediction has shape {}".format(raw_y_pred.shape))
    y_pred = convert_raw_y_pred(raw_y_pred)
    print ("prediction converted to class idx has shape {}".format(y_pred.shape))
    print ("decoder embedding prediction has shape {}".format(raw_y_pred_decoder_embeddings.shape))
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1_score = get_f1(y_true, y_pred)
    print ("precision on dev = {}".format(precision))
    print ("recall on dev = {}".format(recall))
    print ("f1 score on dev = {}".format(f1_score))
           
    # write out dev predictions
    modelsDir = 'dev_Predictions'
    print ("saving prediction data under directory: {}".format(modelsDir))
    saveDevPredictionsData(modelName, raw_y_pred, raw_y_pred_decoder_embeddings, y_pred, modelsDir)
    print ("please use loadutils.loadDevPredictionsData(modelName, modelsDir='dev_Predictions') to load :\n raw_y_pred, raw_y_pred_decoder_embeddings, y_pred")
    
if __name__ == '__main__':
    
    main()
    