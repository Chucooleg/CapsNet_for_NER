
import sys
import json
import buildCNNModel as cnn
from loadutils import retrieve_model, loadProcessedData

def printUsage():
    print("USAGE:\n\ntrain a CNN model")
    print("All training data must have already been saved with loadutils.saveProcessedData()")
    print("<model name> <hyper parameters file (JSON)> ")


def main():
    """
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
    
    if hypers["use_pos_tags"]:
        trainX_dict["x_pos"] = trainX_pos_cat
        devX_list_arrayS += [devX_pos_cat]
    
    if hypers['use_capitalization_info']:
        trainX_dict["x_capital"] = trainX_capitals_cat
        devX_list_arrayS += [devX_capitals_cat]
    
    model = cnn.draw_cnn_model( hyper_param=hypers, embedding_matrix=embedding_matrix, verbose=True)
    model = cnn.compile_cnn_model( hypers, model)
    
    print( "Training Model:", modelName)
    caps.fit_model( hypers, model, modelName, trainX_dict, devX_list_arrayS, trainY_dict, devY_list_arrayS)


if __name__ == '__main__':
    main()
    