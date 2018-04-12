
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
           devX_pos_cat, trainY_cat, devY_cat, embedding_matrix = loadProcessedData()
    
    # contruct training dicts
    trainX_dict = {'x':trainX}
    devX_list = [devX]
    
    if hypers["use_pos_tags"]:
        trainX_dict["x_pos"] = trainX_pos_cat
        devX_list += [devX_pos_cat]
    
    if hypers['use_capitalization_info']:
        trainX_dict["x_capital"] = trainX_capitals_cat
        devX_list += [devX_capitals_cat]
    
    model = cnn.draw_capsnet_model( hyper_param=hypers, embedding_matrix=embedding_matrix, verbose=True)
    model = cnn.compile_model( hypers, model)
    
    print( "Training Model:", modelName)
    cnn.fit_model( hypers, model, modelName, trainX_dict, devX_list, trainY_cat, devY_cat)


if __name__ == '__main__':
    main()
    