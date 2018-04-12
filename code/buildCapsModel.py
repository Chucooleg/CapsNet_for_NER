


from keras import callbacks, optimizers
from keras import backend as KB
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Reshape, Concatenate
from keras.layers import Flatten, SpatialDropout1D, Conv1D
from keras.models import Model
from keras.utils import plot_model

# capsule layers from Xifeng Guo 
# https://github.com/XifengGuo/CapsNet-Keras
from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask



# Build the CapsNet model

def draw_capsnet_model( hyper_param, embedding_matrix=None ,verbose=True):
    """
    Input: hyper_parameters dictionary
    
    Construct:
        input layers : x , x_pos(o), x_captialization(o)
        embedding matrix : use_glove or randomly initialize
        conv1 : first convolution layer
        primarycaps : conv2 and squash function applied
        ner_caps : make 8 ner capsules of specified dim
        out_pred : calc length of 8 ner capsules as 8 prob. predictions over 8 ner classes
    
    Returns: keras.models.Model object
    """
    # input layer(s)
    x = Input(shape=(hyper_param['maxlen'],), name='x')
    if hyper_param['use_pos_tags'] : 
        x_pos = Input(shape=(hyper_param['maxlen'],hyper_param['poslen']), name='x_pos')
    if hyper_param['use_capitalization_info'] : 
        x_capital = Input(shape=(hyper_param['maxlen'], hyper_param['capitallen']), name='x_capital') 

    # embedding matrix
    if hyper_param['use_glove']:
        embed = Embedding(hyper_param['max_features'], hyper_param['embed_dim'], weights=[embedding_matrix],\
                          input_length=hyper_param['maxlen'], trainable=hyper_param['allow_glove_retrain'])(x)
    else:
        embed = Embedding(hyper_param['max_features'], hyper_param['embed_dim'], input_length=hyper_param['maxlen'],\
                          embeddings_initializer="random_uniform" )(x)

    # concat embeddings with additional features
    if hyper_param['use_pos_tags'] and hyper_param['use_capitalization_info'] : 
        embed = Concatenate(axis=-1)([embed, x_pos, x_capital])
    elif hyper_param['use_pos_tags'] and (not hyper_param['use_capitalization_info']) :
        embed = Concatenate(axis=-1)([embed, x_pos])
    elif (not hyper_param['use_pos_tags']) and hyper_param['use_capitalization_info'] :
        embed = Concatenate(axis=-1)([embed, x_capital])    
    else :
        embed = embed

    # add dropout here
    if hyper_param['embed_dropout'] > 0.0:
        embed = SpatialDropout1D( hyper_param['embed_dropout'])(embed)

    # feed embeddings into conv1
    conv1 = Conv1D( filters=hyper_param['conv1_filters'], \
                   kernel_size=hyper_param['conv1_kernel_size'],\
                   strides=hyper_param['conv1_strides'], \
                   padding=hyper_param['conv1_padding'],\
                   activation='relu', name='conv1')(embed)

    # make primary capsules
    if hyper_param['use_2D_primarycaps']:
        convShape = conv1.get_shape().as_list()
        conv1 = Reshape(( convShape[1], convShape[2], 1))(conv1)
        primaryCapLayer = PrimaryCap
    else:
        primaryCapLayer = PrimaryCap1D

    primarycaps = primaryCapLayer(conv1, \
                             dim_capsule=hyper_param['primarycaps_dim_capsule'],\
                             n_channels=hyper_param['primarycaps_n_channels'],\
                             kernel_size=hyper_param['primarycaps_kernel_size'], \
                             strides=hyper_param['primarycaps_strides'], \
                             padding=hyper_param['primarycaps_padding'])

    # make ner capsules
    ner_caps = CapsuleLayer(num_capsule=hyper_param['ner_classes'], \
                            dim_capsule=hyper_param['ner_capsule_dim'], \
                            routings=hyper_param['num_dynamic_routing_passes'], \
                            name='nercaps')(primarycaps)

    # replace each ner capsuel with its length
    out_pred = Length(name='out_pred')(ner_caps)

    if verbose:
        print ("x", x.get_shape())
        if hyper_param['use_pos_tags'] : print ("x_pos", x_pos.get_shape())
        if hyper_param['use_capitalization_info'] : print ("x_capital", x_capital.get_shape())
        print ("embed", embed.get_shape())
        print ("embed", embed.get_shape())
        print ("conv1", conv1.get_shape())
        print ("primarycaps", primarycaps.get_shape())   
        print ("ner_caps", ner_caps.get_shape())
        print ("out_pred", out_pred.get_shape())

    # return final model
    if hyper_param['use_pos_tags'] and hyper_param['use_capitalization_info'] : 
        capsmodel = Model(inputs=[x, x_pos, x_capital], outputs=[out_pred])
    elif hyper_param['use_pos_tags'] and (not hyper_param['use_capitalization_info']) :
        capsmodel = Model(inputs=[x, x_pos], outputs=[out_pred])
    elif (not hyper_param['use_pos_tags']) and hyper_param['use_capitalization_info'] :
        capsmodel = Model(inputs=[x, x_capital], outputs=[out_pred])   
    else :
        capsmodel = Model(inputs=[x], outputs=[out_pred])

    return capsmodel


# margin loss
def margin_loss( y_true, y_pred):
    L = y_true * KB.square(KB.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * KB.square(KB.maximum(0., y_pred - 0.1))
    return KB.mean(KB.sum(L, 1))

# decoder loss work in progress
def decoder_loss( decoder_y_true, decoder_y_pred):
    # get cosine similarity from A3
    dot_products = np.dot(decoder_y_true, decoder_y_pred)
    l2norm_products = np.multiply(np.linalg.norm(decoder_y_true), np.linalg.norm(decoder_y_pred))
    cos_sim = np.divide(dot_products, l2norm_products)
    return -cos_sim


# compile the model
def compile_model( hyper_param, model):
    """
    Input: keras.models.Model object, see draw_capsnet_model() output. 
    This is a graph with all layers drawn and connected
    
    do:
        compile with loss function and optimizer
    
    Returns: compiled model
    """
    if hyper_param['optimizer'] == "Adam":
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif hyper_param['optimizer'] == "SGD": 
        opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    elif hyper_param['optimizer'] == None:
        raise Exception("No optimizer specified")

    if hyper_param.get('use_decoder') == True:
        model_loss = margin_loss # work in progress
    else:
        model_loss = margin_loss
    
    x = model.compile(optimizer=opt, #'adam',
                  loss=model_loss,
                  metrics=['accuracy'])
    
    return model

def fit_model( hyper_param, model, modelName, trainX_dict, devX_list_arrayS, trainY_array, devY_array):
    #Saving weights and logging
    log = callbacks.CSVLogger(hyper_param['save_dir'] + '/{0}_historylog.csv'.format(modelName))
    tb = callbacks.TensorBoard(log_dir=hyper_param['save_dir'] + '/tensorboard-logs', \
                               batch_size=hyper_param['batch_size'], histogram_freq=hyper_param['debug'])
    checkpoint = callbacks.ModelCheckpoint(hyper_param['save_dir'] + '/weights-{epoch:02d}.h5', \
                                           save_best_only=True, save_weights_only=True, verbose=1)
    es = callbacks.EarlyStopping(patience=hyper_param['stopping_patience'], verbose=2)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    model.summary()
    
    # Save a png of the model shapes and flow
    # must have installed pydot and graphviz...
    # conda install pydot
    # conda install -c anaconda graphviz
    # sometimes graphviz is a little squirrely, if so, use: pip install graphviz
    plot_model( model, to_file=hyper_param['save_dir'] + '/{0}.png'.format(modelName), show_shapes=True)

    #loss = margin_loss
    
    data = model.fit( x=trainX_dict, 
                      y=trainY_array, 
                      batch_size=hyper_param['batch_size'], 
                      epochs=hyper_param['epochs'], 
                      validation_data=[devX_list_arrayS, devY_array], 
                      callbacks=[log, tb, checkpoint, es], 
                      verbose=1)





