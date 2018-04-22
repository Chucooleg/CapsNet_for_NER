# CapsNet_for_NER
Adapt Capsule Network for Name Entity Recognition Task  
<hr>  

## setup GCP gpu for tensorflow and keras
https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272

|Internal Link|About|
|--|--|
|[Team Project Proposal Link](https://docs.google.com/document/d/18QAYJCnR6R6I7ZAx1IiJ15jRZTj0gFlsQEk012ih-xU/edit?usp=sharing)|For submission|

|CapsNet Concepts|About|
|--|--|
|[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)|Sabour, Frosst, Hinton (2017)|
|[Transforming Auto-encoders](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf)|Hinton, Krizhevsky, Wang (2011)|
|[Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026)|Chiu and Nichols (2016)|
|[Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://www.aclweb.org/anthology/W03-0419)|Erik F. Tjong Kim SangandFien De Meulder, 2016|
|[Capsule Networks (CapsNets) – ](https://www.youtube.com/watch?v=pPN8d0E3900)|Video concept intro|

|CapsNet Implementation|About|
|--|--|
|[Tensorflow Implementation 1](https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb)|.ipynb based on Dynamic Routing between Capsules (MNIST)|
|[Video of above](https://www.youtube.com/watch?v=pPN8d0E3900&feature=youtu.be)|Video walkthrough (MNIST)|
|[How to implement Capsule Nets using Tensorflow](https://www.youtube.com/watch?v=2Kawrd5szHE)|Video walkthrough (MNIST)
|[CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)|repo Keras w/ TensorFlow backend (MNIST)
|[CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)|repo TensorFlow (MNIST)|
|[Dynamic Routing Between Capsules](https://github.com/gram-ai/capsule-networks)|repo PyTorch (MNIST)|
|[CapsNet for Natural Language Processing](https://gitlab.com/stefan-it/capsnet-nlp)|repo CapsNet for Sentiment Analysis|



|NER Dataset|About|
|--|--|
|[OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)|download, intro|
|[CoNLL-2003 Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)|download, intro|
|[Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://www.aclweb.org/anthology/W03-0419)|Introduction of the dataset|
|[CoNLL-2003 Shared Task](https://gist.github.com/JackNhat/0dc0b57b248df1b970a0d64475b31580)|CoNLL-2003 Benchmark papers|
|[OntoNotes coreference annotation and modeling](https://github.com/magizbox/underthesea/wiki/TASK-CONLL-2012)|OntoNotes Benchmark papers|
|[Named Entity Recognition: Exploring Features](http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf)|Explore faetures for NER task. Both CoNLL-2003 and OntoNotes version 4 are used.|

## Files
### /code
|File|Description|
|--|--|
|buildCapsModel.py| capsnet model implementation|
|buildCNNModel.py| cnn model implementation|
|capsulelayers.py| capsnet modules - slightly adapted from [Xifeng Guo](https://github.com/XifengGuo/CapsNet-Keras)'s implementation|
|evaluation_helper.py| helper functions for model evaluation|
|Examine_History.ipynb| code to plot and investigate model training history and dev results|
|error_analysis_demo.ipynb| notebook to generate a model evaluation report |
|error_analysis_testset.ipynb| notebook to generate model evaluation reports for test set |
|glove_helper.py| helper code for loading Glove embeddings|
|loadutils.py| helper functions for loading and storing models and the data set|
|model_testing.ipynb| demo code for best model and baseline test set performance |
|model_training_tmpl.ipynb| notebook to orchestrate and run model training sessions |
|trainCapsModel.py| interface code to train a capsnet model|
|trainCNNModel.py| interface code to train a CNN model|

### /common
> Helper code for building a vocabulary

### /data
> CoNLL-2003 data set


### /models
> final models and their training history
