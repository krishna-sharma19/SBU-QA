We’ve taken pre-trained embeddings from BERT model - https://github.com/google-research/bert           
Automatic Q/A generation from TheGadFlyProject - https://github.com/TheGadflyProject/TheGadflyProject                 
You’ll need TensorFlow, Spacy, Python 3.6, nltk
Most of our work is in 3 files - 
- doc_classifier.ipnb
- fine_tuning_squad.ipynb
- fine_tunign_sbu_squad.ipynb
- doc_classifier.ipnb (Jupyter notebook)                           
  For finding context of a given question we wrote the doc_retrieval module, it has jupyter notebook which generates json that            can be passed to the trained model for prediction
- fine_tuning_squad.ipynb (colab notebook)                                                   
In the BERT module, we load the pre-trained embedding and run run_squad.py which was given with the BERT repository     
- FAQs
How can I train using run_squad.py?
\item The list of files that you modified and the specific functions within each file you modified for your project. 
\item A list of commands that provide how you train and test your baseline and the systems you built. 
\item A list of the major software requirements that are needed to run your system. (E.g. Tensorflow 2.3, Python 243.12, CUDA abd2.0, nltk-2401.11, allen-nlp 5.0). 
