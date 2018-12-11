
# SOURCES
We’ve taken pre-trained embeddings from BERT model - https://github.com/google-research/bert           
Automatic Q/A generation from TheGadFlyProject - https://github.com/TheGadflyProject/TheGadflyProject                 
- REQUIREMENTS
You’ll need TensorFlow, Spacy, Python 3.6, nltk           
# INTRODUCTION
Due to high volume of data on the internet, it is becoming increasingly difficult to search for relevant information. Search engines can be  useful to find such information but sometimes it is difficult to answer question asked in natural language. Question-Answering systems for large datasets can be easily trained to give great results but training a deep neural network on a small dataset leads to poor results. One such example is a Question-Answering system for Stony Brook University. To develop QA system for such a scenario, we propose methods to learn weights from other large datasets and then fine-tune it using Stony Brook University website data. We built QA system for SBU using BERT base pre-trained weights. We used several other techniques to retrieve correct context paragraphs and evaluate our system. 

We took pre-trained weights of BERT base system and then fine-tuned it with SQuAD QA training data. To decide paragraph of answer we used multiway classification and infersent.

Most of our work is in 3 files - 
- doc_classifier.ipnb               
- fine_tuning_squad.ipynb                     
- fine_tunign_sbu_squad.ipynb                    
### doc_classifier.ipnb (Jupyter notebook)                                             
  For finding context of a given question we wrote the doc_retrieval module, it has jupyter notebook which generates json that            can be passed to the trained model for prediction
### fine_tuning_squad.ipynb (colab notebook)                                                                  
In the BERT module, we load the pre-trained embedding and run run_squad.py which was given with the BERT repository     
# SETUP
We use Google collaboratory to explore the BERT model and out experiments went well so we decided to fine tune our network on colab, so the it does not matter where you keep files locally, you need to upload them to the notebooks directory structure or upload it to your drive and then mount your drive. The folder BERT/bert_reqs contains all the requirements you'll need, so make sure you're using it correctly in colab. There are 2 options - 
- Upload file directly from local system
- Mount Google cloud on Colab and Upload files from Google cloud (RECOMMENDED)           
We recommend using cloud, because we experienced that it was faster and it's frustruating to browse for requirements everytime the runtime enviroment resets or notebook resets.
# FAQs                         
### How can I train using run_squad.py?                      
You can fine tune your own network with pre-trained embeddings on SQUAD using the following command - 
This command is also present in fine_tuning_squad.ipynb                   
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=train-v1.2.json \
  --do_predict=True \
  --predict_file='handmade_qa_sbu.json' \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR 
### How can I predict my own questions and contexts?   
You just need to generate json from doc_classifier.ipynb by passing the list of question and context to the create_json() methond and then run the fine tuned network for prediction.            
NOTE: Please note that output folder must not be empty and should contain checkpoint and data files           
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=train-v1.2.json \
  --do_predict=True \
  --predict_file='handmade_qa_sbu.json' \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR 
### How can I test on your handmade json file?  
You can test our system on fine-tuned network with the hand annotated dataset with the following command
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=train-v1.2.json \
  --do_predict=True \
  --predict_file='handmade_qa_sbu.json' \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR 


