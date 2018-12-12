from __future__ import unicode_literals, print_function
import os

def downloadData(url):
    from urllib.request import urlopen 
    import html2text
    import resource
    import content
    import resource
    
    #url='https://www.cs.stonybrook.edu/faq-page'
    page = urlopen(url)
    html_content = page.read()
    print(html_content)
    charset = resource.headers.get_content_charset()
    content = content.decode(charset)
    rendered_content = html2text.html2text(content)
    file = open('file_text.txt', 'w')
    file.write(rendered_content)
    file.close()
     
     
    import bs4
    import urllib.request
     
    webpage=str(urllib.request.urlopen("https://en.wikipedia.org/wiki/Stony_Brook_University").read())
    soup = bs4.BeautifulSoup(webpage)
     
    print(soup.get_text())

import wikipedia

def formatSentences(data):
    #sbu = sbu.content.replace("\n",'')
    
    #from spacy.en import English
    import spacy
    #nlp = English()
    nlp = spacy.load('en')
    doc = nlp(data)
    sentencesT = [sent.string.strip() for sent in doc.sents]
    
    sentences = []
    for ss in sentencesT:
        for s in ss.split('\n'):
            if len(s)>3:
                if s[0] == '=' or s[-1] == '=' or len(s.split(' ')) < 4:
                    pass
                else:
                    sentences.append(s)
        
    return sentences

def getWikiData(topic):
    sbu = wikipedia.page(topic)
    print(sbu.content)
    return sbu.content

def preprocessFile(inputFile="/Volumes/10/NLP/bert/sbu_small_pretrain_raw.txt",outputFile="/Volumes/10/NLP/bert/sbu_small_pretrain.txt"):
    f=open(inputFile, "r")
    if f.mode == 'r':
        contents =f.read()
        sentences = formatSentences(contents)
    with open(outputFile, 'w') as file:
        for line in sentences:
            file.write(line+"\n")
    print(outputFile+" writing done")
    
getWikiData("Stony Brook University")
#preprocessFile()

pretrainData = "\
python create_pretraining_data.py \
  --input_file=/Volumes/10/NLP/bert/sbu_small_pretrain.txt \
  --output_file=/Volumes/10/NLP/bert/sbu_small_pretrain.tfrecord \
  --vocab_file=/Volumes/10/NLP/bert/pretrainedBase/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  —max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5"

'''
python run_pretraining.py   
--input_file=/Volumes/10/NLP/bert/tf_examples.tfrecord   
--output_dir=/Volumes/10/NLP/bert/pretrainedSBU   
--do_train=True   
--do_eval=True   
--bert_config_file=/Volumes/10/NLP/bert/pretrainedBase/bert_config.json   
--init_checkpoint=/Volumes/10/NLP/bert/pretrainedBase/bert_model.ckpt   
--train_batch_size=32   
--max_seq_length=128   
--max_predictions_per_seq=20   
--num_train_steps=20   --num_warmup_steps=10   --learning_rate=2e-4
'''
pretranin = "python run_pretraining.py \
  --input_file=/Volumes/10/NLP/bert/sbu_small_pretrain.tfrecord \
  --output_dir=./temp/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=/Volumes/10/NLP/bert/pretrainedBase/bert_config.json \
  --init_checkpoint=/Volumes/10/NLP/bert/pretrainedBase/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-4"
  
#os.system(pretranin)