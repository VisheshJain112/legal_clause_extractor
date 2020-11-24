# -*- coding: utf-8 -*-


"""

Import Statements & Requirements

""""
#Requirements
#!python -m spacy download en_core_web_lg
#!pip install python-docx
#!pip install bert-extractive-summarizer
#import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras
import nltk
import os
import docx
import pickle
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = stopwords.words('english')
from re import sub
from gensim.utils import simple_preprocess
import gensim.downloader as api
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import pandas as pd

!pip install spacy==2.1.3
!pip install transformers==2.2.2
!pip install neuralcoref

!python -m spacy download en_core_web_md
from summarizer import Summarizer
!pip install spacy==2.3.2

"""
Gives Base Agnostic Functions

"""
class BaseTask:
    """
      Reads The .docx and converts in All Text String

      Param : filepath
      Returns : All_text string

    """
    def read_the_document(self,filepath):
    
      doc = docx.Document(filepath)
      allText = []
      for docpara in doc.paragraphs:
        allText.append(docpara.text)
      doc = ' '.join(allText)
      return doc
    """
    Gives Corresponding Sentence from the doc as n-window-length from left and n-window-length from the right of the mathced query
    
    Param : 
            exact_query - query to be found
            whole_doc = doc
            window_length = length from rght and left
      
    Returns :
            Corresponding sentence
    """

    
    def get_corres_sent(self,exact_query, whole_doc, window_length = 40):
      start_idx = whole_doc.find(exact_query)

      corres_sent_start_idx = max(start_idx - window_length,0)  #max for dealing with edge case
      corres_sent_end_idx = min(start_idx + len(exact_query) + window_length, len(whole_doc)) #dealing with edge case
      corres_sent = whole_doc[corres_sent_start_idx : corres_sent_end_idx]
      return corres_sent

    """
      Lemmetaize the text
    
    """

    def getLemmText(self,text):
      tokens=word_tokenize(text)
      lemmatizer = WordNetLemmatizer()
      tokens=[lemmatizer.lemmatize(word) for word in tokens]
      return ' '.join(tokens)
    
    """
      Stem the Text
    
    """
    def getStemmText(self,text):
      tokens=word_tokenize(text)
      ps = PorterStemmer()
      tokens=[ps.stem(word) for word in tokens]
      return ' '.join(tokens)

    """
    Gives the context and label

    Params :
            *name_tuple = tuple of context,label
    Returns :
            context_tuple = tuple for context
            label_tuple = tuple for corres label
    """

    def get_context_and_label(self, name_tuples):
      context_tuple = []
      label_tuple = []
      for idx, name_tuple in enumerate(name_tuples):
        context,label = name_tuple
        context = self.getLemmText(context)
        context = self.getStemmText(context)
        context_tuple.append(context)
        label_tuple.append(label)
      return context_tuple, label_tuple

    """
    
    Creates the DataFrame with given column names

    """
    def create_the_dataframe(self):
      column_names = ['File Name', 'Aggrement Value', 'Aggrement Start Date',
            'Aggrement End Date', 'Renewal Notice (Days)', 'Party One',
            'Party Two']

      df = pd.DataFrame(columns = column_names)
      return df

    """
    Encoding as **kwargs to convert categorical tensor
    """

    def encoding(*args,**kwargs):
      for idx,label in enumerate(args):
        args[idx] = kwargs[label]

      args = to_categorical(args)
      return args.astype(np.int32)

    """

    Calculates max Glove Semenatic Similarity mathched String

    Params :
            query_string = Standard_string
            list_of_sentences = list of all Sentences 
    Returns :
            Index of max matched sentence
    """
    
    def get_the_glove_semenatic_similarity(self,query_string, list_of_sentences):
      query_string = query_string
      documents = list_of_sentences


      stopwords = ['the', 'and', 'are', 'a']


      def preprocess(doc):
          # Tokenize, clean up input document string
          doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
          doc = sub(r'<[^<>]+(>|$)', " ", doc)
          doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
          doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
          return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

      # Preprocess the documents, including the query string
      corpus = [preprocess(document) for document in documents]
      query = preprocess(query_string)
      # Load the model: this is a big file, can take a while to download and open
      
      similarity_index = WordEmbeddingSimilarityIndex(self.glove_model)

      # Build the term dictionary, TF-idf model
      dictionary = Dictionary(corpus+[query])
      tfidf = TfidfModel(dictionary=dictionary)

      # Create the term similarity matrix.  
      similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
      query_tf = tfidf[dictionary.doc2bow(query)]

      index = SoftCosineSimilarity(
                  tfidf[[dictionary.doc2bow(document) for document in corpus]],
                  similarity_matrix)

      doc_similarity_scores = index[query_tf]

      # Output the sorted similarity scores and documents
      sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
      for idx in sorted_indexes:
        return idx


  """
  Plots the Accuracy/Loss Graph

  """


    def graph_plots(history, string):
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()

"""

Classify by using BiDirectional LSTM which name goes to Party 1 and Party 2

"""

class NameExtractorModel(BaseTask):

    """
    Reads Trainig .csv and creates a named_tuple 
    Params : None
    Returns : 
            <*named_tuple> = [<*context>,<*label>]

    """
    def fetch_training_data(self):
      df = pd.read_csv('/content/TrainingTestSet.csv')
      self.name_tuples = []
      
      for index,row in df.iterrows():
        label_filename = row['File Name']

        list_of_class = ['Party One','Party Two']
        path = "/content/Training_data"
        try:
          filepath = (os.path.join(path,label_filename+".pdf"+".docx") )
          whole_doc = self.read_the_document(filepath)
          for label_class in list_of_class :
            
            exact_query = row[label_class]
            corres_sent = self.get_corres_sent(exact_query,whole_doc)
            self.name_tuples.append([corres_sent,(label_class.split())[1]])
        except :
          pass

     
    """
    Converts into seperate tuples
    Params : None
    Returns : 
            *context_tuple
            *label_tuple
    """    

    def pre_processing(self):
      self.context_tuple,self.label_tuple = self.get_context_and_label(self.name_tuples)
      
    
    """
    
    Divides into train and test split

    """

    def split_train_and_test(self,test_size = 0.33, random_state = 20):
      self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.context_tuple, 
                                                      self.label_tuple, 
                                                      test_size = test_size, 
                                                      random_state = random_state )
      
   """

   Set HyperParameters

   """
    
    def set_hyperparams(self,EMBEDDING_DIMENSION = 64, VOCABULARY_SIZE = 1000, MAX_LENGTH = 20, OOV_TOK = '<OOV>', TRUNCATE_TYPE = 'post', PADDING_TYPE = 'post'):
      self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
      self.VOCABULARY_SIZE = VOCABULARY_SIZE
      self.MAX_LENGTH = MAX_LENGTH
      self.OOV_TOK = '<OOV>'
      self.TRUNCATE_TYPE = 'post'
      self.PADDING_TYPE = 'post'

    """

    Generate Trainng, Validation Batches for Training and Tokenizer

    """

    def generate_batches(self):
      self.tokenizer = Tokenizer(num_words = self.VOCABULARY_SIZE, oov_token = self.OOV_TOK)
      self.tokenizer.fit_on_texts(list(self.xtrain) + list(self.xtest))
      self.xtrain_sequences = self.tokenizer.texts_to_sequences(self.xtrain)
      self.xtest_sequences = self.tokenizer.texts_to_sequences(self.xtest)
      self.word_index = self.tokenizer.word_index
      self.xtrain_pad = sequence.pad_sequences(self.xtrain_sequences, maxlen=self.MAX_LENGTH, padding=self.PADDING_TYPE, truncating=self.TRUNCATE_TYPE)
      self.xtest_pad = sequence.pad_sequences(self.xtest_sequences, maxlen=self.MAX_LENGTH, padding=self.PADDING_TYPE, truncating=self.TRUNCATE_TYPE)
      self.label_tokenizer = Tokenizer()
      self.label_tokenizer.fit_on_texts(list(self.ytrain))
      self.training_label_seq = np.array(self.label_tokenizer.texts_to_sequences(self.ytrain))
      self.test_label_seq = np.array(self.label_tokenizer.texts_to_sequences(self.ytest))
      self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])


    """
    
    BiDirectional LSTM Model 
    
    """  

    def get_model(self):
      self.model = Sequential()
      self.model.add(Embedding(len(self.word_index) + 1,
                          self.EMBEDDING_DIMENSION))
      self.model.add(SpatialDropout1D(0.3))
      self.model.add(Bidirectional(LSTM(self.EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
      self.model.add(Dense(self.EMBEDDING_DIMENSION, activation='relu'))
      self.model.add(Dropout(0.8))
      self.model.add(Dense(self.EMBEDDING_DIMENSION, activation='relu'))
      self.model.add(Dropout(0.8))
      self.model.add(Dense(2))
      self.model.add(Activation('softmax'))
      self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
      
      
    """
    
    Trainng The Model
    Params : epochs
    Returns : History <To visualize and evalucate perfomance>
    
    """

    def train_the_model(self, num_epochs=10):
      num_epochs = num_epochs
      print("--------------------TRAINIG NAME EXTRACTOR MODEL--------------------------")
      history = self.model.fit(self.xtrain_pad, self.training_label_seq, epochs=num_epochs, validation_data=(self.xtest_pad, self.test_label_seq), verbose=0)
      print("--------------------NAME EXTRACTOR MODEL INTIALIZED----------------------")
      return history
    
    """
    
    Saves the model and Tokenizer
    
    """

    def save_the_model_and_tokenizer(self):
      with open('name_tokenizer.pickle', 'wb') as handle:
        pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
      self.model.save('name_model')

      
    """

    View QQ Plots for Acuuracy and Loss of our Model > Also Helps to Visualize Overfitting

    """

    def display_accuracy_and_loss(history):
      graph_plots(history, "accuracy")
      graph_plots(history, "loss")

    """

    On a Given Text Predicts Which Class it is More Likely to Belong

    """

    def get_prediction(self,text):
      
      with open('name_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
      
      model = keras.models.load_model('name_model')
      phrase = text
      tokens = tokenizer.texts_to_matrix([phrase])
      predictions = list(model.predict(np.array(tokens)))
      class_label = predictions.index(max(predictions))
      if class_label == 0:
        return 'Party One'
      elif class_label == 1:
        return 'Party Two'
      else:
        return None

"""

Classify by using BiDirectional LSTM which amount goes to Agrrement Value or None

"""
class AmountExtractorModel(BaseTask):
    """
    
    Reads Trainig .csv and creates a named_tuple 
    Params : None
    Returns : 
            <*named_tuple> = [<*context>,<*label>]

    
    """    
    
    def fetch_training_data(self):
      df = pd.read_csv('/content/TrainingTestSet.csv')
      self.name_tuples = []
      
      for index,row in df.iterrows():
        label_filename = row['File Name']

        list_of_class = ['Aggrement Value']
        path = "/content/Training_data"
        try:
          filepath = (os.path.join(path,label_filename+".pdf"+".docx") )
          whole_doc = self.read_the_document(filepath)
          for label_class in list_of_class :
            
            exact_query = str(int(row[label_class]))
            corres_sent = self.get_corres_sent(exact_query,whole_doc)
            
            self.name_tuples.append([corres_sent,(label_class.split())[1]])
            
        except :
          pass

     
        
    """
    
    Converts into seperate tuples
    Params : None
    Returns : 
            *context_tuple
            *label_tuple
    """    

    def pre_processing(self):
      self.context_tuple,self.label_tuple = self.get_context_and_label(self.name_tuples)
    
      
    
    """
    
    Splits in Training and Testing
    
    """  


    def split_train_and_test(self,test_size = 0.33, random_state = 20):
      self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.context_tuple, 
                                                      self.label_tuple, 
                                                      test_size = test_size, 
                                                      random_state = random_state )
      

    """
    
    Set HyperParameters

    """
    def set_hyperparams(self,EMBEDDING_DIMENSION = 64, VOCABULARY_SIZE = 1000, MAX_LENGTH = 20, OOV_TOK = '<OOV>', TRUNCATE_TYPE = 'post', PADDING_TYPE = 'post'):
      self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
      self.VOCABULARY_SIZE = VOCABULARY_SIZE
      self.MAX_LENGTH = MAX_LENGTH
      self.OOV_TOK = '<OOV>'
      self.TRUNCATE_TYPE = 'post'
      self.PADDING_TYPE = 'post'

    """

    Generate Training and Testing Batches

    """
    def generate_batches(self):
      self.tokenizer = Tokenizer(num_words = self.VOCABULARY_SIZE, oov_token = self.OOV_TOK)
      self.tokenizer.fit_on_texts(list(self.xtrain) + list(self.xtest))
      self.xtrain_sequences = self.tokenizer.texts_to_sequences(self.xtrain)
      self.xtest_sequences = self.tokenizer.texts_to_sequences(self.xtest)
      self.word_index = self.tokenizer.word_index
      self.xtrain_pad = sequence.pad_sequences(self.xtrain_sequences, maxlen=self.MAX_LENGTH, padding=self.PADDING_TYPE, truncating=self.TRUNCATE_TYPE)
      self.xtest_pad = sequence.pad_sequences(self.xtest_sequences, maxlen=self.MAX_LENGTH, padding=self.PADDING_TYPE, truncating=self.TRUNCATE_TYPE)
      self.label_tokenizer = Tokenizer()
      self.label_tokenizer.fit_on_texts(list(self.ytrain))
      self.training_label_seq = np.array(self.label_tokenizer.texts_to_sequences(self.ytrain))
      self.test_label_seq = np.array(self.label_tokenizer.texts_to_sequences(self.ytest))
      self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

    """

    BiDirectional LSTM Model

    """  

    def get_model(self):
      self.model = Sequential()
      self.model.add(Embedding(len(self.word_index) + 1,
                          self.EMBEDDING_DIMENSION))
      self.model.add(SpatialDropout1D(0.3))
      self.model.add(Bidirectional(LSTM(self.EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
      self.model.add(Dense(self.EMBEDDING_DIMENSION, activation='relu'))
      self.model.add(Dropout(0.8))
      self.model.add(Dense(self.EMBEDDING_DIMENSION, activation='relu'))
      self.model.add(Dropout(0.8))
      self.model.add(Dense(1))
      self.model.add(Activation('sigmoid'))
      self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
      
      
    """
    
    Trainng The Model
    Params : epochs
    Returns : History <To visualize and evalucate perfomance>
    
    """

    def train_the_model(self, num_epochs=10):
      num_epochs = num_epochs
      print("-----------------TRAINING AMOUNT EXTRACTOR MODEL--------------------------------")
      history = self.model.fit(self.xtrain_pad, self.training_label_seq, epochs=num_epochs, validation_data=(self.xtest_pad, self.test_label_seq), verbose=0)
      print("----------------AMOUNT EXTRACTOR MODEL INTIALIZED-------------------------------")
      return history
    

    """
    
    Saves the model and Tokenizer
    
    """    
    def save_the_model_and_tokenizer(self):
      with open('amount_tokenizer.pickle', 'wb') as handle:
        pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
      self.model.save('amount_model')

    
    """

    View QQ Plots for Acuuracy and Loss of our Model > Also Helps to Visualize Overfitting

    """

    def display_accuracy_and_loss(history):
      graph_plots(history, "accuracy")
      graph_plots(history, "loss")

    
    """

    On a Given Text Predicts Which Class it is More Likely to Belong

    """
    
    def get_prediction(self,text):
      
      with open('amount_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
      
      model = keras.models.load_model('amount_model')
      phrase = text
      tokens = tokenizer.texts_to_matrix([phrase])
      predictions = list(model.predict(np.array(tokens)))
      class_label = predictions.index(max(predictions))
      if class_label == 0:
        return 'Aggrement Value'
     
      else:
        return None

"""

Extracts Date and Classifies Them into Multi-Labeled Classes

"""

class DateExtractorModel(BaseTask):
    def __init__(self,doc):
      #Intialize Spacy Model 
      self.spacy_nlp = spacy.load('en_core_web_lg')
      #Intialize BERT Model
      self.bert_model = Summarizer()
      print("--------------------------BERT MODEL INTIALIZED----------------------")
      #Intialize Glove Model
      self.glove_model = api.load("glove-wiki-gigaword-50")
      print("--------------------------GLOVE MODEL INTIALIZED----------------------")
    
      self.doc = doc

      
      self.BERT_processing
      self.Spacy_processing
    
    """
    
    Loads The Doc into Spacy Cluster
    
    """
    def Spacy_processing(self):
      self.doc = self.spacy_nlp(self.doc)
    
    """

    Uses BERT Summazrization Power To Lessen The DOC Dimension and keep the relevant Information
    
    """
    
    def BERT_processing(self):
      
      result = self.bert_model(self.doc, min_length=20)
      full = ''.join(result)
      full = full.replace("(", " ")
      self.doc = full

  
    """
    
    Uses Spacy NER
    Givea us a Standard Tuple based on entity Query

    Params :
            entity
    Returns :
            <*list_of_ner> = (Sentence, actual_label)
    
    """
    def get_the_standard_tuple(self, entity):
      list_of_ner = []
      for sent in self.doc.sents:
        for token in sent.ents:
          if token.label_ == entity:
            no_num = True
            text = token.text
            text = text.replace('.',' ')
            
            for nest_token in nlp(text):
              if nest_token.pos_== "NUM":
                no_num = False
                break
            if no_num:
              pass
            else:
              list_of_ner.append((sent.text,text))

        return list_of_ner

    """

    Gives us the List of Probable Sentences Using NER

    """

    def get_the_probable_sent(self, list_of_ner):
      list_of_probable_sentences = [str(sent) for sent,value in list_of_ner]
      return list_of_probable_sentences

    """

    Uses Glove Semenatic Similarity to Give us the Most probable Match

    """
    
    def get_the_corres_value(self, query_string , list_of_ner):
      list_of_probable_sentences = self.get_the_probable_sent(list_of_ner)
      try:
        corres_idx = self.get_the_glove_semenatic_similarity(query_string,
                                                      list_of_probable_sentences)
      
        corres_sent,corres_value = list_of_probable_sentences[corres_idx]
        return corres_value
      except:
        pass

    """

    Gets Event Data
    Params :
            start_standard_query : Query on which we Generalise our semetic match
            entity : Entity for the Query
    Returns:
          Corresponding Most Matched Sentence
    
    """
    def get_event_data(self, start_standard_query, entity):
      list_of_ner = self.get_the_standard_tuple(entity)
      start_standard_query = start_standard_query
      return self.get_the_corres_value(start_standard_query , list_of_ner)

    """

    Gets Aggreement_Start_Date

    """
    def get_aggrement_start_date(self):
      query = 'Rental Agreement made on  Executed on wef  Mr' 
      entity = "DATE"
      return self.get_event_data(query, entity)
    """
    
    Gets Aggrement End Date
    
    """
    
    def get_aggrement_end_date(self):
      query = 'This Rental Agreement Duration for only months from this date of Agreement. if both the parties mutual understanding'
      entity = "DATE"
      return self.get_event_data(query, entity)

    """

    Gets Renewal Notice Days

    """
    def get_renewal_notice_days(self):
      query = 'The Tenancy can be terminated by either party by giving Month notice to the opposite party'
      entity = "DATE"
      return self.get_event_data(query, entity)

"""

Intializing and Training The Model Class

"""

class IntializeModels:
  #Intialize Name Extractor Model
  def Intialize_Name_Extractor_Model(self,name_extractor_model):
    self.name_extractor = name_extractor_model
    self.name_extractor.fetch_training_data()
    self.name_extractor.pre_processing()
    self.name_extractor.split_train_and_test()
    self.name_extractor.set_hyperparams()
    self.name_extractor.generate_batches()
    self.name_extractor.get_model()
    self.name_extractor.train_the_model(num_epochs=6)
    self.name_extractor.save_the_model_and_tokenizer()
  #Intialize Amount Extractor Model
  def Intialize_Amount_Extractor_Model(self, amount_extractor_model):
    self.amount_extractor = amount_extractor_model
    self.amount_extractor.fetch_training_data()
    self.amount_extractor.pre_processing()
    self.amount_extractor.split_train_and_test()
    self.amount_extractor.set_hyperparams()
    self.amount_extractor.generate_batches()
    self.amount_extractor.get_model()
    self.amount_extractor.train_the_model(num_epochs=2)
    self.amount_extractor.save_the_model_and_tokenizer()

"""

Main Class Inherits teh obj and attribites 

"""

class MainTask:
  def __init__(self):
    self.basetask = BaseTask()

  """
  
  Intialize All The Models
  
  """
  def Intialize_Models(self):
  
    print("------------------INTIALIZING SPACY MODEL----------------")
    self.nlp_spacy = spacy.load('en_core_web_lg')
    print("--------------------SPACY MODEL INTIALIZED----------")

    print("--------------------INTIALIZING NAME EXTRACTOR MODEL--------------------")
    intialize_models = IntializeModels()
    name_extractor_model = NameExtractorModel()
    intialize_models.Intialize_Name_Extractor_Model(name_extractor_model)
    print("--------------------INTIALIZING AMOUNT EXTRACTOR MODEL--------------------")
    amount_extractor_model = AmountExtractorModel()
    intialize_models.Intialize_Amount_Extractor_Model(amount_extractor_model)
    print("--------------------INTIALIZING DATE EXTRACTOR MODEL--------------------")
    

  """

  Excecute Function : Gives The Final Output
  Params:
        doc_filepath : Document Fliepath
        output_filepath : Output .csv Filepath
  Returns:
        .csv with extracted Meta Data at output_filepath
  
  """
  def execute(self,doc_filepath, output_filepath = 'extracted_metadata.csv'):
    #Checks the Filepath
    if not os.path.exists(doc_filepath):
      raise IOException('DOCX File does not exist: %s' % doc_filepath)
    
    if not os.path.exists(output_filepath):
      print("No .csv file found at %s " % output_filepath)
      print("Creating new output csv")
      output_filepath = 'extracted_metadata.csv'
      #Creates a DataFrame
      self.extracted_metadata_df = self.basetask.create_the_dataframe()
      self.extracted_metadata_df.to_csv(output_filepath)

    
    self.extracted_metadata_df = pd.read_csv(output_filepath)
    doc = self.nlp_spacy(self.basetask.read_the_document(doc_filepath))
    self.metadata_extractor = MetaDataExtractor(doc)
    meta_json = self.metadata_extractor.get_meta_json()
    meta_json['File Name'] = doc_filepath
    extracted_metadata_df = pd.DataFrame.from_dict(meta_json)
    self.extracted_metadata_df = self.extracted_metadata_df.append(extracted_metadata_df)

"""

Extracts MetaData Inherits Predictive Stage From Each Trained Model Gives the extracted meta data

"""


class MetaDataExtractor:
    def __init__(self, doc):
      #intialize meta_json
      self.meta_json = {  
                          "File Name" : '',
                          "Aggrement Value" : '',
                          "Aggrement Start Date" : '',
                          "Aggrement End Date" : '',
                          "Renewal Notice (Days)" : '',
                          "Party One" : '',
                          "Party Two" : ''
                      }

      #intialize all the models and create a obj
      self.amount_extractor = AmountExtractorModel()
      self.name_extractor = NameExtractorModel()
      self.doc = doc
      self.date_extractor = DateExtractorModel(self.doc)
    

    """
    
    Gives the class and corres Value
    
    """
    def get_name(self):
      for sent in self.doc.sents:
        for token in sent.ents:
          if token.label_ == "PERSON":
            self.meta_json[self.name_extractor.get_prediction(sent.text)] = token.text

    """
    
    Gives the class and corres Value
    
    """

    def get_amount(self):
      for sent in self.doc.sents:
        for token in sent:
          if token.pos_ == "NUM":
            self.meta_json[self.amount_extractor.get_prediction(sent.text)] = token.text

    """

    Gives the class and corres Value

    """
    def get_date(self):
      self.meta_json['Aggrement Start Date'] = self.date_extractor_beta.get_aggrement_start_date()
      self.meta_json['Aggrement End Date'] = self.date_extractor_beta.get_aggrement_end_date()
      self.meta_json['Renewal Notice (Days)'] = self.date_extractor_beta.get_renewal_notice_days()

    """
    
    Calling Function
    Gives a Completed Output Meta Data Extracted

    """
    def get_meta_json(self):

      self.get_date_beta()
      
      self.get_amount()
      
      self.get_name()
      
      return self.meta_json

#Execute Statements
main = MainTask()
main.Intialize_Models()
main.execute(filepath)



