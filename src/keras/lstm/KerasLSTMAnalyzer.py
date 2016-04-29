from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout, Activation

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

import sys
sys.path.append('../../../../')
from Util.util.data.DataPrep import *


import theano
import theano.tensor as T

from NN_Analyzer.src.keras.common.DrawWeights import *
from LSTM.src.WordEmbeddingLayer import *




class KerasLSTMAnalyzer(object):
    def build_model_for_SentimentAnalysis(self,loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']):
        print('Loading data...')

        X_train, y_train, self.word_to_index, self.index_to_word, self.labels_count = DataPrep.load_one_hot_sentiment_data("../../../../LSTM/data/sentiment/trainsentence_and_label_binary.txt")
        X_test, y_test = DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../../../LSTM/data/sentiment/devsentence_and_label_binary.txt",self.word_to_index, self.index_to_word,self.labels_count)

        #X_train, y_train = WordEmbeddingLayer.load_embedded_data(path="../../../../LSTM/data/",name="train",representation="glove.840B.300d")
        #X_test, y_test = WordEmbeddingLayer.load_embedded_data(path="../../../../LSTM/data/",name="dev",representation="glove.840B.300d")

        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        maxlen = 100 #max([max([len(X) for X in X_train]), max([len(X) for X in X_test])])

        print('Pad sequences (samples x time)')
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)


        self.model = Sequential()
        self.model.add(LSTM(input_dim=300,output_dim=300,input_length=maxlen,dropout_W=0., dropout_U=0.))
        self.model.add(Dense(3))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)



        draw_weights = DrawWeights(figsize=(4, 4), layer_id=1, \
    param_id=0, weight_slice=(slice(None), 0))

        self.model.fit(X_train, y_train, batch_size=32, nb_epoch=2,
          validation_data=(X_test, y_test),callbacks=[draw_weights])
        score, acc = self.model.evaluate(X_test, y_test,
                            batch_size=32)
        print('Test score:', score)
        print('Test accuracy:', acc)


    def build_forward_step_from_learned_weights(self,model):
        x = T.matrix('x').astype(theano.config.floatX)



        U_i = model.layers[1].W_i
        U_f = model.layers[1].W_f
        U_o = model.layers[1].W_o

        W_i = model.layers[1].U_i
        W_f = model.layers[1].U_f
        W_o = model.layers[1].U_o

        b_i = model.layers[1].b_i
        b_f = model.layers[1].b_f
        b_o = model.layers[1].b_o

        U = model.layers[1].U_c


        def forward_step(x_t, prev_state,prev_content):
            input_gate = T.nnet.hard_sigmoid(T.dot((U_i),x_t) + T.dot(W_i,prev_state) + b_i)
            forget_gate = T.nnet.hard_sigmoid(T.dot((U_f),x_t) + T.dot(W_f,prev_state)+ b_f)
            output_gate = T.nnet.hard_sigmoid(T.dot((U_o),x_t) + T.dot(W_o,prev_state)+ b_o)



            stabilized_input = T.tanh(T.dot((U),x_t) + T.dot(W,prev_state) + b)

            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            return [s,s,c,input_gate,forget_gate,output_gate]

        [self.output,hidden_state,memory_content,self.input_gate,self.forget_gate,self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[x],
            truncate_gradient=-1,
            outputs_info=[None,dict(initial=T.zeros(model.layers[1].output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(model.layers[1].output_dim,dtype=theano.config.floatX))
                          ,None,None,None
                          ])


        self.get_gates =theano.function([x],[model.layers[1].output,self.input_gate,self.forget_gate,self.output_gate])




if __name__ == '__main__':
    kl = KerasLSTMAnalyzer()
    kl.build_model_for_SentimentAnalysis()






