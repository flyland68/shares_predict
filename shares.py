#!/bin/env python

import sys
import tensorflow as tf

import dataset


LSTM_NUM_HIDDEN=16
DNN_HIDDEN=32 
EMBEDDING_SIZE = 16

HISTORY_LENGTH=30
DAILY_FEATURE_SIZE=6
DATE_FEATURE_VOCABS=['0', '1']
LABELS_VOCABS=['%s' % (i-10) for i in range(21)]

def print_to_file(real_y, pred_y, y_one_, y_one, filename):
    with open(filename, 'w') as f:
        for i in range(len(real_y)):
            print >> f, '%s\t%s' % (real_y[i], pred_y[i])
            print >> f, ', '.join([str(value) for value in y_one_[i]])
            print >> f, ', '.join([str(value) for value in y_one[i]])

def _rnn(inputs, num_hidden):
    rnn_cell = tf.contrib.rnn.LSTMCell(num_hidden)
    output, _ = tf.nn.dynamic_rnn(
            rnn_cell,
            inputs,
            dtype=tf.float32)
    return output, num_hidden

def _one_hot(features, vocabs):
#    vocabs = tf.constant(vocabs)
    table = tf.contrib.lookup.index_table_from_tensor(vocabs, num_oov_buckets=1, default_value=-1)
    indices = table.lookup(features)
    return tf.one_hot(indices, len(vocabs))

def _lstm_attention_model(share_features, date_feature):
    ''' share_features: [batch, HISTORY_LENGTH, DAILY_FEATURE_SIZE] '''
    
    lstm_features = tf.reshape(share_features, [-1, DAILY_FEATURE_SIZE])
    ''' embedding_layer: [batch * HISTORY_LENGTH, DAILY_FEATURE_SIZE] '''
    
    embedding_layer = tf.layers.dense(inputs=lstm_features, units=EMBEDDING_SIZE, activation=tf.nn.relu)
    ''' embedding_layer: [batch * HISTORY_LENGTH, EMBEDDING_SIZE] '''
    
    embedding_layer = tf.reshape(embedding_layer, [-1, HISTORY_LENGTH, EMBEDDING_SIZE])
    ''' embedding_layer: [batch, HISTORY_LENGTH, EMBEDDING_SIZE] '''

    lstm_layer, feature_size = _rnn(embedding_layer, LSTM_NUM_HIDDEN)
    lstm_layer = tf.reshape(lstm_layer, [-1, HISTORY_LENGTH * LSTM_NUM_HIDDEN])

#    date_feature = _one_hot(date_feature, DATE_FEATURE_VOCABS)
#    dnn_input = tf.concat([lstm_layer, date_feature], axis=1)
    dnn_input = lstm_layer
    
#    dnn_input = tf.layers.dense(inputs=dnn_input, units=DNN_HIDDEN, activation=tf.nn.relu)
    dnn_feature = tf.layers.dense(inputs=dnn_input, units=len(LABELS_VOCABS), activation=tf.nn.tanh)
    return tf.nn.softmax(dnn_feature), lstm_layer, embedding_layer


def train(filename, test_filename):
    daily_infos = dataset.load_daily_infos(filename)
    history_share_inputs, date_feature_inputs, outputs = dataset.generate_train_set(daily_infos, HISTORY_LENGTH)

    test_daily_infos = dataset.load_daily_infos(test_filename)
    test_history_share_inputs, test_date_feature_inputs, test_outputs = dataset.generate_train_set(test_daily_infos, HISTORY_LENGTH)
 
    length = len(history_share_inputs)
    
    share_history_placeholder = tf.placeholder(tf.float32, [None, HISTORY_LENGTH, DAILY_FEATURE_SIZE])
    date_feature_placeholder = tf.placeholder(tf.string, [None])
    
    y_ = tf.placeholder(tf.string, [None])
    y_one_hot_ = _one_hot(y_, LABELS_VOCABS) 
    y_one_hot, watch_lstm, watch_embedding = _lstm_attention_model(share_history_placeholder, date_feature_placeholder)
    y = tf.argmax(y_one_hot, 1)
    y = y - tf.constant(10, dtype=tf.int64)
    
    correct_prediction = tf.equal(tf.cast(tf.string_to_number(y_, out_type=tf.int32), tf.int64), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot_, logits=y_one_hot))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    init_op = tf.group(
            tf.global_variables_initializer(), 
            tf.local_variables_initializer(), 
            tf.tables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        batch_size = 800
        for epoch in range(10000000):
            for i in range(length / batch_size + 1):
                start = i * batch_size
                end = i * batch_size + batch_size
                if start >= length:
                    break
                
                _loss, _, real_y, pred_y = sess.run([loss, train_step, y_, y],
                    feed_dict = {
                        share_history_placeholder: history_share_inputs[start:end], 
                        date_feature_placeholder: date_feature_inputs[start:end],
                        y_: outputs[start:end]
                    })
                    
            if epoch % 10 == 0:
                _loss, _accuracy, real_y, pred_y = sess.run([loss, accuracy, y_, y],
                feed_dict = {
                    share_history_placeholder: history_share_inputs, 
                    date_feature_placeholder: date_feature_inputs,
                    y_: outputs
                })
                
                print '---------------TRAIN--------------'
                print 'epoch:%s\tloss:%s\taccuracy:%s' % (epoch, _loss, _accuracy)
 
                _loss, _accuracy, real_y, pred_y, _y_one_hot_, _y_one_hot, _watch_lstm, _watch_embedding = sess.run([loss, accuracy, y_, y, y_one_hot_, y_one_hot, watch_lstm, watch_embedding],
                feed_dict = {
                    share_history_placeholder: test_history_share_inputs, 
                    date_feature_placeholder: test_date_feature_inputs,
                    y_: test_outputs
                })
                
                print '---------------TEST--------------'
                print 'epoch:%s\tloss:%s\taccuracy:%s' % (epoch, _loss, _accuracy)
#                print_to_file(real_y, pred_y, _y_one_hot_, _y_one_hot, 'result/result_%s' % epoch)
#                print _watch_lstm[0][-10:]
#                print _watch_lstm[1][-10:]
#                print _watch_embedding[0][-1][0:10]
#                print _watch_embedding[1][-1][0:10]
 




if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    train(train_file, test_file)
    
