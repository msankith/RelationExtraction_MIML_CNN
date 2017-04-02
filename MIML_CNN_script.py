'''
Created on 05-Nov-2016

@author: suvadeep
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf

from MIML_CNN import MIML_CNN
from Input_Data import Data_Sets
import math


    

    
def do_eval(self, 
            sess, 
            eval_correct,
            sentence_placeholder,
            entity_points_placeholder,
            label_placeholder,
            keep_prob_placeholder,
            data_set):
    tp_count = tn_count = fp_count = fn_count = 0
    total_count = 0
    while total_count < data_set.num_examples:
        sentence, entity_points, label = data_set.next_batch(self.batch_size)
        (tp, tn, fp, fn) = sess.run(eval_correct, 
                                    feed_dict={sentence_placeholder: sentence,
                                               entity_points_placeholder: entity_points,
                                               label_placeholder: label,
                                               keep_prob_placeholder: 1})
        tp_count += tp
        tn_count += tn
        fp_count += fp
        fn_count += fn
        
    return (tp_count, tn_count, fp_count, fn_count)
    
    
def run_training(data_sets, params):
    num_epoch = params["num_epoch"]
    res_dir = params["result_dir"]
    batch_size = params["batch_size"]
    keep_prob = params["keep_prob"]
    bag_size = params["bag_size"]
    sentence_length = params["sentence_length"]
    word_dimension = params["word_dimension"]
    relation_count = params["relation_count"]
        
    # Save params into the res_dir
    dir = os.path.join(os.getcwd(), res_dir)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    with open(os.path.join(dir, 'params.txt'), 'w') as f:
        f.write(str(params))
        
    miml_cnn = MIML_CNN(params)
            
    graph = tf.Graph()

    with graph.as_default():
        # Generate place holder
        sentence_placeholder = tf.placeholder(tf.float32,shape=[batch_size, bag_size, sentence_length, word_dimension])
        entity_points_placeholder = tf.placeholder(tf.int32,shape=[batch_size, bag_size, 2])
        label_placeholder = tf.placeholder(tf.float32,shape=[batch_size, relation_count])
        keep_prob_placeholder = tf.placeholder(tf.float32)
        
        # Build graph
        logits = miml_cnn.inference(sentence_placeholder, entity_points_placeholder, keep_prob_placeholder)
        loss = miml_cnn.loss(logits, label_placeholder)
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = miml_cnn.training(loss)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = miml_cnn.evaluation(logits, label_placeholder)

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        with tf.Session(graph=graph) as sess:
            # Run the Op to initialize the variables.
            sess.run(init)

            for epoch in range(num_epoch):
                ep = data_sets.train.epochs_completed
                while ep == data_sets.train.epochs_completed:
                    batch_of_sentences, batch_of_labels, batch_of_entity_points = \
                                                        data_sets.train.next_batch(batch_size)
                    feed_dict={sentence_placeholder: batch_of_sentences,
                               label_placeholder: batch_of_labels,
                               entity_points_placeholder: batch_of_entity_points,
                               keep_prob_placeholder: keep_prob}
                    _, current_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            
            
                '''
                #print('Training Data Eval:')
                (tr_a, tr_p, tr_r, tr_f) = self.do_eval(sess,
                                                        eval_correct,
                                                        sentence_placeholder,
                                                        entity_points_placeholder,
                                                        label_placeholder,
                                                        keep_prob_placeholder,
                                                        data_sets.train)

                # Evaluate against the validation set.
                #print('Validation Data Eval:')
                (va_a, va_p, va_r, va_f) = self.do_eval(sess,
                                                        eval_correct,
                                                        sentence_placeholder,
                                                        entity_points_placeholder,
                                                        label_placeholder,
                                                        keep_prob_placeholder,
                                                        data_sets.validation)

                # Evaluate against the test set.
                #print('Test Data Eval:')
                (te_a, te_p, te_r, te_f) = self.do_eval(sess,
                                                        eval_correct,
                                                        sentence_placeholder,
                                                        entity_points_placeholder,
                                                        label_placeholder,
                                                        keep_prob_placeholder,
                                                        data_sets.test)
                '''


    return
    

params = {}
params["word_dimension"] = 50
params["pos_dimension"] = 5
params["senetence_length"] = 88
params["bag_size"] = 78

params["conv_window"] = 3
params["num_filters"] = 230
params["relations_count"] = 16
params["batch_size"] = 50
params["loss_function"] = 'sigmoid' # can be one of 'sigmoid' or 'l2'
params["max_pos"] = 2 * params["senetence_length"]
params["keep_prob"] = 0.5
        
params["rho"] = 0.95
params["epsilon"] = math.exp(-6)

params["num_epoch"] = 1
params["result_dir"] = 'result'
    
datasets = Datasets('dataset_name')
run_training(data_sets, params)