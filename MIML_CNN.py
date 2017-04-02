from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils

 

class MIML_CNN:
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.word_dimension = params["word_dimension"]
        self.pos_dimension = params["pos_dimension"]
        self.sentence_length = params["senetence_length"]
        self.bag_size = params["bag_size"]

        self.conv_window = params["conv_window"]
        self.num_filters = params["num_filters"]
        self.relations_count = params["relations_count"]
        self.batch_size = params["batch_size"]
        self.loss_function = params["loss_function"]
        self.max_pos = params["max_pos"]
        
        self.rho = params["rho"]
        self.epsilon = params["epsilon"]


    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def max_pool1d(x, ksize):
        ksize.insert(1, 1)
        ksize.insert(1, 1)
        y = tf.expand_dims(x, 1)
        y = tf.expand_dims(x, 1)
        return tf.squeeze(tf.nn.max_pool(y, ksize=ksize, \
                                        strides=[1, 1, 1, 1], padding='VALID'))

    def conv1d(self, x, W):
        return tf.squeeze(tf.nn.conv2d(tf.expand_dims(x, 1), tf.expand_dims(W, 0), [1, 1, 1, 1], 'SAME'))


    def new_conv_layer(self, instance_features_array, entityPoints, num_filters):
        shape= [self.conv_window, self.word_dimension + self.pos_dimension, num_filters]
        
        weights = self.new_weights(shape = shape)
        biases = self.new_biases(length = num_filters)
    
        newInstances = tf.unstack(instance_features_array,axis=0)
        entityPoints = tf.unstack(entityPoints,axis=0)
    
        bagVectors=[]
        for i,bag in enumerate(newInstances):
            tempLayer = self.conv1d(bag, weights)
            tempLayer += biases
            tempLayer = tf.nn.tanh(tempLayer)
            instances = tf.unstack(tempLayer,axis=0)
            breakPoints = tf.unstack(entityPoints[i],axis=0)
        
            before = []
            middle = []
            after = []
        
            for j,instance in enumerate(instances):
                pieces=tf.split(instance,tf.stack([breakPoints[j][0],breakPoints[j][1]-breakPoints[j][0],self.sentence_length-breakPoints[j][1]]), 0)
                before.append(tf.reduce_max(pieces[0],axis=0))
                middle.append(tf.reduce_max(pieces[1],axis=0))
                after.append(tf.reduce_max(pieces[2],axis=0))
        
            bagVector = tf.concat([tf.reduce_max(tf.stack(before),axis=0),
                               tf.reduce_max(tf.stack(middle),axis=0),
                               tf.reduce_max(tf.stack(after),axis=0)],axis=0)
        
            bagVectors.append(bagVector)
        
        return  tf.stack(bagVectors)

        
        
    def inference(self, instance_features_array, entityPoints, left_word_pos_array, right_word_pos_array, keep_prob):
        with tf.name_scope('preprocess_layer'):
            left_pos_vectors = self.new_weights([self.max_pos, self.pos_dimension])
            right_pos_vectors = self.new_weights([self.max_pos, self.pos_dimension])
            left_pos_features = tf.gather(left_pos_vectors, left_word_pos_array)
            right_pos_features = tf.gather(right_pos_vectors, right_word_pos_array)
            instance_features_array = tf.concat([instance_features_array, left_pos_features, right_pos_features], axis=4)
        
        with tf.name_scope('conv_layer'):
            bagVectors = self.new_conv_layer(instance_features_array, entityPoints, self.num_filters)
            
        with tf.name_scope("dropout"):
            bagVectors_drop = tf.nn.dropout(bagVectors, keep_prob)

        # Compute softmax
        with tf.name_scope('linear_layer'):
            weights_sm = self.new_weights([3*self.num_filters, self.relations_count])
            biases_sm = utils.bias_variable([self.relations_count])
            logits = tf.matmul(bagVectors_drop, weights_sm) + biases_sm

        return logits


    def loss(self, logits, labels, func):
        if self.loss_function == 'sigmoid':
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        else: # == 'l2'
            prediction = tf.nn.softmax(logits, dim=-1, name=None)
            loss = tf.nn.l2_loss(prediction - labels)
        
        return loss


    def evaluation(self, logits, labels):
        return self.tf_confusion_metrics(logits, labels)
        

    def training(self, loss):
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        self.optimizer = tf.train.AdadeltaOptimizer(rho=self.rho, epsilon=self.epsilon)
        train_op = tf.train.AdadeltaOptimizer.minimize(loss, global_step=global_step)
        
        return train_op
