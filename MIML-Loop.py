
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from numpy import zeros, newaxis
from sklearn.metrics import f1_score


# In[2]:

wordDimension=302
sentenceLength=88 #max words in sentence
Instances=78 #Channels
relations = 16

convWindow = 2#words at a time\
num_filters1 = 10
relationscount=16
batch_size=2


# In[95]:

wordLen = 302

relations = []
instances = []

f=open("data_50.txt")
miml={}
maxLen=sentenceLength
maxLenSent=sentenceLength
maxInstances=Instances
sent=''
dummy = np.zeros(shape=(wordDimension))
dummyTokens=['dummy']*maxLenSent
dummyFeatures=np.zeros(shape=(sentenceLength,wordDimension),dtype=float)
dummyPos=[4,8]

while True:
    line = f.readline()
    if not line: break
    
    del relations[:]
    del instances[:]
    featuresArray = []
    entityPos = []
    entity1 = int(line)
    entity2 = int(f.readline())
    relActive=int(f.readline())
    for i in range(relActive):
        rel = int(f.readline())
        relations.append(rel)
    instancesActive = int(f.readline())
    if instancesActive > Instances:
        print(str(entity1) + "\t" + str(entity2))
        continue
    for i in range(instancesActive):
        sentence = f.readline()
        lineEle = [float(x) for x in sentence.split()]
        sentenceVec = [lineEle[i:i+wordLen] for i in range(0, len(lineEle), wordLen)]
        zeros = [0.]*wordLen
        for j in range(sentenceLength - len(sentenceVec)):
            sentenceVec.append(zeros)
        featuresArray.append(sentenceVec)
        pos = f.readline().split()
        entityPos.append([int(pos[0]), int(pos[1])])
    for i in range(instancesActive,Instances):
        featuresArray.append(dummyFeatures)
        entityPos.append(dummyPos)
    featuresArray=np.array(featuresArray)
    entityPos = np.array(entityPos, dtype=np.int32)
    entityPos.sort()
    relTemp = [0]*relationscount
    for r in relations:
        relTemp[r-1]=1
    miml[str(entity1) + "\t" + str(entity2)]=(relTemp,featuresArray.astype(float), entityPos, instancesActive)
print("done")
entities = [l for l in miml]


# In[96]:

print(len(miml["5	6"][2]))
print (miml["5	6"][1].shape)
print (miml["5	6"][2].shape)
# print(dummyFeatures.shape)


# In[98]:

instance_features_array = tf.placeholder(tf.float32,shape=[batch_size,Instances,sentenceLength,wordDimension])
entityPointsPlaceholder = tf.placeholder(tf.int32,shape=[batch_size,Instances,2])
# placeHolder = np.array([np.array(xi) for xi in instance_features_array])
target = tf.placeholder(tf.float32,shape=[batch_size,relationscount])
# entityPointsPlaceholder=tf.placeholder(tf.int32,shape=[1,2])


# In[99]:

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def max_pool1d(x, ksize):
    ksize.insert(1, 1)
    ksize.insert(1, 1)
    y = tf.expand_dims(x, 1)
    y = tf.expand_dims(x, 1)
    return tf.squeeze(tf.nn.max_pool(y, ksize=ksize, \
                                        strides=[1, 1, 1, 1], padding='VALID'))

def conv1d(x, W):
    return tf.squeeze(tf.nn.conv2d(tf.expand_dims(x, 1), tf.expand_dims(W, 0), [1, 1, 1, 1], 'SAME'))

# In[198]:

def new_conv_layer(instance_features_array,entityPoints,numFilters):
    shape= [convWindow, wordDimension, numFilters]
    
    weights = new_weights(shape=shape)
    biases = new_biases(length=numFilters)
    
    newInstances = tf.unstack(instance_features_array,axis=0)
    entityPoints = tf.unstack(entityPoints,axis=0)
    
    bagVectors=[]
    for i,bag in enumerate(newInstances):
        tempLayer = conv1d(bag, weights)
        tempLayer += biases
        tempLayer = tf.nn.tanh(tempLayer)
        instances = tf.unstack(tempLayer,axis=0)
        breakPoints = tf.unstack(entityPoints[i],axis=0)
        
        before = []
        middle = []
        after = []
        
        for j,instance in enumerate(instances):
            pieces=tf.split(instance,tf.stack([breakPoints[j][0],breakPoints[j][1]-breakPoints[j][0],sentenceLength-breakPoints[j][1]]), 0)
            before.append(tf.reduce_max(pieces[0],axis=0))
            middle.append(tf.reduce_max(pieces[1],axis=0))
            after.append(tf.reduce_max(pieces[2],axis=0))
        
        bagVector = tf.concat([tf.reduce_max(tf.stack(before),axis=0),
                               tf.reduce_max(tf.stack(middle),axis=0),
                               tf.reduce_max(tf.stack(after),axis=0)],axis=0)
        
        bagVectors.append(bagVector)
        
    return  tf.stack(bagVectors)


# In[199]:

bagRepresentation=new_conv_layer(instance_features_array,entityPointsPlaceholder)





# In[103]:

relationBatch=[]
bagBatch=[]
itr=0
while itr < len(entities):
    bagBatch = np.zeros([batch_size,Instances,sentenceLength,wordDimension])
    entityPoints = np.zeros([batch_size,Instances,2])
    relationBatch = np.zeros([batch_size,relationscount])
    for batchItr in range(batch_size):
        tempData=miml[entities[itr+batchItr]]
        relationBatch[batchItr,:] = np.array(tempData[0])
        bagBatch[batchItr,:,:,:] = tempData[1]
        entityPoints[batchItr,:] = tempData[2]
    itr+=batch_size
    feed_dict_input={}
    feed_dict_input[entityPointsPlaceholder] = entityPoints
    feed_dict_input[instance_features_array] = bagBatch
    feed_dict_input[target] = relationBatch
    break



session = tf.Session()
session.run(tf.global_variables_initializer())
fe=session.run(bagRepresentation,feed_dict=feed_dict_input)


# In[23]:
'''
elements=0
while elements < len(entities):
    if miml[entities[elements]][3] < 4 or miml[entities[elements]][3]>10:
        elements+=1
        continue
    instancesCount=[0]*batch_size
    newData=0
    feed_dict_data={}
    feed_dict_data[instancesPlaceholder]=[]
    feed_dict_data[target]=[]
    feed_dict_data[entityPointsPlaceholder]=[]
    feed_dict_data[bagInstancesCount]=[]
    for itr in range(batch_size):
        pair=entities[elements+itr] 
        print (pair)
        if newData ==0:
            feed_dict_data[instancesPlaceholder]=miml[pair][1][:,:,:,newaxis]
            feed_dict_data[entityPointsPlaceholder] = miml[pair][2]
            instancesCount[itr]=miml[pair][3]
            feed_dict_data[bagInstancesCount] =instancesCount
            feed_dict_data[bagCount[itr]]=[miml[pair][3]]
#             feed_dict_data[bagInstancesCountArray[itr]]=instancesCount
            feed_dict_data[target]= [miml[pair][0]]
#                 print (miml[pair][0])
        else:
            tempDD=miml[pair][1][:,:,:,newaxis]
            feed_dict_data[instancesPlaceholder]=np.concatenate((feed_dict_data[instancesPlaceholder],tempDD))
            feed_dict_data[entityPointsPlaceholder]=np.concatenate((feed_dict_data[entityPointsPlaceholder],miml[pair][2]))
            instancesCount[itr]=miml[pair][3]
            feed_dict_data[bagInstancesCount] = instancesCount
            feed_dict_data[target].append(miml[pair][0])
            feed_dict_data[bagCount[itr]]=[miml[pair][3]]
        newData=1
    elements+=batch_size
    print(miml[pair][3])
    break
'''

