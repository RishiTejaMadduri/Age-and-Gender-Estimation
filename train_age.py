#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("/home/rishi/Projects/Matroid/Rishi_Challenge/Data")
import data_loader
import argparse
from vgg import*
import datetime
import time

import warnings
warnings.filterwarnings("ignore")

# In[2]:


face_dataset='/home/rishi/Projects/Matroid/Rishi_Challenge/Data/adience_data_preprocessing/faces/faces_dataset.h5'
vgg_weights='/home/rishi/Projects/Matroid/Rishi_Challenge/Data/vgg_data_preprocessing/pkl_weights'


# In[3]:


# # def main():
    
# parser = argparse.ArgumentParser(description="Train Age and Gender Estimation")
# parser.add_argument("--audience_dataset", type=str, default=face_dataset)
# parser.add_argument("--folder_to_test", default=1)
# #Model Hyperparameters
# parser.add_argument("--dropout_keep_prob", type=int, default=0.5, help="Dropout keep probability")
# parser.add_argument("--weight_decay", type=int, default=1e-3, help="Weight decay rate for L2 regularization")
# # Training Parameters
# parser.add_argument("--learning rate", type=int, default=1e-2, help="Starter Learning Rate")
# parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
# parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
# parser.add_argument("--evaluate_every", type=int, default=2, help="Evaluate model on dev set after this many steps (default: 50)")
# parser.add_argument("--moving_average", type=bool, default=True, help="Enable usage of Exponential Moving Average")

# args = parser.parse_args()
# # try:
# #     args = parser.parse_args()
# #     print(args)
# # except:
# #     return 1


# In[4]:


#Loading Data
train_data, train_label, test_data, test_label, bgr_mean=data_loader.load_dataset(face_dataset, 1)
bgr_mean = [round(x, 4) for x in bgr_mean]


# In[5]:


acc_list = [0]
loss_train_list = [0]
loss_test_list = [0]

sess = tf.Session()


# In[6]:


bgr_mean=[93.5940, 104.7624, 129.1863]
cnn = VGGFace(bgr_mean, vgg_weights, num_classes=8, weight_decay=5e-4)
vgg_known_acc_max = [0.65, 0.51, 0.59, 0.49, 0.59]


# In[19]:


learning_rate=1e-2
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
lr_decay_fn = lambda lr, global_step : tf.train.exponential_decay(lr, global_step, 100, 0.95, staircase=True)
train_op = tf.contrib.layers.optimize_loss(loss=cnn.loss, global_step=global_step, clip_gradients=4.0, learning_rate=learning_rate, optimizer=lambda lr: optimizer, learning_rate_decay_fn=lr_decay_fn)


# In[20]:


timestamp = str(int(time.time()))
out_dir = os.path.join(os.path.expanduser('~'), 'volume', "runs", timestamp)
print("Writing to {}\n".format(out_dir))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


# In[21]:


sess.run(tf.global_variables_initializer())


# In[22]:


dropout_keep_prob=0.5
# Train Step and Test Step
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: dropout_keep_prob}
    _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Step {}, Loss {:g}, Acc {:g}".format(time_str, step, loss, accuracy))

def test_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
    loss, preds = sess.run([cnn.loss, cnn.predictions], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return preds, loss


# In[23]:


batch_size=32
num_epochs=75
train_batches = data_loader.batch_iter(list(zip(train_data, train_label)), batch_size, num_epochs)


# In[ ]:


folder_to_test=1
for train_batch in train_batches:
    x_batch, y_batch = zip(*train_batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    evaluate_every=30
    # Testing loop
    if current_step % evaluate_every == 0:
        print("\nEvaluation:")
        i = 0
        index = 0
        sum_loss = 0
        test_batches = data_loader.batch_iter(list(zip(test_data, test_label)), batch_size, 1)
        y_preds = np.ones(shape=len(test_label), dtype=np.int)
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            preds, test_loss = test_step(x_test_batch, y_test_batch)
            sum_loss += test_loss
            res = np.absolute(preds - np.argmax(y_test_batch, axis=1))
            y_preds[index:index+len(res)] = res
            i += 1
            index += len(res)

        time_str = datetime.datetime.now().isoformat()
        acc = np.count_nonzero(y_preds==0)/len(y_preds)
        acc_list.append(acc)
        print("{}: Evaluation Summary, Loss {:g}, Acc {:g}".format(time_str, sum_loss/i, acc))
        print("{}: Current Max Acc {:g} with in Iteration {}".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*evaluate_every)))

        if max(acc_list) > vgg_known_acc_max[folder_to_test - 1]:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved current model checkpoint with max accuracy to {}\n".format(path))


# In[ ]:




