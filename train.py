import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import datetime
import data_helpers
from cnn import TextCNN
from tensorflow.contrib import learn

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3, 4, 5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


x_train, y_train, x_test, y_test = data_helpers.load_data_and_labels()


# max_document_length = max([len(x.split(" ")) for x in x_train])
max_document_length = 170   
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.fit_transform(x_test)))

# max_document_length 4948


shuffle_indices = np.random.permutation(np.arange(len(y_train))) 
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
# Vocabulary Size: 191309
# Train/Dev split: 159571/153164

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

            # x_train.shape
            # y_train.shape 
            # vocab_processor.vocabulary_ 
            # embedding_size 
            # filter_sizes 
            # num_filters 
        
        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(7e-4)
        train_op = optimizer.minimize(cnn.loss)

        # 우리가 알고 있는 아래와 같은 함수와 동일하다. 
        # optimizer = tf.train.AdamOptimizer(1e-3)
        # train_op = optimizer.minimize(cnn.loss, global_step=global_step)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, loss, accuracy = sess.run(
                [train_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


        def test_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # Dont apply dropout when you excute eval model.
            }
            predictions = sess.run(cnn.predictions, feed_dict)
            # loss, accuracy, predictions = sess.run(
            #     [cnn.loss, cnn.accuracy, cnn.predictions],
            #     feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
            return predictions
        
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for i, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            print(i,end=" ")
            train_step(x_batch, y_batch)
            # current_step = tf.train.global_step(sess, global_step)

        print('Training Finish!')
        df = pd.DataFrame()
        test_blocks = data_helpers.blocks(list(zip(x_test, y_test)), 1024)
        for i, block in enumerate(test_blocks):
            x_block, y_block = zip(*block)
            pred = test_step(x_block, y_block)
            df = df.append(pd.DataFrame(pred))
        
        print('Prediction Finish!')
        df.round().mean()

        submission = pd.read_csv('./data/sample_submission.csv')
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = np.array(df)
        submission.to_csv('submission.csv', index=False)
