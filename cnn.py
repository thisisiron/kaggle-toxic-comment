import tensorflow as tf
import numpy as np




class TextCNN(object):
    """

    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters):

        # x_train.shape (159571, sequence_length)
        # y_train.shape (159571, 6)
        # vocab_processor.vocabulary_ 
        # embedding_size 
        # filter_sizes 
        # num_filters 
        # l2_reg_lambda 

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W shape (vocab_size, embedding_size)
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # embedded_chars shape = (?, sequence_length, embedding_size)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # embedded_chars_expanded shape = (?, sequence_length, embedding_size, 1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # filter shape [H, W, C, NUM]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
                b = tf.Variable(tf.truncated_normal(shape=[num_filters], stddev=0.05), name="b")
                # If padding equals VALID, the contents are discarded. But If padding equals SAME, the contents are appended by 0.
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print(h.shape)
                # Maxpooling over the outputs
                # 한 줄씩 MAX Pool을 한다.
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        # Use concat function to combine the entire contents
        # h_pool shape (?, 1, 1, 384)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("layer1"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, 100],
                initializer= tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.truncated_normal(shape=[100],stddev = 0.05), name="b1")
            layer1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="layer1")
            layer1 = tf.nn.relu(layer1)

        with tf.name_scope("layer2"):
            W2 = tf.get_variable(
                "W2",
                shape=[100, num_classes],
                initializer= tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.truncated_normal(shape=[num_classes],stddev = 0.05), name="b2")
            self.scores = tf.nn.xw_plus_b(layer1, W2, b2, name="scores")
            self.predictions = tf.nn.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.round(self.predictions), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
