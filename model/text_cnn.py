# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/6
# File Name: text_cnn
# Edit Author: lnest
# ------------------------------------

import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 char_embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.vocab_size = vocab_size
        self.embedding_size = char_embedding_size
        self.filter_size = filter_sizes
        self.num_filters = num_filters

    def forword(self, batch_data):
        input_x, input_y = batch_data
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="word_embeding")
            # self.W_1 = tf.Variable(
            #     tf.random_uniform([char_size, char_embedding_size], -1.0, 1.0),
            #     name='char_embeding')

            self.embedded_words = tf.nn.embedding_lookup(self.W, input_x)
            # self.embedded_chars = tf.nn.embedding_lookup(self.W_1, self.input_x)

            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_chars = conv_pool(self.embedded_chars_expanded, char_embedding_size, filter_sizes, num_filters, sequence_length)
        pooled_outputs_words = conv_pool(self.embedded_words_expanded, word_embedding_size, filter_sizes, num_filters, sequence_length)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")