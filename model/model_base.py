# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/8/13
# File Name: model_base
# Edit Author: lnest
# ------------------------------------
import tensorflow as tf

import model.func as F


class Base:
    def __init__(self, args, batch_data):
        self.segged, self.word, self.data_id, self.label = batch_data
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.args = args
        self.loss = self.get_loss()

    def forword(self):
        with tf.variable_scope("emb"):
            with tf.variable_scope("word"):
                embeddings = tf.Variable(tf.random_uniform([self.args.word_size, self.args.word_hidden], -1.0, 1.0))
                word_emb = tf.nn.embedding_lookup(embeddings, self.word)
            with tf.variable_scope('segg'):
                embeddings_segg = tf.Variable(tf.random_uniform([self.args.segg_size, self.args.segg_hidden], -1.0, 1.0))
                segg_emb = tf.nn.embedding_lookup(embeddings_segg, self.segged)

            context_emb = tf.concat([word_emb, segg_emb], axis=2)
            # context_emb = F.highway(context_emb, 128, scope="highway")

        with tf.variable_scope('word_conv_pool'):
            word_conv_3 = F.conv(word_emb, 128, activation=tf.nn.relu, kernel_size=3, name='conv_3d')
            max_pool_3 = tf.
            word_conv_4 = F.conv(word_emb, 128, activation=tf.nn.relu, kernel_size=4, name='conv_4d')
            word_conv_5 = F.conv(word_emb, 128, activation=tf.nn.relu, kernel_size=5, name='conv_5d')


        with tf.variable_scope('seg_conv_pool'):
            seg_conv_1 = F.conv(word_emb, 256, activation=tf.nn.relu, kernel_size=1, name='conv_1d')
            seg_conv_1 = F.conv(word_emb, 256, activation=tf.nn.relu, kernel_size=2, name='conv_2d')


        with tf.variable_scope('output'):
            pass

    def get_loss(self):
        return ''

    def train(self):
        pass

    def test(self):
        pass
