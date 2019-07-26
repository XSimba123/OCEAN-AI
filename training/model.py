import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
LEARNING_RATE = 0.0005
AUX_DATA_DIGITS = 125

def merge_alex_net_model_fn(features, labels, mode):
    inputs = features['X1']
    imgs = features['X2']
    num_classes = 90
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_keep_prob = 0.5
    scope = 'alexnet'
    with tf.variable_scope(scope) as sc:
        net = tf.layers.conv2d(inputs,
                               filters=64,
                               kernel_size=(11, 11),
                               strides=4,
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv1')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool1')
        net = tf.layers.conv2d(net,
                               filters=192,
                               kernel_size=(5, 5),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv2')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool2')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv3')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv4')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv5')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc6')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout6')
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc7')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout7')

        _net = tf.layers.conv2d(imgs,
                                filters=64,
                                kernel_size=(11, 11),
                                strides=4,
                                activation=tf.nn.relu,
                                padding='same',
                                name='_conv1')
        _net = tf.layers.max_pooling2d(_net,
                                       pool_size=(3, 3),
                                       strides=2,
                                       padding='same',
                                       name='_pool1')
        _net = tf.layers.conv2d(_net,
                                filters=192,
                                kernel_size=(5, 5),
                                activation=tf.nn.relu,
                                padding='same',
                                name='_conv2')
        _net = tf.layers.max_pooling2d(_net,
                                       pool_size=(3, 3),
                                       strides=2,
                                       padding='same',
                                       name='_pool2')
        _net = tf.layers.conv2d(_net,
                                filters=384,
                                kernel_size=(3, 3),
                                activation=tf.nn.relu,
                                padding='same',
                                name='_conv3')
        _net = tf.layers.conv2d(_net,
                                filters=384,
                                kernel_size=(3, 3),
                                activation=tf.nn.relu,
                                padding='same',
                                name='_conv4')
        _net = tf.layers.conv2d(_net,
                                filters=256,
                                kernel_size=(3, 3),
                                activation=tf.nn.relu,
                                padding='same',
                                name='_conv5')
        _net = tf.layers.flatten(_net)
        _net = tf.layers.dense(_net,
                               units=4096,
                               activation=tf.nn.relu,
                               name='_fc6')
        _net = tf.layers.dropout(_net,
                                 rate=dropout_keep_prob,
                                 training=is_training,
                                 name='_dropout6')
        _net = tf.layers.dense(_net,
                               units=4096,
                               activation=tf.nn.relu,
                               name='_fc7')
        _net = tf.layers.dropout(_net,
                                 rate=dropout_keep_prob,
                                 training=is_training,
                                 name='_dropout7')

        concat_vec = tf.concat([net, _net], 1)
        logits = tf.layers.dense(concat_vec,
                                 units=num_classes,
                                 name='_fc8')
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = opt.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def alex_net_model_fn(features, labels, mode):
    inputs = features['X1']
    num_classes = 50
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_keep_prob = 0.5
    scope = 'alexnet'
    with tf.variable_scope(scope) as sc:
        net = tf.layers.conv2d(inputs,
                               filters=64,
                               kernel_size=(11, 11),
                               strides=4,
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv1')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool1')
        net = tf.layers.conv2d(net,
                               filters=192,
                               kernel_size=(5, 5),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv2')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool2')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv3')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv4')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv5')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc6')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout6')
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc7')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout7')
        logits = tf.layers.dense(net,
                                 units=num_classes,
                                 name='fc8')
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = opt.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def VGG_net_model_fn(features, labels, mode):
    inputs = features['X1']
    num_classes = 50
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_keep_prob = 0.5
    scope = 'VGGnet'
    with tf.variable_scope(scope) as sc:
        net = tf.layers.conv2d(inputs,
                               filters=64,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block1_conv1')
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block1_conv2')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=(2, 2),
                                      name='block1_pool')

        net = tf.layers.conv2d(net,
                               filters=128,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block2_conv1')
        net = tf.layers.conv2d(net,
                               filters=128,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block2_conv2')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      name='block2_pool')

        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block3_conv1')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block3_conv2')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block3_conv3')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      name='block3_pool')

        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block4_conv1')
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block4_conv2')
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block4_conv3')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      name='block4_pool')

        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block5_conv1')
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block5_conv2')
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='block5_conv3')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      name='block5_pool')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc1')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout1')
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc2')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout2')
        logits = tf.layers.dense(net,
                                 units=num_classes,
                                 name='prediction')
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = opt.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def concat_alex_net_model_fn(features, labels, mode):
    inputs = features['X1']
    concat_vector = features['X2']
    num_classes = 2
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_keep_prob = 0.5
    scope = 'concat_alexnet'
    with tf.variable_scope(scope) as sc:
        net = tf.layers.conv2d(inputs,
                               filters=64,
                               kernel_size=(11, 11),
                               strides=4,
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv1')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool1')
        net = tf.layers.conv2d(net,
                               filters=192,
                               kernel_size=(5, 5),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv2')
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(3, 3),
                                      strides=2,
                                      padding='same',
                                      name='pool2')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv3')
        net = tf.layers.conv2d(net,
                               filters=384,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv4')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv5')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,
                              units=4096,
                              activation=tf.nn.relu,
                              name='fc6')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout6')
        net = tf.layers.dense(net,
                              units=AUX_DATA_DIGITS,
                              activation=tf.nn.relu,
                              name='fc7')
        net = tf.layers.dropout(net,
                                rate=dropout_keep_prob,
                                training=is_training,
                                name='dropout7')
        net = tf.concat([net, concat_vector], 1)
        logits = tf.layers.dense(net,
                                 units=num_classes,
                                 name='fc8')
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'logits': logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = opt.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )
