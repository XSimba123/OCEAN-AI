import tensorflow as tf
from .model import concat_alex_net_model_fn, alex_net_model_fn, VGG_net_model_fn

BATCH_SIZE = 64
NUM_EPOCH = 1
SHUFFLE = False
EPOCH = 10

def get_classifier(model_dir, model_type):
    if model_type == 'alex_net':
        clf = tf.estimator.Estimator(
            model_fn=alex_net_model_fn,
            model_dir=model_dir
        )
    elif model_type == 'concat_net':
        clf = tf.estimator.Estimator(
            model_fn=concat_alex_net_model_fn,
            model_dir=model_dir
        )
    elif model_type == 'VGG_net':
        clf = tf.estimator.Estimator(
            model_fn=VGG_net_model_fn,
            model_dir=model_dir
        )
    else:
        print('%s is not supported.' % model_type)
        return None
    return clf


def train_model(classifier, data, label, param=None):
    b_size = BATCH_SIZE if param is None or 'batch_size' not in param else param['batch_size']
    num_epoch = NUM_EPOCH if param is None or 'num_epoch' not in param else param['num_epoch']
    shuffle = SHUFFLE if param is None or 'shuffle' not in param else param['shuffle']

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=data,
        y=label,
        batch_size=b_size,
        num_epochs=num_epoch,
        shuffle=shuffle
    )

    classifier.train(input_fn=input_fn)


def eval_model(classifier, data, label, param=None):
    num_epoch = NUM_EPOCH if param is None or 'num_epoch' not in param else param['num_epoch']
    shuffle = SHUFFLE if param is None or 'shuffle' not in param else param['shuffle']

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=data,
        y=label,
        num_epochs=num_epoch,
        shuffle=shuffle
    )

    res = classifier.evaluate(input_fn=input_fn)
    print(res)
