import tensorflow as tf


def get_prediction(classifier, data, label):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=data,
        y=label,
        num_epochs=1,
        shuffle=False
    )
    res = classifier.predict(input_fn=input_fn)
    return res


