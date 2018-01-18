import json
import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers


def forward_propagation_for_predict(X, parameters):
    """
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"
                  the shapes are given in initialize_parameters

    Returns:
    Z4 -- the output of the last LINEAR unit
    """

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']

    # Input 28 x 28 x 1, Padding = 2, Stride = 1, Filter = 5 x 5 x 1 x 6, Output = 28 x 28 x 6
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    # Input 28 x 28 x 6, Padding = 0, Stride = 2, Filter = 2 x 2, Output = 14 x 14 x 6
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Input 14 x 14 x 6, Padding = 0, Stride = 1, Filter = 5 x 5 x 6 x 16, Output = 10 x 10 x 16
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
    A2 = tf.nn.relu(Z2)
    # Input 10 x 10 x 16, Padding = 0, Stride = 2, Filter = 2 x 2, Output = 5 x 5 x 16
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    P2 = layers.flatten(P2)

    Z3 = tf.matmul(tf.transpose(W3), tf.transpose(P2)) + b3.reshape(-1, 1)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(tf.transpose(W4), A3) + b4.reshape(-1, 1)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.matmul(tf.transpose(W5), A4) + b5.reshape(-1, 1)

    return Z5


def predict(X, parameters):
    m, _, _, _ = X.shape
    x = tf.placeholder("float", [None, 28, 28, 1])

    Z5 = forward_propagation_for_predict(x, parameters)
    p = tf.argmax(Z5)

    print(X.shape)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    for i in range(m):
        print("Prediction:", prediction[i])

    return prediction


def handler(event, context):

    if 'image' not in event:
        return {
            'statusCode': 400,
            'message': 'Image not found in the request.'
        }

    mnist_images = np.array([event['image']])
    tf.reset_default_graph()

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './lenet5_trained_model')
        parameters = {
            'W1': sess.run('W1:0'),
            'W2': sess.run('W2:0'),
            'W3': sess.run('fully_connected/weights:0'),
            'b3': sess.run('fully_connected/biases:0'),
            'W4': sess.run('fully_connected_1/weights:0'),
            'b4': sess.run('fully_connected_1/biases:0'),
            'W5': sess.run('fully_connected_2/weights:0'),
            'b5': sess.run('fully_connected_2/biases:0')
        }

        prediction = predict(mnist_images, parameters)

        return {
            'isBase64Encoded': False,
            'statusCode': 200,
            'body': json.dumps({
                'data': prediction.tolist()
            })
        }
