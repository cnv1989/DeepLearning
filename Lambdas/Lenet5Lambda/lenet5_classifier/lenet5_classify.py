import json
import numpy as np
import tensorflow as tf


def handler(event, context):

    if 'image' not in event:
        return {
            'statusCode': 400,
            'message': 'Image not found in the request.'
        }

    mnist_images = np.array([event['image']])

    tf.reset_default_graph()
    graph = tf.get_default_graph()
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './lenet5_trained_model')
        X = graph.get_tensor_by_name('Placeholder:0')
        pred = graph.get_tensor_by_name('ArgMax:0')
        prediction = pred.eval(feed_dict={X: mnist_images})

        return {
            'isBase64Encoded': False,
            'statusCode': 200,
            'body': json.dumps({
                'data': prediction.tolist()
            })
        }
