try:
  import unzip_requirements
except ImportError:
  pass

import base64
import boto3
import json
import numpy as np

from scipy import misc

from PIL import Image
from io import BytesIO

lambda_client = boto3.client('lambda')


def process(event, context):
    try:
        json_data = json.loads(event['body'])
    except ValueError as e:
        return {
            'statusCode': 400,
            'message': e
        }

    if 'image' not in json_data:
        return {
            'statusCode': 400,
            'message': 'Image not found in the request.'
        }

    img_base64 = np.array(json_data['image'])
    image = Image.open(BytesIO(base64.b64decode(img_base64)))
    mnist_image = misc.imresize(image, size=(28, 28)).reshape(28, 28, 1)
    mnist_image = np.asarray(mnist_image)/255.0
    payload = json.dumps({
        "image": mnist_image.tolist()
    }).encode()

    invoke_response = lambda_client.invoke(
        FunctionName="Lenet5Classifier-dev-Lenet5Classify",
        InvocationType='RequestResponse',
        Payload=payload
    )

    response = invoke_response['Payload'].read()

    response = {
        "statusCode": 200,
        "body": json.dumps(response)
    }

    return response
