# Serverless REST API to classify MINST images

This repo constains serverless implementation for classifying MINST handwritten digit images into numerical output.

Two lambda functions are implemented to work withing the code size limts enforced by AWS lambda.

**image_processor**: This lambda function is triggered in response to POST request to `/dev/predict` endpoint implemented via API gateway. The endpoint accepts images in Base64. The function consumes the base64 input, converts it to image, resizes the image and converts and reshapes the image to python list and finally invokes the `lenet5_classifier` function with the stringified list payload. The function outputs the result of the invocation in the body of the response.


**lenet5_classifier** This lambda function takes the the nested python list as input, converts it into numpy array and feeds it into the learned model and outputs the result of the classification.


# Setup

Setups to setup and deploy lambdas into AWS account


1. Configure aws credentials.

    `aws configure`

2. To deploy each lambda function `cd` into the corresponding directory and perform following operations.

    `npm install`

    `npm run sls deploy -- -v`

3. Once the two functions are deployed you can test the API using

    `curl -X POST -H "Content-Type: application/json" -d '{"image" : "'"$( base64 /path/to/minst/image.png )"'"}' https://###.us-west-1.amazonaws.com/dev/predict`
