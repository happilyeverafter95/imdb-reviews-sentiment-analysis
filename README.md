# Tensorflow Serving 2.0

Serving a sentiment analysis model (movie review sentiment) using Tensorflow 2.0 and Tensorflow serving.

If you're using Tensorflow 1.X, please refer to my [toxic-comment-classifier](https://github.com/happilyeverafter95/toxic-comment-classifer) for a serving example with a text classifier. I would recommend migrating to the newer versions as Tensorflow 2.X introduces a lot of additional support for NLP models.

## Usage Requirements

Training script written and tested for in Python 3.8

* Install all Python dependencies using `pip install -r requirements.txt`

To train the model and output the SavedModel object, run `python -m model.train`. This will output the SavedModel files to `classifier/saved_models/1`.

## Serve the Model

For more information, refer to the official [tensorflow/serving repo](https://github.com/tensorflow/serving).

### Start the Server

Instructions paraphrased and adapted from the TensorFlow Serving repo.

1. Install Docker
2. Fetch the latest version of TensorFlow Serving Docker docker pull tensorflow/serving
3. Specify the directory for export. In the root directory of this repo, run `ModelPath="$(pwd)/classifier"`
   
**To start the server:**

```
docker run -t --rm -p 8501:8501 \
    -v "$ModelPath/saved_models:/models/sentiment_analysis" \
    -e MODEL_NAME=sentiment_analysis \
    tensorflow/serving
```

### Sample Curl Command

```
curl -d '{"inputs":{"review": ["worst movie EVER"]}}' \
  -X POST http://localhost:8501/v1/models/sentiment_analysis:predict
```

All preprocessing steps are applied to the input prior to prediction. You might also notice that the payload has an additional parameter alongside the actual prediction. This was defined as a part of the model signature and can be customized to provide more meaningful metadata.