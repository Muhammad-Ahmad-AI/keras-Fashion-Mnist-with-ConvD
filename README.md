# keras-Fashion-Mnist-with-ConvD
Table of Contents

Why we made Fashion-MNIST

Get the Data

Usage

Benchmark

Visualization

Contributing

Contact

Citing Fashion-MNIST

License

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example how the data looks (each class takes three-rows):





Why we made Fashion-MNIST
The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

To Serious Machine Learning Researchers
Seriously, we are talking about replacing MNIST. Here are some good reasons:

MNIST is too easy. Convolutional nets can achieve 99.7% on MNIST. Classic machine learning algorithms can also achieve 97% easily. Check out our side-by-side benchmark for Fashion-MNIST vs. MNIST, and read "Most pairs of MNIST digits can be distinguished pretty well by just one pixel."
MNIST is overused. In this April 2017 Twitter thread, Google Brain research scientist and deep learning expert Ian Goodfellow calls for people to move away from MNIST.
MNIST can not represent modern CV tasks, as noted in this April 2017 Twitter thread, deep learning expert/Keras author François Chollet.
Get the Data
Many ML libraries already include Fashion-MNIST data/API, give it a try!

You can use direct links to download the dataset. The data is stored in the same format as the original MNIST data.

Name	Content	Examples	Size	Link	MD5 Checksum
train-images-idx3-ubyte.gz	training set images	60,000	26 MBytes	Download	8d4fb7e6c68d591d4c3dfef9ec88bf0d
train-labels-idx1-ubyte.gz	training set labels	60,000	29 KBytes	Download	25c81989df183df01b3e8a0aad5dffbe
t10k-images-idx3-ubyte.gz	test set images	10,000	4.3 MBytes	Download	bef4ecab320f06d8554ea6380940ec79
t10k-labels-idx1-ubyte.gz	test set labels	10,000	5.1 KBytes	Download	bb300cfdad3c16e7a12a480ee83cd310
Alternatively, you can clone this GitHub repository; the dataset appears under data/fashion. This repo also contains some scripts for benchmark and visualization.

git clone git@github.com:zalandoresearch/fashion-mnist.git
Labels
Each training and test example is assigned to one of the following labels:

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
Usage
Loading data with Python (requires NumPy)
Use utils/mnist_reader in this repo:

import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
Loading data with Tensorflow
Make sure you have downloaded the data and placed it in data/fashion. Otherwise, Tensorflow will download and use the original MNIST.

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
Note, Tensorflow supports passing in a source url to the read_data_sets. You may use:

data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
Also, an official Tensorflow tutorial of using tf.keras, a high-level API to train Fashion-MNIST can be found here.

Loading data with other machine learning libraries
To date, the following libraries have included Fashion-MNIST as a built-in dataset. Therefore, you don't need to download Fashion-MNIST by yourself. Just follow their API and you are ready to go.

Apache MXNet Gluon
deeplearn.js
Kaggle
Pytorch
Keras
Edward
Tensorflow
TensorFlow Datasets
Torch
JuliaML
Chainer
You are welcome to make pull requests to other open-source machine learning packages, improving their support to Fashion-MNIST dataset.

Loading data with other languages
As one of the Machine Learning community's most popular datasets, MNIST has inspired people to implement loaders in many different languages. You can use these loaders with the Fashion-MNIST dataset as well. (Note: may require decompressing first.) To date, we haven't yet tested all of these loaders with Fashion-MNIST.

C
C++
Java
Python and this and this
Scala
Go
C#
NodeJS and this
Swift
R and this
Matlab
Ruby
Rust
