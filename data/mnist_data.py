import os
import tensorflow as tf
from six.moves import urllib
import gzip
import numpy as np

WORK_DIRECTORY = os.path.join(os.path.expanduser('~'), "data/mnist/")
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def maybe_download(data_dir, filename, url):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)
  filepath = os.path.join(data_dir, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
  return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def load(data_dir, url='http://yann.lecun.com/exdb/mnist/', subset='train'):
  # Get the data.
  train_data_filename = maybe_download(data_dir, 'train-images-idx3-ubyte.gz', url)
  train_labels_filename = maybe_download(data_dir, 'train-labels-idx1-ubyte.gz', url)
  test_data_filename = maybe_download(data_dir, 't10k-images-idx3-ubyte.gz', url)
  test_labels_filename = maybe_download(data_dir, 't10k-labels-idx1-ubyte.gz', url)

  # Extract it into numpy arrays.
  if subset=='train':
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    return train_data, train_labels
  elif subset=='test':
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    return test_data, test_labels
  else:
    raise NotImplementedError("subset must be train or test")

class DataLoader(object):
  """ an object that generates batches of MNIST data for training """

  def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
    """
    - data_dir is location where to store files
    - subset is train|test
    - batch_size is int, of #examples to load at once
    - rng is np.random.RandomState object for reproducibility
    """

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.return_labels = return_labels

    # create temporary storage for the data, if not yet created
    if not os.path.exists(data_dir):
        print('creating folder', data_dir)
        os.makedirs(data_dir)

    # load MNIST training data to RAM
    self.data, self.labels = load(os.path.join(data_dir,'mnist'), subset=subset)

    self.p = 0 # pointer to where we are in iteration
    self.rng = np.random.RandomState(1) if rng is None else rng

  def get_observation_size(self):
    return self.data.shape[1:]

  def get_num_labels(self):
    return np.amax(self.labels) + 1

  def reset(self):
    self.p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    # on first iteration lazily permute all data
    if self.p == 0 and self.shuffle:
      inds = self.rng.permutation(self.data.shape[0])
      self.data = self.data[inds]
      self.labels = self.labels[inds]

    # on last iteration reset the counter and raise StopIteration
    if self.p + n > self.data.shape[0]:
      self.reset() # reset for next time we get called
      raise StopIteration

    # on intermediate iterations fetch the next batch
    x = self.data[self.p : self.p + n]
    y = self.labels[self.p : self.p + n]
    self.p += self.batch_size

    if self.return_labels:
      return x,y
    else:
      return x

  next = __next__  # Python 2 compatibility


if __name__=="__main__":
  d = DataLoader(WORK_DIRECTORY, 'train', 12)
  print(d.data.shape)
