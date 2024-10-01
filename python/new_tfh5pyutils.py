import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_control_flow_v2()
# Enable eager execution
# tf.enable_eager_execution()

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from HiggsAnalysis.CombinedLimit.tfsparseutils import SimpleSparseTensor

def maketensor(h5dset):
  
  ndims = len(h5dset.shape)
  if ndims>1:
    raise Exception("Only flat tensors are supported")
  
  if 'original_shape' in h5dset.attrs:
    shape = h5dset.attrs['original_shape']
  else:
    shape = h5dset.shape
    
  if h5dset.size == 0:
    return tf.zeros(shape,h5dset.dtype)
  
  chunksize = h5dset.chunks[0]
  chunkiter = range(0,h5dset.size,chunksize)
  
  def readChunk(i):
    ielem = chunkiter[i]
    aout = h5dset[ielem:ielem+chunksize]
    if not aout.shape == h5dset.chunks:
      aout.resize(h5dset.chunks)
    return aout  
      
  #calculate number of chunks
  nchunks = len(chunkiter)  
  
  #create tf Dataset which reads one chunk at a time.  The last chunk may need to be padded to match the chunk size.
  #Chunks are then batched together if necessary, and then the padded part is sliced out if necessary
  #n.b. map_and_batch is used instead of the simple dataset batch function because in tf 1.6 this apparently avoids an
  #extra copy in memory
  
  dset = tf.data.Dataset.range(nchunks)
  # dset = dset.map(lambda x: tf.reshape(tf.py_func(readChunk,[x],tf.as_dtype(h5dset.dtype)),h5dset.chunks))
  dset = dset.map(lambda x: tf.reshape(tf.py_function(readChunk,[x],tf.as_dtype(h5dset.dtype)),h5dset.chunks))
  if nchunks>1:
    # dset = dset.apply(tf.contrib.data.map_and_batch(lambda x: x, nchunks))
    dset = dset.apply(tf.data.experimental.map_and_batch(lambda x: x, nchunks))
    if not h5dset.shape[0]%h5dset.chunks[0] == 0:
      paddedshape = (nchunks*chunksize,)
      dset = dset.map(lambda x: tf.reshape(x,paddedshape)[:h5dset.shape[0]])

  dset = dset.map(lambda x: tf.reshape(x,shape))    
  dset = dset.cache().repeat(-1)
  atensor = dset.make_one_shot_iterator().get_next()
  # iterator = tf.compat.v1.data.make_one_shot_iterator(dset)
  # atensor = iterator.get_next()
  # iterator = iter(dset)
  # atensor = next(iterator)
  return atensor

def makesparsetensor(h5group):
  indices = maketensor(h5group['indices'])
  values = maketensor(h5group['values'])
  dense_shape = h5group.attrs['dense_shape']

  return SimpleSparseTensor(indices,values,dense_shape)
