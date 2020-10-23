import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.framework import ops


class SimpleSparseTensor:
  def __init__(self, indices, values, dense_shape):
    self.indices = indices
    self.values = values
    self.dense_shape = dense_shape

def simple_sparse_tensor_dense_matmul(sp_a,
                               b,
                               adjoint_a=False,
                               adjoint_b=False,
                               name=None):
  
  #sp_a = _convert_to_sparse_tensor(sp_a)
  with ops.name_scope(name, "SparseTensorDenseMatMul",
                      [sp_a.indices, sp_a.values, b]) as name:
    b = ops.convert_to_tensor(b, name="b")
    #to be compatible with different tensorflow versions where the name of the op changed
    try:
      dense_mat_mul = gen_sparse_ops.sparse_tensor_dense_mat_mul
    except:
      dense_mat_mul = gen_sparse_ops._sparse_tensor_dense_mat_mul
      
    return dense_mat_mul(
        a_indices=sp_a.indices,
        a_values=sp_a.values,
        a_shape=sp_a.dense_shape,
        b=b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b)

#helper function to cache static results using data api
def makeCache(x):
  it = tf.data.Dataset.from_tensors(x).cache().repeat().make_initializable_iterator()
  tf.add_to_collection('cache_initializers', it.initializer)
  return it.get_next()

#slice a sparse tensor along axis 0 from begin to the end of the array
#cache the indices, since these should not depend on the values in the sparse tensor
def simple_sparse_slice0begin(in_sparse, begin, doCache = False):

  out_shape = [in_sparse.dense_shape[0] - begin] + list(in_sparse.dense_shape)[1:]
  offset = [begin] + [0]*(len(in_sparse.dense_shape)-1)

  idxs = tf.where(in_sparse.indices[:,0] >= begin)
  idxs = tf.squeeze(idxs,-1)
  idxs = tf.pad(idxs, [[0,1]], constant_values=in_sparse.indices.shape[0])
  startidx = idxs[0]
  startidx = tf.cast(startidx,tf.shape(in_sparse.indices).dtype)
        
  out_indices = in_sparse.indices[startidx:] - offset
  
  if doCache:
    startidx, out_indices = makeCache((startidx, out_indices))
  
  out_values = in_sparse.values[startidx:]
  return SimpleSparseTensor(out_indices, out_values, out_shape)

def simple_sparse_to_dense(in_sparse):
  return tf.scatter_nd(in_sparse.indices, in_sparse.values, in_sparse.dense_shape)
