from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_control_flow_v2()

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.util import nest

from scipy.optimize import SR1, LinearConstraint, NonlinearConstraint, Bounds


__all__ = ['ScipyTROptimizerInterface']

def jacobian(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None,
             parallel_iterations=10,
             back_prop = True,
             stop_gradients = None):
  """Constructs the jacobian of sum of `ys` with respect to `x` in `xs`.
  `jacobians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the jacobian of `sum(ys)`.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of jacobian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  kwargs = {
      "colocate_gradients_with_ops": colocate_gradients_with_ops,
      "gate_gradients": gate_gradients,
      "aggregation_method": aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  #hessians = []
  #_gradients = gradients(ys, xs, **kwargs)
  gradient = ys
  x = xs
  #for gradient, x in zip(_gradients, xs):  
  # change shape to one-dimension without graph branching
  gradient = array_ops.reshape(gradient, [-1])

  # Declare an iterator and tensor array loop variables for the gradients.
  n = array_ops.size(gradient)
  loop_vars = [
      array_ops.constant(0, dtypes.int32),
      tensor_array_ops.TensorArray(x.dtype, n)
  ]
  # Iterate over all elements of the gradient and compute second order
  # derivatives.
  _, hessian = control_flow_ops.while_loop(
      lambda j, _: j < n,
      lambda j, result: (j + 1,
                          result.write(j, tf.gradients(gradient[j], x, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients, aggregation_method=aggregation_method, stop_gradients=stop_gradients)[0])),
      loop_vars,
      parallel_iterations = parallel_iterations,
      back_prop = back_prop,
  )

  _shapey = array_ops.shape(ys)
  _shape = array_ops.shape(x)
  _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                        array_ops.concat((_shapey, _shape), 0))
  #hessians.append(_reshaped_hessian)
  return _reshaped_hessian


def sum_loop(loop_fn, loop_fn_accumulators, iters, parallel_iterations=10, back_prop=True):
  """Runs `loop_fn` `iters` times and sums the outputs.
  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  sums corresponding outputs of the different runs.
  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of loop_fn.
    iters: Number of iterations for which to run loop_fn.
  Returns:
    Returns a nested structure of stacked output tensor objects with the same
    nested structure as the output of `loop_fn`.
  """

  flat_loop_fn_accumulators = nest.flatten(loop_fn_accumulators)
  is_none_list = []

  def while_body(i, *ta_list):
    """Body of while loop."""
    fn_output = nest.flatten(loop_fn(i))
    if len(fn_output) != len(flat_loop_fn_accumulators):
      raise ValueError(
          "Number of expected outputs, %d, does not match the number of "
          "actual outputs, %d, from loop_fn" % (len(flat_loop_fn_accumulators),
                                                len(fn_output)))
    outputs = []
    del is_none_list[:]
    is_none_list.extend([x is None for x in fn_output])
    for out, ta in zip(fn_output, ta_list):
      # TODO(agarwal): support returning Operation objects from loop_fn.
      if out is not None:
        ta = ta + out
      outputs.append(ta)
    return tuple([i + 1] + outputs)

  ta_list = control_flow_ops.while_loop(
      lambda i, *ta: i < iters, while_body, [0] + [
          accumulator
          for accumulator in flat_loop_fn_accumulators
      ],parallel_iterations=parallel_iterations, back_prop=back_prop)[1:]

  # TODO(rachelim): enable this for sparse tensors

  output = [None if is_none else ta
            for ta, is_none in zip(ta_list, is_none_list)]
  return nest.pack_sequence_as(loop_fn_accumulators, output)

def _accumulate(list_):
  total = 0
  yield total
  for x in list_:
    total += x
    yield total


def _get_shape_tuple(tensor):
  return tuple(tensor.get_shape().as_list())


def _prod(array):
  prod = 1
  for value in array:
    prod *= value
  return prod


def _compute_gradients(tensor, var_list):
  grads = gradients.gradients(tensor, var_list)
  # tf.gradients sometimes returns `None` when it should return 0.
  return [
      grad if grad is not None else array_ops.zeros_like(var)
      for var, grad in zip(var_list, grads)
  ]

class ExternalOptimizerInterface(object):
  """Base class for interfaces with external optimization algorithms.
  Subclass this and implement `_minimize` in order to wrap a new optimization
  algorithm.
  `ExternalOptimizerInterface` should not be instantiated directly; instead use
  e.g. `ScipyOptimizerInterface`.
  @@__init__
  @@minimize
  """

  def __init__(self,
               loss,
               var_list=None,
               equalities=None,
               inequalities=None,
               var_to_bounds=None,
               **optimizer_kwargs):
    """Initialize a new interface instance.
    Args:
      loss: A scalar `Tensor` to be minimized.
      var_list: Optional `list` of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      equalities: Optional `list` of equality constraint scalar `Tensor`s to be
        held equal to zero.
      inequalities: Optional `list` of inequality constraint scalar `Tensor`s
        to be held nonnegative.
      var_to_bounds: Optional `dict` where each key is an optimization
        `Variable` and each corresponding value is a length-2 tuple of
        `(low, high)` bounds. Although enforcing this kind of simple constraint
        could be accomplished with the `inequalities` arg, not all optimization
        algorithms support general inequality constraints, e.g. L-BFGS-B. Both
        `low` and `high` can either be numbers or anything convertible to a
        NumPy array that can be broadcast to the shape of `var` (using
        `np.broadcast_to`). To indicate that there is no bound, use `None` (or
        `+/- np.infty`). For example, if `var` is a 2x3 matrix, then any of
        the following corresponding `bounds` could be supplied:
        * `(0, np.infty)`: Each element of `var` held positive.
        * `(-np.infty, [1, 2])`: First column less than 1, second column less
          than 2.
        * `(-np.infty, [[1], [2], [3]])`: First row less than 1, second row less
          than 2, etc.
        * `(-np.infty, [[1, 2, 3], [4, 5, 6]])`: Entry `var[0, 0]` less than 1,
          `var[0, 1]` less than 2, etc.
      **optimizer_kwargs: Other subclass-specific keyword arguments.
    """
    self._loss = loss
    self._equalities = equalities or []
    self._inequalities = inequalities or []

    if var_list is None:
      self._vars = variables.trainable_variables()
    else:
      self._vars = list(var_list)

    packed_bounds = None
    if var_to_bounds is not None:
      left_packed_bounds = []
      right_packed_bounds = []
      for var in self._vars:
        shape = var.get_shape().as_list()
        bounds = (-np.infty, np.infty)
        if var in var_to_bounds:
          bounds = var_to_bounds[var]
        left_packed_bounds.extend(list(np.broadcast_to(bounds[0], shape).flat))
        right_packed_bounds.extend(list(np.broadcast_to(bounds[1], shape).flat))
      packed_bounds = list(zip(left_packed_bounds, right_packed_bounds))
    self._packed_bounds = packed_bounds

    self._update_placeholders = [
        array_ops.placeholder(var.dtype) for var in self._vars
    ]
    self._var_updates = [
        var.assign(array_ops.reshape(placeholder, _get_shape_tuple(var)))
        for var, placeholder in zip(self._vars, self._update_placeholders)
    ]

    loss_grads = _compute_gradients(loss, self._vars)
    equalities_grads = [
        _compute_gradients(equality, self._vars)
        for equality in self._equalities
    ]
    inequalities_grads = [
        _compute_gradients(inequality, self._vars)
        for inequality in self._inequalities
    ]

    self.optimizer_kwargs = optimizer_kwargs

    self._packed_var = self._pack(self._vars)
    self._packed_loss_grad = self._pack(loss_grads)
    self._packed_equality_grads = [
        self._pack(equality_grads) for equality_grads in equalities_grads
    ]
    self._packed_inequality_grads = [
        self._pack(inequality_grads) for inequality_grads in inequalities_grads
    ]

    dims = [_prod(_get_shape_tuple(var)) for var in self._vars]
    accumulated_dims = list(_accumulate(dims))
    self._packing_slices = [
        slice(start, end)
        for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
    ]

  def minimize(self,
               session=None,
               feed_dict=None,
               fetches=None,
               step_callback=None,
               loss_callback=None,
               **run_kwargs):
    """Minimize a scalar `Tensor`.
    Variables subject to optimization are updated in-place at the end of
    optimization.
    Note that this method does *not* just return a minimization `Op`, unlike
    `Optimizer.minimize()`; instead it actually performs minimization by
    executing commands to control a `Session`.
    Args:
      session: A `Session` instance.
      feed_dict: A feed dict to be passed to calls to `session.run`.
      fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
        as positional arguments.
      step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
      loss_callback: A function to be called every time the loss and gradients
        are computed, with evaluated fetches supplied as positional arguments.
      **run_kwargs: kwargs to pass to `session.run`.
    """
    session = session or ops.get_default_session()
    feed_dict = feed_dict or {}
    fetches = fetches or []

    loss_callback = loss_callback or (lambda *fetches: None)
    step_callback = step_callback or (lambda xk: None)

    # Construct loss function and associated gradient.
    loss_grad_func = self._make_eval_func([self._loss,
                                           self._packed_loss_grad], session,
                                          feed_dict, fetches, loss_callback)

    # Construct equality constraint functions and associated gradients.
    equality_funcs = self._make_eval_funcs(self._equalities, session, feed_dict,
                                           fetches)
    equality_grad_funcs = self._make_eval_funcs(self._packed_equality_grads,
                                                session, feed_dict, fetches)

    # Construct inequality constraint functions and associated gradients.
    inequality_funcs = self._make_eval_funcs(self._inequalities, session,
                                             feed_dict, fetches)
    inequality_grad_funcs = self._make_eval_funcs(self._packed_inequality_grads,
                                                  session, feed_dict, fetches)

    # Get initial value from TF session.
    initial_packed_var_val = session.run(self._packed_var)

    # Perform minimization.
    packed_var_val = self._minimize(
        initial_val=initial_packed_var_val,
        loss_grad_func=loss_grad_func,
        equality_funcs=equality_funcs,
        equality_grad_funcs=equality_grad_funcs,
        inequality_funcs=inequality_funcs,
        inequality_grad_funcs=inequality_grad_funcs,
        packed_bounds=self._packed_bounds,
        step_callback=step_callback,
        optimizer_kwargs=self.optimizer_kwargs)
    var_vals = [
        packed_var_val[packing_slice] for packing_slice in self._packing_slices
    ]

    # Set optimization variables to their new values.
    session.run(
        self._var_updates,
        feed_dict=dict(zip(self._update_placeholders, var_vals)),
        **run_kwargs)

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):
    """Wrapper for a particular optimization algorithm implementation.
    It would be appropriate for a subclass implementation of this method to
    raise `NotImplementedError` if unsupported arguments are passed: e.g. if an
    algorithm does not support constraints but `len(equality_funcs) > 0`.
    Args:
      initial_val: A NumPy vector of initial values.
      loss_grad_func: A function accepting a NumPy packed variable vector and
        returning two outputs, a loss value and the gradient of that loss with
        respect to the packed variable vector.
      equality_funcs: A list of functions each of which specifies a scalar
        quantity that an optimizer should hold exactly zero.
      equality_grad_funcs: A list of gradients of equality_funcs.
      inequality_funcs: A list of functions each of which specifies a scalar
        quantity that an optimizer should hold >= 0.
      inequality_grad_funcs: A list of gradients of inequality_funcs.
      packed_bounds: A list of bounds for each index, or `None`.
      step_callback: A callback function to execute at each optimization step,
        supplied with the current value of the packed variable vector.
      optimizer_kwargs: Other key-value arguments available to the optimizer.
    Returns:
      The optimal variable vector as a NumPy vector.
    """
    raise NotImplementedError(
        'To use ExternalOptimizerInterface, subclass from it and implement '
        'the _minimize() method.')

  @classmethod
  def _pack(cls, tensors):
    """Pack a list of `Tensor`s into a single, flattened, rank-1 `Tensor`."""
    if not tensors:
      return None
    elif len(tensors) == 1:
      return array_ops.reshape(tensors[0], [-1])
    else:
      flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
      return array_ops.concat(flattened, 0)

  def _make_eval_func(self, tensors, session, feed_dict, fetches,
                      callback=None):
    """Construct a function that evaluates a `Tensor` or list of `Tensor`s."""
    if not isinstance(tensors, list):
      tensors = [tensors]
    num_tensors = len(tensors)

    def eval_func(x):
      """Function to evaluate a `Tensor`."""
      augmented_feed_dict = {
          var: x[packing_slice].reshape(_get_shape_tuple(var))
          for var, packing_slice in zip(self._vars, self._packing_slices)
      }
      augmented_feed_dict.update(feed_dict)
      augmented_fetches = tensors + fetches

      augmented_fetch_vals = session.run(
          augmented_fetches, feed_dict=augmented_feed_dict)

      if callable(callback):
        callback(*augmented_fetch_vals[num_tensors:])

      return augmented_fetch_vals[:num_tensors]

    return eval_func

  def _make_eval_funcs(self,
                       tensors,
                       session,
                       feed_dict,
                       fetches,
                       callback=None):
    return [
        self._make_eval_func(tensor, session, feed_dict, fetches, callback)
        for tensor in tensors
    ]



class ScipyTROptimizerInterface(ExternalOptimizerInterface):

  _DEFAULT_METHOD = 'trust-constr'


  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
    hess = optimizer_kwargs.pop('hess', SR1())
    bounds = optimizer_kwargs.pop('bounds', None)

    constraints = []
    for func, grad_func, tensor in zip(equality_funcs, equality_grad_funcs,self._equalities):
      lb = np.zeros(tensor.shape,dtype=initial_val.dtype)
      ub = lb
      constraints.append(NonlinearConstraint(func, lb, ub, jac = grad_func, hess=SR1()))
    for func, grad_func, tensor in zip(inequality_funcs, inequality_grad_funcs,self._inequalities):
      lb = np.zeros(tensor.shape,dtype=initial_val.dtype)
      ub = np.inf*np.ones(tensor.shape,dtype=initial_val.dtype)
      constraints.append(NonlinearConstraint(func, lb, ub, jac = grad_func, hess=SR1(),keep_feasible=False))

    import scipy.optimize  # pylint: disable=g-import-not-at-top

    if packed_bounds != None:
      lb = np.zeros_like(initial_val)
      ub = np.zeros_like(initial_val)
      for ival,(lbval,ubval) in enumerate(packed_bounds):
        lb[ival] = lbval
        ub[ival] = ubval
        if lbval==ubval:
          lb[ival] = initial_val[ival]
          ub[ival] = initial_val[ival]
      isnull = np.all(np.equal(lb,-np.inf)) and np.all(np.equal(ub,np.inf))
      if not isnull:
        constraints.append(LinearConstraint(np.eye(initial_val.shape[0],dtype=initial_val.dtype),lb,ub,keep_feasible=True))
    elif bounds != None:
      lb = np.copy(bounds.lb)
      ub = np.copy(bounds.ub)
      isnull = np.all(np.equal(lb,-np.inf)) and np.all(np.equal(ub,np.inf))
      fixed = np.equal(lb,ub)
      lb = np.where(fixed,initial_val,lb)
      ub = np.where(fixed,initial_val,ub)
      if not isnull:
        constraints.append(LinearConstraint(np.eye(initial_val.shape[0],dtype=initial_val.dtype),lb,ub,keep_feasible=True))

    minimize_args = [loss_grad_func, initial_val]
    minimize_kwargs = {
        'jac': True,
        'hess' : hess,
        'callback': None,
        'method': method,
        'constraints': constraints,
        'bounds': None,
    }

    for kwarg in minimize_kwargs:
      if kwarg in optimizer_kwargs:
        if kwarg == 'bounds':
          # Special handling for 'bounds' kwarg since ability to specify bounds
          # was added after this module was already publicly released.
          raise ValueError(
              'Bounds must be set using the var_to_bounds argument')
        raise ValueError(
            'Optimizer keyword arg \'{}\' is set '
            'automatically and cannot be injected manually'.format(kwarg))

    minimize_kwargs.update(optimizer_kwargs)
    
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)

    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [result.message, result.fun]
    if hasattr(result, 'nit'):
      # Some optimization methods might not provide information such as nit and
      # nfev in the return. Logs only available information.
      message_lines.append('  Number of iterations: %d')
      message_args.append(result.nit)
    if hasattr(result, 'nfev'):
      message_lines.append('  Number of functions evaluations: %d')
      message_args.append(result.nfev)
    logging.info('\n'.join(message_lines), *message_args)

    return result['x']
