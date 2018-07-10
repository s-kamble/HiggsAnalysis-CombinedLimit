from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.ops import gradients_impl

from scipy.optimize import SR1, LinearConstraint, NonlinearConstraint, Bounds


__all__ = ['ScipyTROptimizerInterface']

class TrustSR1:
    
  def initialize(self, loss, var, initialtrustradius = 1.,doSR1=True):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(True, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(True)
    self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    alist = []
    alist.append(tf.assign(self.trustradius,initialtrustradius))
    alist.append(tf.assign(self.loss_old, loss))
    alist.append(tf.assign(self.predicted_reduction, 0.))
    alist.append(tf.assign(self.var_old, var))
    alist.append(tf.assign(self.atboundary_old,False))
    alist.append(tf.assign(self.doiter_old,False))
    alist.append(tf.assign(self.isfirstiter,True))

    alist.append(tf.assign(self.B,tf.eye(int(var.shape[0]),dtype=var.dtype)))
    alist.append(tf.assign(self.H,tf.eye(int(var.shape[0]),dtype=var.dtype)))
    alist.append(tf.assign(self.doscaling,True))
    alist.append(tf.assign(self.grad_old,self.grad))
    

    return tf.group(alist)
  
  def setHessApprox(self,b,sess):
    self.B.load(b,sess)
    self.H.load(np.linalg.inv(b),sess)
    self.doscaling.load(False, sess)
  
  def minimize(self, loss, var):
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    edmtol = 1e-8
    #edmtol = 0.
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    
    def hesspexact(v):
      return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    def hesspapprox(B,v):
      return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    def doSR1Scaling(Bin,Hin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      H = Hin/scale
      return (B,H,tf.constant(False))
    
    def doSR1Update(Bin,Hin,yin,dxin):
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      Bx = tf.matmul(Bin,dx)
      dyBx = y - Bx
      den = tf.matmul(dyBx,dx,transpose_a=True)
      deltaB = tf.matmul(dyBx,dyBx,transpose_b=True)/den
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.less(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      deltaB = tf.where(dentest,tf.zeros_like(deltaB),deltaB)
      #deltaB = tf.where(self.doiter_old, deltaB, tf.zeros_like(deltaB))
      
      Hy = tf.matmul(Hin,y)
      dxHy = dx - Hy
      deltaH = tf.matmul(dxHy,dxHy,transpose_b=True)/tf.matmul(dxHy,y,transpose_a=True)
      deltaH = tf.where(dentest,tf.zeros_like(deltaH),deltaH)
      #deltaH = tf.where(self.doiter_old, deltaH, tf.zeros_like(deltaH))
      
      B = Bin + deltaB
      H = Hin + deltaH
      return (B,H)
    
    grad = self.grad
    B = self.B
    H = self.H
    
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    B,H = tf.cond(self.doiter_old, lambda: doSR1Update(B,H,dgrad,dx), lambda: (B,H))    
    
    #ng = -grad
    #Bng = hesspexact(ng)
    #B,H = tf.cond(self.doiter_old,lambda: doSR1Update(B,H,Bng,ng), lambda: (B,H))
    
    #nHg = -hesspapprox(H,grad)
    #BnHg = hesspexact(nHg)
    #B,H = tf.cond(self.doiter_old, lambda: doSR1Update(B,H,BnHg,nHg), lambda: (B,H) )
    
    #e,v = tf.self_adjoint_eig(B)
    #e0 = e[0]
    #v0 = v[0]
    
    #Bv0 = hesspexact(v0)
    #B,H = tf.cond(self.doiter_old, lambda : doSR1Update(B,H,Bv0,v0), lambda: (B,H))
    #B,H = doSR1Update(B,H,Bv0,v0)
    
    
    
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,0.15),tf.logical_not(isconverged))
    
    #trustradius_out = tf.Print(trustradius_out,[self.loss_old, loss, actual_reduction,self.predicted_reduction,isnull,rho,trustradius_out,isconvergedxtol,isconvergededmtol,doiter,e0])
    
    #def build_solu():
      #solu = -hesspapprox(H,grad)
    
    def build_sol():
      grad = self.grad
      e0 = tf.self_adjoint_eigvals(B)[0]

      solu = -hesspapprox(H,grad)
      usesolu = e0>0. 
      usesolu = usesolu & (tf.reduce_sum(tf.square(solu)) < tf.square(trustradius_out))

      def dfval_quad(twov):
          return tf.reduce_sum(grad*twov) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(B,tf.reshape(twov,[-1,1])),[-1])*twov)      
  

      def build_iter():

        twograd = grad
        twohess = B

        #ispos = e0>0.
        
        #d0 = ng
        #d1 = tf.where(ispos,nHg,v0)
        
        #norm0 = tf.reciprocal(tf.sqrt(tf.reduce_sum(tf.square(d0))))
        #norm1 = tf.reciprocal(tf.sqrt(tf.reduce_sum(tf.square(d1))))
        
        #d0 *= norm0
        #d1 *= norm1
              
        #Bd0 = norm0*Bng
        #Bd1 = norm1*tf.where(ispos,BnHg, Bv0)
        
        #twograd = tf.stack([tf.reduce_sum(grad*d0),tf.reduce_sum(grad*d1)],axis=0)
        twogradcol = tf.reshape(twograd,[-1,1])
        
        #twohess0 = tf.stack([tf.reduce_sum(d0*Bd0),tf.reduce_sum(d0*Bd1)],axis=0)
        #twohess1 = tf.stack([tf.reduce_sum(d1*Bd0),tf.reduce_sum(d1*Bd1)],axis=0)
        #twohess = tf.stack([twohess0,twohess1],axis=0)
        
        #def dfval_quad(twov):
          #return tf.reduce_sum(twograd*twov) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(twohess,tf.reshape(twov,[-1,1])),[-1])*twov)      
        
        #solu = -tf.matrix_solve(twohess,twogradcol)
        
        #twoeig0 = tf.self_adjoint_eigvals(twohess)[0]
        twoeig0 = e0
        #lim = -twoeig0 + 1e-8
        start = -1.1*twoeig0
        lim = tf.maximum(-1.01*twoeig0, twoeig0 + 0.1)
        
        alpha = tf.where(twoeig0>0., tf.zeros_like(start), start)
        
        #alpha = tf.maximum(alpha,lim)
        #alpha = tf.Print(alpha,[alpha])
        twoeye = tf.eye(int(twohess.shape[0]),dtype=twohess.dtype)
        for i in range(8):
          #alpha = tf.Print(alpha,[alpha])
          m = twohess + alpha*twoeye
          r = tf.cholesky(m)
          pl = -tf.cholesky_solve(r,twogradcol)
          #pl = -tf.matrix_solve(m,twogradcol)
          ql = tf.matrix_triangular_solve(r,pl,adjoint=True, lower=True)
          #ql = tf.matrix_solve(tf.transpose(r),pl)
          plnormsq = tf.reduce_sum(tf.square(pl))
          qlnormsq = tf.reduce_sum(tf.square(ql))
          alpha = alpha + (plnormsq/qlnormsq)*(tf.sqrt(plnormsq)-trustradius_out)/trustradius_out
          alpha = tf.maximum(alpha,lim)
        
        #usesolu = twoeig0>0. 
        #usesolu = usesolu & (tf.reduce_sum(tf.square(solu)) < tf.square(trustradius_out))
        #atboundary_out = tf.logical_not(usesolu)
        
        #print(solu.shape)
        #print(pl.shape)
        pl = tf.reshape(pl,[-1])
        #pl = tf.where(usesolu,solu,pl)
        #pl =tf.Print(pl,[pl])
        
        #predicted_reduction_out = -dfval_quad(pl)
        #var_out = var + pl
        #var_out = pl[0]*d0 + pl[1]*d1
        
        return pl
        
        #return [var_out,predicted_reduction_out,atboundary_out,grad]

      
      
      sol = tf.cond(usesolu, lambda: solu, lambda: build_iter())
      
      
      predicted_reduction_out = -dfval_quad(sol)
      
      return [var+sol, predicted_reduction_out, tf.logical_not(usesolu), grad]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]

class TrustNCG:
    
  def initialize(self, loss, var, initialtrustradius = 1., doSR1 = False):
    self.doSR1 = doSR1
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(True, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    if doSR1:
      #self.B = tf.Variable(tf.eye(num_rows=var.shape[0],num_columns=var.shape[0],dtype=var.dtype),trainable=False)
      self.B = tf.Variable(tf.zeros([var.shape[0],var.shape[0]],dtype=var.dtype),trainable=False)
      self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    alist = []
    alist.append(tf.assign(self.trustradius,initialtrustradius))
    alist.append(tf.assign(self.loss_old, loss))
    alist.append(tf.assign(self.predicted_reduction, 0.))
    alist.append(tf.assign(self.var_old, var))
    alist.append(tf.assign(self.atboundary_old,False))
    alist.append(tf.assign(self.doiter_old,False))
    alist.append(tf.assign(self.isfirstiter,True))
    if doSR1:
      alist.append(tf.assign(self.B,tf.zeros([var.shape[0],var.shape[0]],dtype=var.dtype)))
      alist.append(tf.assign(self.grad_old,self.grad))

    return tf.group(alist)
  
  def setHessApprox(self,b,sess):
    self.B.load(b,sess)
  
  def minimize(self, loss, var):
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    edmtol = 1e-8
    #edmtol = 0.
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    #isnull = tf.logical_and(tf.equal(actual_reduction,0.),tf.equal(self.predicted_reduction,0.))
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    
    if self.doSR1:

      grad = self.grad
      y = tf.reshape(grad - self.grad_old,[-1,1])
      dx = tf.reshape(var - self.var_old,[-1,1])
      Bx = tf.matmul(self.B,dx)
      dyBx = y - Bx
      den = tf.matmul(dyBx,dx,transpose_a=True)
      deltaB = tf.matmul(dyBx,dyBx,transpose_b=True)/den
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.less(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      deltaB = tf.where(dentest,tf.zeros_like(deltaB),deltaB)
      deltaB = tf.where(self.doiter_old, deltaB, tf.zeros_like(deltaB))      
      B = self.B + deltaB
      
    isconvergedxtol = trustradius_out < xtol
    isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,0.15),tf.logical_not(isconverged))
    
    #trustradius_out = tf.Print(trustradius_out,[self.loss_old, loss, actual_reduction,self.predicted_reduction,isnull,rho,trustradius_out,isconvergedxtol,isconvergededmtol,doiter])
    
    def build_loop():
      if self.doSR1:
        grad = self.grad
      else:
        grad = tf.gradients(loss,var, gate_gradients=True)[0]
      
      def hesspexact(v):
        return tf.gradients(grad*tf.stop_gradient(v),var, gate_gradients=True)[0]
        
      def hesspapprox(v):
        return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])
        
      if self.doSR1:
        hessp = hesspapprox
      else:
        hessp = hesspexact
        
      def dfval_quad(v):
        return tf.reduce_sum(grad*v) + 0.5*tf.reduce_sum(hessp(v)*v)
      
      def cond(z,r,d,atboundary,attol):
        return tf.logical_and(tf.logical_not(atboundary),tf.logical_not(attol))
      
      def body(z,r,d,atboundary,attol):
        #CGSteihaugSubproblem adapted from scipy newton cg implementation
        #(This is algorithm (7.2) of Nocedal and Wright 2nd edition.)
        
        #find trust region boundaries along search direction
        a = tf.reduce_sum(tf.square(d))
        b = 2.*tf.reduce_sum(z*d)
        c = tf.reduce_sum(tf.square(z)) - tf.square(trustradius_out)
        sqrt_discriminant = tf.sqrt(tf.square(b) - 4.*a*c)
        sign = tf.where(b >= 0., tf.ones_like(b),-tf.ones_like(b))
        aux = b + sign*sqrt_discriminant
        tap = -0.5*aux/a
        tbp = -2.*c/aux
        tasort = tap<tbp
        ta = tf.where(tasort, tap, tbp)
        tb = tf.where(tasort, tbp, tap)
        pa = z + ta*d
        pb = z + tb*d
        
        #compute curvature along search direction
        Bd = hessp(d)
        dBd = tf.reduce_sum(d*Bd)
        
        #compute quadratic approximation to function value at boundaries
        #(do it conditionally, since the cost is non-trivial)
        negativecurvature = tf.less_equal(dBd,0.)
        fvalsbound = tf.cond(negativecurvature, lambda : [dfval_quad(pa),dfval_quad(pb)], lambda : [tf.zeros_like(loss),tf.zeros_like(loss)])
        negativebound = tf.where(tf.less(fvalsbound[0],fvalsbound[1]),pa,pb)
                
        r_squared = tf.reduce_sum(tf.square(r))
        alpha = r_squared / dBd
        z_prop = z + alpha * d
        
        z_prop_norm = tf.sqrt(tf.reduce_sum(tf.square(z_prop)))
        atboundary_next = tf.logical_or(negativecurvature, tf.greater(z_prop_norm,trustradius_out))
                
        z_next = tf.where(negativecurvature, negativebound, tf.where(atboundary_next,pb,z_prop))
        
        r_next = r + alpha * Bd
        r_next_squared = tf.reduce_sum(tf.square(r_next))
        attol_next = tf.less(tf.sqrt(r_next_squared),tolerance)

        beta_next = r_next_squared / r_squared
        d_next = -r_next + beta_next * d

        return [z_next, r_next, d_next, atboundary_next, attol_next]
      
      z = tf.zeros_like(var)
      r = grad
      d = -grad
      atboundary = tf.constant(False)
      attol = tf.constant(False)
      
      gradmag = tf.sqrt(tf.reduce_sum(tf.square(grad)))
      tolerance = tf.minimum(0.5*tf.ones_like(loss), tf.sqrt(gradmag)) * gradmag
      
      loop_vars = [z,r,d,atboundary,attol]
      #print([z.shape,r.shape,d.shape,atboundary.shape,attol.shape])

      #solve subproblem iteratively
      z_out,r_out,d_out,atboundary_out,attol_out = tf.while_loop(cond, body, loop_vars, parallel_iterations=64, back_prop=False, maximum_iterations=20.*var.shape[0])
    
      #print([z_out.shape,r_out.shape,d_out.shape,atboundary_out.shape, attol_out.shape])
    
      predicted_reduction_out = -dfval_quad(z_out)
      var_out = var + z_out
      return [var_out,predicted_reduction_out,atboundary_out,grad]
    

    loopout = tf.cond(doiter, lambda: build_loop(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
    
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      #oldvarassign = tf.Print(var_out,[],message="oldvarassign")
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      if self.doSR1:
        alist.append(tf.assign(self.B,B))
        alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]

class SR1Mod(SR1):
    def __init__(self, min_denominator=1e-8, init_scale='auto'):
      self.initval = None
      super(SR1Mod, self).__init__(min_denominator, init_scale)

    def setInitVal(self, h):
      self.initval = h
      self.init_scale = 1.
      
    def setInitxg(self, xgval):
      self.xval = xgval[0]
      self.gval = xgval[1]
      
    def setHessp(self,fun):
      self.hessp = fun
      
    def setHess(self,fun):
      self.hess = fun

    def initialize(self, n, approx_type):
        """Initialize internal matrix.
        Allocate internal memory for storing and updating
        the Hessian or its inverse.
        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type not in ('hess', 'inv_hess'):
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        # Create matrix
        if self.approx_type == 'hess':
            if self.initval is None:
              self.B = np.eye(n, dtype=float)
            else:
              self.B = self.initval
        else:
            if self.initval is None:
              self.H = np.eye(n, dtype=float)
            else:
              self.H = self.initval

    #def _update_implementation(self, delta_x, delta_grad):
      #dgradpred = np.matmul(self.B,delta_x)
      #dgradactual = self.hessp(self.xval,delta_x)
      
      #self.xval += delta_x
      #self.gval += delta_grad
      
      #dgrad = np.max(np.abs((dgradpred-dgradactual)/dgradactual))
      #if dgrad > 1000.:
        #print("recomputing hessian, dgrad = %f" % dgrad)
        #self.B = self.hess(self.xval)
      #else:
        #super(SR1Mod, self)._update_implementation(delta_x, delta_grad)
      

      #if self.approx_type == 'hess' and not self.hess is None:
        #dgradpred = np.matmul(self.B,delta_x)
        #dgrad = np.max(np.abs(0.5*(delta_grad - dgradpred)/(delta_grad+dgradpred)))
        ##dgrad = np.max(np.abs((delta_grad - dgradpred)/self.gval))
        ##dgrad = np.max(np.abs(delta_grad/np.matmul(self.B,delta_x)))
        ##if dgrad > 100.:
        #if True:
          #print("recomputing hessian, dgrad = %f" % dgrad)
          #self.B = self.hess(self.xval)
        #else:
          #super(SR1Mod, self)._update_implementation(delta_x, delta_grad)
      #else:
        #super(SR1Mod, self)._update_implementation(delta_x, delta_grad)
            
    #def _update_implementation(self, delta_x, delta_grad):
      #self.xval += delta_x
      #self.gval += delta_grad
      #if self.approx_type == 'hess' and not self.hessp is None:
        #super(SR1Mod, self)._update_implementation(self.gval, self.hessp(self.xval,self.gval))
        #super(SR1Mod, self)._update_implementation(delta_x, self.hessp(self.xval,delta_x))
        #mineig = np.amin(np.linalg.eigvalsh(self.B))
        #if mineig<0.:
          #dxalt = np.linalg.solve(self.B-mineig*np.eye(self.B.shape[0],dtype=self.B.dtype),self.gval)
          #super(SR1Mod, self)._update_implementation(dxalt, self.hessp(self.xval,dxalt))
        #dxalt2 = np.linalg.solve(self.B,self.gval)
        #super(SR1Mod, self)._update_implementation(dxalt2, self.hessp(self.xval,dxalt2))
        ##super(SR1Mod, self)._update_implementation(delta_x, delta_grad)
      #else:
        #super(SR1Mod, self)._update_implementation(delta_x, delta_grad)
        

class JacobianCompute:
  def __init__(self, ys, xs):
    self.nrows = xs.shape[0]
    self.x = xs
    self.rowidx = tf.placeholder(tf.int32, shape=[])
    self.jacrow = tf.gradients(ys[self.rowidx],xs)[0]
    
  def compute(self,sess,xval=None):
    jacrows = []
    feed_dict = {}
    if not xval is None:
      feed_dict[self.x] = xval
    for irow in range(self.nrows):
      feed_dict[self.rowidx] = irow
      jacrows.append(sess.run(self.jacrow, feed_dict = feed_dict))
    return np.stack(jacrows,axis=0)

def jacobian(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
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
  n = array_ops.size(x)
  loop_vars = [
      array_ops.constant(0, dtypes.int32),
      tensor_array_ops.TensorArray(x.dtype, n)
  ]
  # Iterate over all elements of the gradient and compute second order
  # derivatives.
  _, hessian = control_flow_ops.while_loop(
      lambda j, _: j < n,
      lambda j, result: (j + 1,
                          result.write(j, tf.gradients(gradient[j], x)[0])),
      loop_vars
  )

  _shape = array_ops.shape(x)
  _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                        array_ops.concat((_shape, _shape), 0))
  #hessians.append(_reshaped_hessian)
  return _reshaped_hessian

class ScipyTROptimizerInterface(ExternalOptimizerInterface):

  _DEFAULT_METHOD = 'trust-constr'


  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
    hess = optimizer_kwargs.pop('hess', SR1())
    hessp = optimizer_kwargs.pop('hessp', None)

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
      isnull = np.all(np.equal(lb,-np.inf)) and np.all(np.equal(ub,np.inf))
      if not isnull:
        constraints.append(LinearConstraint(np.eye(initial_val.shape[0],dtype=initial_val.dtype),lb,ub,keep_feasible=True))

    minimize_args = [loss_grad_func, initial_val]
    minimize_kwargs = {
        'jac': True,
        'hess' : hess,
        #'hess' : None,
        #'hessp' : hessp,
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
