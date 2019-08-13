import numpy as np
import tensorflow as tf
import math

from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.framework import function

class SR1TrustExact:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.grad_old_mag = tf.sqrt(tf.reduce_sum(tf.square(self.grad_old)))
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.UT = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.e = tf.Variable(tf.ones_like(var),trainable=False)
    self.e0 = self.e[0]
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None:
      e,U = tf.self_adjoint_eig(B)
      UT = tf.transpose(U)
      alist.append(tf.assign(self.e,e))
      alist.append(tf.assign(self.UT,UT))
    return tf.group(alist)
  
  def minimize(self, loss, var, grad = None):
    #TODO, consider replacing gather_nd with gather where appropriate (maybe faster?)
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #xtol = 0.
    #eta = 0.
    eta = 0.15
    #eta = 1e-3
    
    #compute ratio of actual reduction in loss function to that
    #predicted from quadratic approximation
    #in order to decide whether to reverse the previous step
    #and whether to enlarge or shrink the trust region
    actual_reduction = self.loss_old - loss
    
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    #trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    def doSR1Scaling(Bin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      return (B,H,tf.constant(False))
    
    #n.b. this has a substantially different form from the usual SR 1 update
    #since we are directly updating the eigenvalue-eigenvector decomposition.
    #The actual hessian approximation is never stored (but memory requirements
    #are similar since the full set of eigenvectors is stored)
    def doSR1Update(ein,UTin,yin,dxin):
      #compute quantities which define the rank 1 update
      #and numerical test to determine whether to perform
      #the update
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      ecol = tf.reshape(ein,[-1,1])
      
      UTdx = tf.matmul(UTin, dx)
      UTy = tf.matmul(UTin,y)
      den = tf.matmul(y,dx,transpose_a=True) - tf.matmul(UTdx,ecol*UTdx,transpose_a=True)
      dyBx =  UTy - ecol*UTdx
      dyBxnormsq = tf.reduce_sum(tf.square(dyBx))
      dyBxnorm = tf.sqrt(dyBxnormsq)
      dxnorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))
      dennorm = dxnorm*dyBxnorm
      absden = tf.abs(den)
      dentest = tf.less(absden,1e-8*dennorm) | tf.equal(tf.reshape(absden,[]),0.)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      
      
      def doUpdate():
        #compute update in the form
        #B->B + rho zz^T with |z|=1
        z = dyBx/dyBxnorm
        signedrho = dyBxnormsq/den
        signedrho = tf.reshape(signedrho,[])
        #signedrho = tf.Print(signedrho,[signedrho],message="signedrho")
        rho = tf.abs(signedrho)

        flipsign = signedrho < 0.
        
        #in case rho<0, reverse order of eigenvalues and eigenvectors and flip signs
        #to ensure consistent ordering
        #z needs to be reversed as well since it was already computed with the original ordering
        einalt = -tf.reverse(ein,axis=(0,))
        UTinalt = tf.reverse(UTin,axis=(0,))
        zalt = tf.reverse(z,axis=(0,))
        
        estart = tf.where(flipsign,einalt,ein)
        UTstart = tf.where(flipsign,UTinalt,UTin)
        z = tf.where(flipsign,zalt,z)
        
        #estart = tf.Print(estart,[estart],message="estart",summarize=10000)
        
        #deflation in case of repeated eigenvalues
        estartn1 = estart[:-1]
        estart1 = estart[1:]
        ischange = tf.logical_not(tf.equal(estartn1,estart1))
        isfirst = tf.concat([[True],ischange],axis=0)
        islast = tf.concat([ischange,[True]],axis=0)
        islast1 = islast[1:]
        issingle1 = tf.logical_and(ischange,islast1)
        issingle = tf.concat([ischange[0:1], issingle1],axis=0)
        isrep = tf.logical_not(issingle)
        isfirstrep = tf.logical_and(isfirst,isrep)
        islastrep = tf.logical_and(islast,isrep)
        
        firstidxsrep = tf.where(isfirstrep)
        lastidxsrep = tf.where(islastrep)
        rrep = lastidxsrep - firstidxsrep + 1
        rrep = tf.reshape(rrep,[-1])
        
        repidxs = tf.where(isrep)
        firstidxs = tf.where(isfirst)
        lastidxs = tf.where(islast)
        r = lastidxs - firstidxs + 1
        r = tf.reshape(r,[-1])
        isrepunique = r > 1
        uniquerepidxs = tf.where(isrepunique)
        nonlastidxs = tf.where(tf.logical_not(islast))
                
        zflat = tf.reshape(z,[-1])
        
        uniqueidxs = tf.cumsum(tf.cast(islast,tf.int32),exclusive=True)
        xisq2 = tf.segment_sum(tf.square(zflat),uniqueidxs)
        
        xisqrep = tf.gather_nd(xisq2,uniquerepidxs)
        abszrep = tf.sqrt(xisqrep)
        
        #TODO (maybe) skip inflation entirely in case there are no repeating eigenvalues
        
        #loop over sets of repeated eigenvalues in order to perform the necessary
        #transformation of the eigenvectors and rank 1 update
        arrsize = tf.shape(firstidxsrep)[0]
        arr0 = tf.TensorArray(var.dtype,size=arrsize,infer_shape=False,element_shape=[None,var.shape[0]])
        deflate_var_list = [arr0, tf.constant(0,dtype=tf.int32)]
        def deflate_cond(arr,j):
          return j<arrsize
        def deflate_body(arr,j):
          size = rrep[j]
          startidx = tf.reshape(firstidxsrep[j],[])
          endidx = startidx + size
          zsub = zflat[startidx:endidx]
          UTsub = UTstart[startidx:endidx]
          magzsub = abszrep[j]
          en = tf.one_hot(size-1,depth=tf.cast(size,tf.int32),dtype=zsub.dtype)
          #this is the vector which implicitly defines the Householder transformation matrix
          v = zsub/magzsub + en
          v = v/tf.sqrt(tf.reduce_sum(tf.square(v)))
          v = tf.reshape(v,[-1,1])
          #protection for v~=0 case (when zsub~=-en), then no transformation is needed
          nullv = tf.reduce_all(tf.equal(tf.sign(zsub),-en))
          v = tf.where(nullv,tf.zeros_like(v),v)
          UTbarsub = UTsub - 2.*tf.matmul(v,tf.matmul(v,UTsub,transpose_a=True))
          arr = arr.write(j,UTbarsub)
          return (arr, j+1)
        
        UTbararr,j = tf.while_loop(deflate_cond,deflate_body,deflate_var_list, parallel_iterations=64, back_prop=False)
        UTbarrep = UTbararr.concat()
        
        #reassemble transformed eigenvectors and update vector
        #now z=0 for repeated eigenvectors except for the last instance
        UTbar = tf.where(issingle, UTstart, tf.scatter_nd(repidxs,UTbarrep, shape=UTstart.shape))
        zbar = tf.where(issingle, zflat, tf.scatter_nd(lastidxsrep,-abszrep, shape=zflat.shape))
                
        #construct deflated system consisting of unique eigenvalues only
        UT1 = tf.gather_nd(UTbar,nonlastidxs)     
        UT2 = tf.gather_nd(UTbar,lastidxs)
        e1 = tf.gather_nd(estart,nonlastidxs)
        d = tf.gather_nd(estart,lastidxs)
        z2 = tf.gather_nd(zbar,lastidxs)
        
        #TODO, check if this reshape is really needed)
        xisq = tf.reshape(xisq2,[-1])
        
        #compute quantities needed for eigenvalue update
        dnorm = d/rho
        dnormn1 = dnorm[:-1]
        dnorm1 = dnorm[1:]
        dnorm1f = tf.concat((dnorm1,[1.+rho]),axis=0)
        dn1 = d[:-1]
        d1 = d[1:]
        deltan1 = dnorm1 - dnormn1
        delta = tf.concat([deltan1,[1.]],axis=0)
        deltaim1 = tf.concat([[1.],deltan1],axis=0)
        rdeltan1 = rho/(d1-dn1)
        rdelta = tf.concat([rdeltan1,[1.]],axis=0)
        rdelta2 = tf.square(rdelta)
        xisqn1 = tf.reshape(xisq[:-1],[-1])
        xisq1n1 = tf.reshape(xisq[1:],[-1])
        xisq1 = tf.concat([xisq1n1, [0.]],axis=0)
        
        dnormi = tf.reshape(dnorm,[-1,1])
        dnormj = tf.reshape(dnorm,[1,-1])
        deltam = dnormj - dnormi
        deltam2 = tf.concat([deltam[1:],deltam[-1:]-tf.ones_like(dnormj)],axis=0)
                        
        t0 = tf.zeros_like(d)

        nupper = tf.minimum(1,tf.shape(d)[0])
        deltambool = tf.ones_like(deltam,dtype=tf.bool)
        deltamones = tf.ones_like(deltam)
        deltamask = tf.matrix_band_part(deltambool,tf.zeros_like(nupper),nupper)
        
        deltamaskdiag = tf.matrix_band_part(deltambool,0,0)
        #deltamaskmid = tf.matrix_band_part(deltambool,1,0)
        deltamasklow  = tf.matrix_band_part(deltambool,-1,0) & tf.logical_not(deltamaskdiag)
        #deltamaskhigh  = tf.matrix_band_part(deltambool,0,-1) & tf.logical_not(deltamaskdiag)
        deltamaskhigh  = tf.matrix_band_part(deltambool,0,-1)
                
        unconverged0 = tf.ones_like(d,dtype=tf.bool)
                  
        loop_vars = [t0,unconverged0,tf.constant(0),t0]
        def cond(yd,unconverged,j,phi):
          return tf.reduce_any(unconverged) & (j<50)
        
        #solution to eigenvalue update in terms of t = (dout - din)/rho
        def body(t,unconverged,j,phi):
          dt = delta - t
          
          t2 = tf.square(t)
          t3 = t*t2
          
          dt2 = tf.square(dt)
          dt3 = dt*dt2
          
          ti = tf.reshape(t,[-1,1])
          frden = tf.reciprocal(deltam-ti)
          #exclude j=i,i+1 terms
          frden = tf.where(deltamask,tf.zeros_like(frden),frden)
          issingular = tf.reduce_any(tf.is_inf(frden),axis=-1)
          xisqj = tf.reshape(xisq,[1,-1])
          s0arg = xisqj*frden
          s1arg = s0arg*frden
          s2arg = s1arg*frden
          
          s0 = tf.reduce_sum(s0arg, axis=-1)
          s1 = tf.reduce_sum(s1arg, axis=-1)
          s2 = tf.reduce_sum(s2arg, axis=-1)
          
          #function value is not actually used, but computed
          #for diagnostic purposes only
          phi = t*dt*(1.+s0) - dt*xisq + t*xisq1
          phi = tf.where(tf.is_nan(phi),tf.zeros_like(phi),phi)
          magw = tf.sqrt(tf.reduce_sum(tf.square(phi)))

          cg = (dt3*s1 + t*dt3*s2)*rdelta + xisq1
          bg = (-t3*s1 + t3*dt*s2)*rdelta - xisq
          ag = 1. + s0 + (2.*t - delta)*s1 - t*dt*s2
          
          a = ag
          b = -delta*ag + bg - cg
          c = t3*s1 - t3*dt*s2 + delta*xisq
          
          told = t
          
          #use three different forms of the quadratic formula depending on the sign of b
          #in order to avoid cancellations/roundoff or nan
          sarg = tf.square(b) - 4.*a*c
          sarg = tf.maximum(sarg,0.)
          s = tf.sqrt(sarg)
          tnom = -0.5*(b+s)/a
          talt = -2.*c/(b-s)
          #with protection for roundoff error which could make -c/a negative
          s2arg = -c/a
          s2arg = tf.maximum(s2arg,0.)
          ts = tf.sqrt(s2arg)
          
          signb = tf.sign(b)
          t = tf.where(tf.equal(signb,0), ts,tf.where(signb>0, tnom, talt))
          
          #protection for singular case
          t = tf.where(issingular, tf.zeros_like(t), t)
          
          #roundoff errors could produce solutions out of bounds for t->delta
          #but these will be discarded later anyways in favor of the alternate solution
          t = tf.maximum(t,0.)
          t = tf.minimum(t,delta)
          
          #when individual eigenvalues have converged we mark them as such
          #but simply keep iterating on the full vector, since any efficiency
          #gains from partially stopping and chopping up the vector would likely incur
          #more overhead, especially on GPU
          tadvancing = t > told
          #if t>0.5*delta we won't use this solution anyways, so we don't care if its converged or not
          #(but leave an extra margin here to avoid any possible numerical issues)
          tunsaturated = t < 0.6*delta
          unconverged = unconverged & tadvancing & tunsaturated
                                 
          #t = tf.Print(t,[magw],message="magw")
          
          return (t,unconverged,j+1,phi)
          
          
        t,unconverged,j,phi = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        
        tassert = tf.Assert(tf.reduce_all((t>=0.) & (t<=delta)),[t],summarize=10000)        
        with tf.control_dependencies([tassert]):
          t = tf.identity(t)
        
        #solution to eigenvalue update in terms of delta-t
        def body2(t,unconverged,j,phi):
          dt = delta - t
          
          t2 = tf.square(t)
          t3 = t*t2
          
          dt2 = tf.square(dt)
          dt3 = dt*dt2
          
          ti = tf.reshape(t,[-1,1])
          frden = tf.reciprocal(deltam2+ti)
          #exclude j=i,i+1 terms
          frden = tf.where(deltamask,tf.zeros_like(frden),frden)
          issingular = tf.reduce_any(tf.is_inf(frden),axis=-1)
          xisqj = tf.reshape(xisq,[1,-1])
          s0arg = xisqj*frden
          s1arg = s0arg*frden
          s2arg = s1arg*frden
          
          s0 = tf.reduce_sum(s0arg, axis=-1)
          s1 = tf.reduce_sum(s1arg, axis=-1)
          s2 = tf.reduce_sum(s2arg, axis=-1)
          
          phi = t*dt*(1.+s0) - t*xisq + dt*xisq1
          phi = tf.where(tf.is_nan(phi),tf.zeros_like(phi),phi)
          #phi = 1.+s0 - xisq/dt + xisq1/t
          magw = tf.sqrt(tf.reduce_sum(tf.square(phi)))

          cg = (-dt3*s1 + t*dt3*s2)*rdelta - xisq
          bg = (t3*s1 + t3*dt*s2)*rdelta + xisq1
          ag = 1. + s0 - (2.*t - delta)*s1 - t*dt*s2
          
          a = ag
          b = -delta*ag + bg - cg
          c = -t3*s1 - t3*dt*s2 - delta*xisq1
          
          told = t
          
          #use two different forms of the quadratic formula depending on the sign of b
          #in order to avoid cancellations/roundoff
          sarg = tf.square(b) - 4.*a*c
          sarg = tf.maximum(sarg,0.)
          s = tf.sqrt(sarg)
          tnom = -0.5*(b-s)/a
          talt = -2.*c/(b+s)
          #with protection for roundoff error which could make -c/a negative
          s2arg = -c/a
          s2arg = tf.maximum(s2arg,0.)
          ts = tf.sqrt(s2arg)
          
          signb = tf.sign(b)
          t = tf.where(tf.equal(signb,0), ts,tf.where(signb>0, talt, tnom))
          
          #protection for singular case
          t = tf.where(issingular, tf.zeros_like(t), t)
          
          #roundoff errors could produce solutions out of bounds for t->delta
          #but these will be discarded later anyways in favor of the alternate solution
          t = tf.maximum(t,0.)
          t = tf.minimum(t,delta)
          
          #when individual eigenvalues have converged we mark them as such
          #but simply keep iterating on the full vector, since any efficiency
          #gains from partially stopping and chopping up the vector would likely incur
          #more overhead, especially on GPU
          tadvancing = t > told
          #if t>0.5*delta we won't use this solution anyways, so we don't care if its converged or not
          #(but leave an extra margin here to avoid any possible numerical issues)
          tunsaturated = t < 0.6*delta
          #tunsaturated = tadvancing
          unconverged = unconverged & tadvancing & tunsaturated

          #t = tf.Print(t,[magw],message="magw2")
          
          return (t,unconverged,j+1,phi)
        
        dt,unconverged2,j2,phi2 = tf.while_loop(cond, body2, loop_vars, parallel_iterations=1, back_prop=False)
        
        t2assert = tf.Assert(tf.reduce_all((dt>=0.) & (dt<=delta)),[dt],summarize=10000)        
        with tf.control_dependencies([t2assert]):
          dt = tf.identity(dt)
                
        d1 = tf.concat([d[1:],d[-1:]+rho],axis=0)
        dout = d + rho*t
        dout2 = d1 - rho*dt
        dnormout = dnorm + t
        dnormout2 = dnorm1f - dt
        #choose solution with higher numerical precision
        tswitch = t <= dt
        dout = tf.where(tswitch,dout, dout2)
        dnormout = tf.where(tswitch,dnormout,dnormout2)
        phiout = tf.where(tswitch,phi,phi2)
        magphi = tf.reduce_sum(tf.square(phiout))
        
        #t = tf.where(tswitch,t,delta-dt)
        #dt = tf.where(tswitch,delta-t,dt)
        #dtim1 = tf.concat(([1.],dt[:-1]),axis=0)

        #dout = tf.Print(dout,[magphi],message="magphi")
        
        #now compute eigenvectors, with rows of this matrix constructed
        #from the solution with the higher numerical precision
        ti = tf.reshape(t,[-1,1])
        dti = tf.reshape(dt,[-1,1])
        D = deltam - ti
        D2 = deltam2 + dti
        Dswitch = tf.reshape(tswitch,[-1,1]) | tf.zeros_like(D, dtype=tf.bool)

        
        D = tf.where(Dswitch,D,D2)
        Dinv = tf.reciprocal(tf.where(tf.equal(D,0.),tf.ones_like(D),D))
        #Dinv = tf.reciprocal(deltam - ti)        
        #Dinv2 = tf.reciprocal(deltam2 + dti)
        
        #Dinv = tf.where(Dinvswitch,Dinv,Dinv2)

        #recompute z as in tr916 to maintain eigenvectors orthogonality
        
        #prodmnum = 
        
        #prodmnum = tf.reshape(dnormout,[1,-1])-tf.reshape(dnorm,[-1,1])
        #prodmnum = tf.reshape(dout,[1,-1])/rho-tf.reshape(dnorm,[-1,1])
        
        deltam1 = deltam[:,1:]
        
        deltam1num = tf.concat([deltam1,1.+deltam[:,-1:]],axis=-1)
        deltam1den = tf.concat([deltam1,tf.ones_like(deltam[:,-1:])],axis=-1)
        
        prodmnum = deltam + tf.reshape(t,[1,-1])
        prodmnum2 = deltam1num - tf.reshape(dt,[1,-1])
        DswitchT = tf.reshape(tswitch,[1,-1]) | tf.zeros_like(prodmnum, dtype=tf.bool)
        prodmnum = tf.where(DswitchT,prodmnum,prodmnum2)
        
        #prodmnumlow = tf.where(deltamasklow,prodmnum,tf.ones_like(prodmnum))
        #prodmnumhigh = tf.where(deltamaskhigh,prodmnum,tf.ones_like(prodmnum))
        
        prodmdenlow = tf.where(deltamasklow,deltam,tf.zeros_like(deltam))
        prodmdenhigh = tf.where(deltamaskhigh,deltam1den,tf.zeros_like(deltam1den))
        
        prodmden = prodmdenlow + prodmdenhigh
        
        nullden = tf.equal(prodmden,0.)
        prodr = tf.where(nullden,tf.ones_like(prodmnum),prodmnum*tf.reciprocal(tf.where(nullden,tf.ones_like(prodmden),prodmden)))
        prod = tf.reduce_prod(prodr,axis=-1)
        absztilde = tf.sqrt(prod)
        ztilde = tf.where(tf.greater_equal(z2,0.),absztilde,-absztilde)

        
        #nulldenlow = tf.equal(prodmdenlow,0.)
        #rlow = tf.where(nulldenlow,tf.ones_like(prodmnumlow),prodmnumlow*tf.reciprocal(tf.where(nulldenlow,tf.ones_like(prodmdenlow),prodmdenlow)))
        #prodlow = tf.reduce_prod(rlow,axis=-1)

        #sliceidx = tf.minimum(tf.shape(deltam)[1],1)

        #nulldenhigh = tf.equal(prodmdenhigh,0.)
        #rhigh = tf.where(nulldenhigh,tf.ones_like(prodmnumhigh),prodmnumhigh*tf.reciprocal(tf.where(nulldenhigh,tf.ones_like(prodmdenhigh),prodmdenhigh)))
        ##prodhigh = tf.reduce_prod(rhigh[:,:-1],axis=-1)
        #prodhigh = tf.reduce_prod(rhigh,axis=-1)
        
        
        ##prodhigh = tf.reduce_prod(prodmnumhigh/prodmdenhigh,axis=-1)
        ##proddiag = t/delta
        ##proddiag1 = dtim1/deltaim1
        
        ##ztilde = tf.sqrt(prodlow*proddiag1*proddiag*prodhigh)
        ##ztilde = tf.sqrt(prodlow*prodhigh*tf.reduce_sum(prodmnum[:,-1:],axis=-1))
        ##ztilde = tf.sqrt(prodlow*prodhigh*prodmnum[:,-1])*tf.sign(z2)
        
        #absztilde = tf.sqrt(prodlow*prodhigh)
        
        #tj = tf.reshape(t,[1,-1])
        #ztilde = tf.sqrt(t*tf.reduce_prod(tf.where(deltamaskdiag,tf.ones_like(deltam),1. + tj/tf.where(deltamaskdiag,tf.ones_like(deltam),deltam)),axis=-1))
        
        #prodmnum = tf.reshape(dout,[1,-1]) - tf.reshape(d,[-1,1])
        #prodmdenlow = tf.reshape(d,[1,-1]) - tf.reshape(d,[-1,1])
        #ztilde = tf.sqrt(tf.reduce_prod(prodmnum,axis=-1)/tf.reduce_prod(tf.where(deltamaskdiag,tf.ones_like(prodmdenlow),prodmdenlow),axis=-1)/rho)
        
        #rhigh = prodmnumhigh/prodmdenhigh
        #ztilde = tf.Print(ztilde,[rhigh],message="rhigh",summarize=10000)
        #ztilde = tf.Print(ztilde,[t],message="t",summarize=1000)
        #ztilde = tf.Print(ztilde,[dt],message="dt",summarize=1000)
        #ztilde = tf.Print(ztilde,[prodlow],message="prodlow",summarize=1000)
        #ztilde = tf.Print(ztilde,[prodhigh],message="prodhigh",summarize=1000)
        #ztilde = tf.Print(ztilde,[proddiag],message="proddiag",summarize=1000)
        #ztilde = tf.Print(ztilde,[proddiag1],message="proddiag1",summarize=1000)
        
        #ztilde = tf.Print(ztilde,[z2],message="z2",summarize=1000)
        #ztilde = tf.Print(ztilde,[ztilde],message="ztilde",summarize=1000)
        #ztilde = tf.Print(ztilde,[(ztilde-tf.abs(z2))/tf.abs(z2)],message="delta z",summarize=1000)

        
        #Dinvz = Dinv*tf.reshape(z2,[1,-1])
        Dinvz = Dinv*tf.reshape(ztilde,[1,-1])
        Dinvzmag = tf.sqrt(tf.reduce_sum(tf.square(Dinvz),axis=-1,keepdims=True))
        Dinvz = Dinvz/Dinvzmag
        
        #n.b. this is the most expensive operation (matrix-matrix multiplication to compute the updated eigenvectors)
        UT2out = tf.matmul(Dinvz,UT2)
        
        #protections for t=0 or t=delta cases
        #if t=0 the eigenvector is unchanged
        #if t=delta then the i+1st eigenvector is shifted
        #to the ith position
        UT21 = tf.concat([UT2[1:],UT2[-1:]],axis=0)
        #tnull = tf.equal(ti,0.)
        #dtnull = tf.equal(dti,0.)        
        tnull = tf.equal(ti,0.) & tswitch
        dtnull = tf.equal(dti,0.) & tf.logical_not(tswitch)
        UT2false = tf.zeros_like(UT2,dtype=tf.bool)
        tnullm = tf.logical_or(UT2false,tnull)
        dtnullm = tf.logical_or(UT2false,dtnull)
        
        #UT2out = tf.where(dtnullm,UT21,UT2out)
        #UT2out = tf.where(tnullm,UT2,UT2out)
                        
        #now put everything back together
        #eigenvalues are still guaranteed to be sorted
        eout = tf.scatter_nd(lastidxs,dout,estart.shape) + tf.scatter_nd(nonlastidxs,e1,estart.shape)
        UTout = tf.scatter_nd(lastidxs,UT2out,UTstart.shape) + tf.scatter_nd(nonlastidxs,UT1,UTstart.shape)
        
        #restore correct order and signs if necessary
        eoutalt = -tf.reverse(eout,axis=(0,))
        UToutalt = tf.reverse(UTout,axis=(0,))
        
        eout = tf.where(flipsign,eoutalt,eout)
        UTout = tf.where(flipsign,UToutalt,UTout)
                
        return (eout,UTout)
      
      e,UT = tf.cond(dentest, lambda: (ein,UTin), doUpdate)
      
      return (e,UT)
    
    esec = self.e
    UTsec = self.UT
    
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    esec,UTsec = tf.cond(self.doiter_old, lambda: doSR1Update(esec,UTsec,dgrad,dx), lambda: (esec,UTsec))  
    
    isconvergedxtol = trustradius_out < xtol
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    
    def build_sol():

      lam = esec
      UT = UTsec
      
      gradcol = tf.reshape(grad,[-1,1])
      
      #projection of gradient onto eigenvectors
      a = tf.matmul(UT, gradcol)
      a = tf.reshape(a,[-1])
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      asq = tf.square(a)
      
      #deal with null gradient components and repeated eigenvectors
      lamn1 = lam[:-1]
      lam1 = lam[1:]
      ischange = tf.logical_not(tf.equal(lamn1,lam1))
      islast = tf.concat([ischange,[True]],axis=0)
      lastidxs = tf.where(islast)
      uniqueidx = tf.cumsum(tf.cast(islast,tf.int32),exclusive=True)
      uniqueasq = tf.segment_sum(asq,uniqueidx)
      uniquelam = tf.gather_nd(lam,lastidxs)
      
      abarindices = tf.where(uniqueasq)
      abarsq = tf.gather_nd(uniqueasq,abarindices)
      lambar = tf.gather_nd(uniquelam,abarindices)
      
      #abar = tf.sqrt(abarsq)
      abarmag = tf.sqrt(abarsq)
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))        
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      #TODO, add handling of additional cases here (singular and "hard" cases)
      
      phisigma0 = phif(sigma0)
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      
      def sigma():
        #tol = 1e-10
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit,phiprimeinit = phiphiprime(sigmainit)
                
        loop_vars = [sigmainit, phiinit,phiprimeinit, tf.constant(True), tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,phiprime,unconverged,j):
          return (unconverged) & (j<maxiter)
        
        def body(sigma,phi,phiprime,unconverged,j):   
          sigmaout = sigma - phi/phiprime
          phiout, phiprimeout = phiphiprime(sigmaout)
          unconverged = (phiout > phi) & (phiout < 0.)
          #phiout = tf.Print(phiout,[phiout],message="phiout")
          return (sigmaout,phiout,phiprimeout,unconverged,j+1)
          
        sigmaiter, phiiter,phiprimeiter,unconverged,jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[phiiter],message="phiiter")
        return sigmaiter
      
      #sigma=0 corresponds to the unconstrained solution on the interior of the trust region
      sigma = tf.cond(usesolu, lambda: tf.zeros([],dtype=var.dtype), sigma)

      #solution can be computed directly from eigenvalues and eigenvectors
      coeffs = -a/(lam+sigma)
      coeffs = tf.reshape(coeffs,[1,-1])
      #p = tf.reduce_sum(coeffs*U, axis=-1)
      p = tf.matmul(UT,tf.reshape(coeffs,[-1,1]),transpose_a=True)
      p = tf.reshape(p,[-1])

      Umag = tf.sqrt(tf.reduce_sum(tf.square(UT),axis=1))
      coeffsmag = tf.sqrt(tf.reduce_sum(tf.square(coeffs)))
      pmag = tf.sqrt(tf.reduce_sum(tf.square(p)))
      #the equivalence of |p| and |coeffs| is a partial test of the orthonormality of the eigenvectors
      #which could be degraded in case of excessive loss of numerical precision
      p = tf.Print(p,[pmag,coeffsmag,sigma],message="pmag,coeffsmag,sigma")

      #predicted reduction also computed directly from eigenvalues and eigenvectors
      predicted_reduction_out = -(tf.reduce_sum(a*coeffs) + 0.5*tf.reduce_sum(lam*tf.square(coeffs)))
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    #doiter = tf.Print(doiter,[doiter],message="doiter")
    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
    
    #assign updated values to stored variables, taking care to define dependencies such that things are executed
    #in the correct order
    alist = []
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.e,esec)) 
      alist.append(tf.assign(self.UT,UTsec)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]





class SR1TrustOBS:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None, H = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None and H is not None:
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
    return tf.group(alist)
  
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    #eta = 0.
    eta = 0.15
    
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
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
    
    #grad = self.grad
    B = self.B
    H = self.H
    
    #dgrad = grad - self.grad_old
    #dx = var - self.var_old
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    B,H = tf.cond(self.doiter_old, lambda: doSR1Update(B,H,dgrad,dx), lambda: (B,H))  
    
  
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():

      lam,U = tf.self_adjoint_eig(B) #TODO: check if what is returned here should actually be UT in the paper
      #U = tf.transpose(U)

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      a = tf.matmul(U, gradcol,transpose_a=True)
      a = tf.reshape(a,[-1])
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      a = tf.Print(a,[amagsq,gmagsq],message = "amagsq,gmagsq")
      
      #a = tf.matmul(U, gradcol,transpose_a=False)
      asq = tf.square(a)
      

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      abarsq = tf.Print(abarsq, [a, abar, lam, lambar], message = "a,abar,lam,lambar")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      def solu():
        return -tf.matmul(H,gradcol)
      
      def sol():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          #phi = tf.Print(phi,[phi],message = "checking phi in cond()")
          return tf.logical_and(phi < -tol, j<maxiter)
          #return tf.logical_and(tf.abs(phi) > tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[sigmaiter,phiiter],message = "sigmaiter,phiiter")
        
        coeffs = -a/(lam+sigmaiter)
        coeffs = tf.reshape(coeffs,[1,-1])
        p = tf.reduce_sum(coeffs*U, axis=-1)
        
        return p
      
      p = tf.cond(usesolu, solu, sol)
      p = tf.reshape(p,[-1])
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      p = tf.Print(p,[magp],message = "magp")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(B,tf.reshape(p,[-1,1])),[-1])*p) )
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

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




class LSR1TrustOBS:
    
  def __init__(self, loss, var,grad, k=100, initialtrustradius = 1.):
    self.k = k
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    #self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.MpsiT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype),trainable = False, validate_shape=False)
    #self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([],dtype=var.dtype)
    self.gamma = tf.Variable(tf.ones([],dtype=var.dtype),trainable=False)
    #self.gamma = tf.Variable(tf.zeros([],dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(True)
    self.updateidx = tf.Variable(tf.zeros([],dtype=tf.int32),trainable = False)
    self.scale = tf.Variable(tf.ones_like(var), trainable=False)
    self.delidx = tf.Variable(tf.zeros([],dtype=tf.int64))
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
  def initialize(self, loss, var, grad):

    #scale = 1./tf.abs(grad)
    
    #gradnorm = grad/tf.sqrt(tf.reduce_sum(tf.square(grad)))
    #hessgrad = tf.gradients(grad*gradnorm,var,gate_gradients=True,stop_gradients=gradnorm)[0]
    #scale = hessgrad
    #scale = tf.sqrt(tf.abs(hessgrad))
    #scale = tf.reciprocal(tf.abs(hessgrad))
    #scale = tf.reciprocal(tf.sqrt(tf.abs(hessgrad)))
    #scale = tf.ones_like(scale)
    
    #scale = tf.gradients(grad,var,gate_gradients=True)[0]/tf.cast(tf.shape(var)[0],var.dtype)
    #scale = tf.sqrt(tf.abs(scale))
    
    #scale = tf.sqrt(tf.abs(tf.linalg.tensor_diag_part(hessian)))
    
    #scale = tf.Print(scale,[scale],message="Variable scaling:",summarize=10000)
    
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    #alist.append(tf.assign(self.scale,scale))
    
    return tf.group(alist)
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    #varorig = var
    
    #var *= self.scale
    #grad = grad/self.scale
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #xtol = 0.
    #eta = 0.
    eta = 0.15
    #eta = 1e-3
    
    #compute ratio of actual reduction in loss function to that
    #predicted from quadratic approximation
    #in order to decide whether to reverse the previous step
    #and whether to enlarge or shrink the trust region
    actual_reduction = self.loss_old - loss
    
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)    
    
    #xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    ##edmtol = math.sqrt(xtol)
    ##edmtol = xtol
    ##edmtol = 1e-8
    ##edmtol = 0.
    #eta = 0.
    ##eta = 0.15
    ##tau1 = 0.1
    ##tau2 = 0.3

    ##defaults from nocedal and wright
    ##eta = 1e-3
    #tau1 = 0.1
    #tau2 = 0.75
          
    #actual_reduction = self.loss_old - loss
    
    ##actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    #isnull = tf.logical_not(self.doiter_old)
    #rho = actual_reduction/self.predicted_reduction
    #rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    #rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    #dgrad = grad - self.grad_old
    #dx = var - self.var_old
    #dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    #trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,tau1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,tau2), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    #trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    
    
    def doSR1Scaling(yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      return (scale,False)
    
    gamma,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(dgrad,dx), lambda: (self.gamma,self.doscaling))
    #gamma = tf.ones_like(gamma)

    
    def doSR1Update(STin,YTin,yin,dxin):
      ycol = tf.reshape(yin,[-1,1])
      dxcol = tf.reshape(dxin,[-1,1])
      
      yrow = tf.reshape(yin,[1,-1])
      dxrow = tf.reshape(dxin,[1,-1])
      
      #dyBx = ycol - Bv(gamma,self.psi,self.M,dxcol)
      dyBx = ycol - Bv(gamma,self.psi,self.MpsiT,dxcol)
      den = tf.matmul(dyBx, dxcol, transpose_a = True)
      #den = tf.reshape(den,[])
      
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.greater(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      nonzero = dentest
      #nonzero = tf.logical_and(dentest,tf.not_equal(actual_reduction,0.))
      
      nonzero = tf.Print(nonzero,[nonzero],message="nonzero")
      
      #nonzero = tf.Print(nonzero, [den,dennorm, dentest, nonzero], message = "den, dennorm, dentest, nonzero")
      
      #nonzero = tf.abs(den) > 1e-8
      
      #doappend = tf.logical_and(nonzero, tf.shape(STin)[0] < self.k)
      #doreplace = tf.logical_and(nonzero, tf.shape(STin)[0] >= self.k)
      
      kcur = tf.shape(STin)[0]
      #sliceidx = tf.where(kcur < self.k, kcur, tf.cast(self.delidx,tf.int32))
      #sliceidx = tf.where(kcur < self.k, kcur, kcur-1)
      sliceidx = tf.where(kcur < self.k, 0, 1)
      #sliceidx = tf.Print(sliceidx,[sliceidx],message="sliceidx")
      
      #print(den.shape)
      
      def update():
        #dxmag = tf.sqrt(tf.reduce_sum(tf.square(dxrow)))
        #dxrown = dxrow/dxmag
        #yrown = yrow/tf.sqrt(tf.reduce_sum(tf.square(yrow)))
        #yrown = yrow/dxmag
        ST = tf.concat([STin[sliceidx:],dxrow],axis=0)
        YT = tf.concat([YTin[sliceidx:],yrow],axis=0)
        
        #ST = tf.concat([STin[:sliceidx],dxrow,STin[sliceidx+1:]],axis=0)
        #YT = tf.concat([YTin[:sliceidx],yrow,YTin[sliceidx+1:]],axis=0)
        return (ST,YT)
      
      ST,YT = tf.cond(nonzero, update, lambda: (STin, YTin))

      return (ST,YT)
    
    ST = self.ST
    YT = self.YT

    #doscaling = tf.constant(False)
    ST,YT = tf.cond(self.doiter_old, lambda: doSR1Update(ST,YT,dgrad,dx), lambda: (ST,YT))
    
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #dx = grad
    #gradconst = tf.identity(grad)
    #dgrad = tf.gradients(grad*tf.stop_gradient(grad),var,gate_gradients=True)[0]
    #dgrad = tf.gradients(grad*gradconst,var,gate_gradients=True,stop_gradients=gradconst)[0]
    #ST,YT = tf.cond(self.doiter_old, lambda: doSR1Update(ST,YT,dgrad,dx), lambda: (ST,YT))
    #ST,YT = tf.cond(tf.logical_or(self.isfirstiter,self.doiter_old), lambda: doSR1Update(ST,YT,dgrad,dx), lambda: (ST,YT))    
    
    #compute compact representation
    S = tf.transpose(ST)
    Y = tf.transpose(YT)
    psi = Y - gamma*S
    psiT = tf.transpose(psi)
    STY = tf.matmul(ST,YT,transpose_b=True)
    D = tf.matrix_band_part(STY,0,0)
    L = tf.matrix_band_part(STY,-1,0) - D
    LT = tf.transpose(L)
    STB0S = gamma*tf.matmul(ST,S)
    Minverse = D + L + LT - STB0S
    MpsiT = tf.matrix_solve(Minverse,psiT)
    
    Q,R = tf.qr(psi)
    #detR = tf.matrix_determinant(R)
    RT = tf.transpose(R)
    MRT = tf.matrix_solve(Minverse,RT)
    RMRT = tf.matmul(R,MRT)
    e,U = tf.self_adjoint_eig(RMRT) #TODO: check if what is returned here should actually be UT in the paper
    QU = tf.matmul(Q,U)
    
    ##print("QU.shape")
    ##print(QU.shape)
    ##QU = tf.Print(QU,[tf.shape(QU)],message="shape(QU)")
    #ST = tf.transpose(QU)
    #YT = tf.reshape(e,[-1,1])*ST
        
    
    
    #M = tf.matrix_inverse(Minverse)
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    #isconvergedxtol = trustradius_out < xtol
    ##isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    #isconvergededmtol = self.predicted_reduction <= 0.
    
    #isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    #doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():
      #grad = self.grad
      
      #compute eigen decomposition
      #psiTpsi = tf.matmul(psiT,psi)
      #epsiTpsi = tf.self_adjoint_eigvals(psiTpsi)
      #e0psiTpsi = tf.reduce_min(epsiTpsi)
      #psiTpsi = tf.Print(psiTpsi,[e0psiTpsi], message = "e0psiTpsi")
      ##psiTpsi = psiTpsi + 4.*tf.maximum(-e0psiTpsi,tf.zeros_like(e0psiTpsi))*tf.eye(tf.shape(psiTpsi)[0],dtype=psiTpsi.dtype)
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      #def chol():
        #RT = tf.cholesky(psiTpsi)
        #R = tf.transpose(RT)
        #return (R,RT)
      
      #def qr():
        #Q,R = tf.qr(psi)
        #RT = tf.transpose(R)
        #return (R,RT)
      
      ##R,RT = tf.cond(e0psiTpsi > 0., chol, qr)
      #R,RT = chol()
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      
      #Q,R = tf.qr(psi)
      #detR = tf.matrix_determinant(R)
      #RT = tf.transpose(R)
      #MRT = tf.matrix_solve(Minverse,RT)
      #RMRT = tf.matmul(R,MRT)
      #e,U = tf.self_adjoint_eig(RMRT) #TODO: check if what is returned here should actually be UT in the paper
      #QU = tf.matmul(Q,U)
      
      #U = tf.transpose(U)
      
      #print("e.shape, U.shape")
      #print([e.shape,U.shape])
      
      #e = tf.Print(e,[tf.concat((e,[1.]),axis=0)[0]],message="e0")

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      
      #QU=Q
      
      #gpll = tf.matmul(tf.matmul(psi,tf.matmul(Rinverse,U)), gradcol,transpose_a=True)
      #gpll = tf.matmul(tf.matmul(psi,tf.matrix_solve(R,U)), gradcol,transpose_a=True)
      #gpll = tf.matmul(QU, gradcol,transpose_a=True)
      gpll = tf.matmul(QU, gradcol,transpose_a=True)
      gradpll = tf.matmul(QU,gpll)
      gradpll = tf.reshape(gradpll,[-1])
      gpll = tf.reshape(gpll,[-1])
      #gpllsq = tf.square(gpll)
      #gmagsq = tf.reduce_sum(tf.square(grad))
      #gpllmagsq = tf.reduce_sum(gpllsq)
      #gperpmagsq = gmagsq - gpllmagsq
      #gperpmagsq = tf.Print(gperpmagsq,[gpllmagsq,gperpmagsq],message="gpllmagsq,gperpmagsq")
      #gperpmagsq = tf.maximum(gperpmagsq,tf.zeros_like(gperpmagsq))
      #gperpmag = tf.sqrt(gperpmagsq)
      gradperp = grad - gradpll
      gperpmagsq = tf.reduce_sum(tf.square(gradperp))
      gperpmag = tf.sqrt(gperpmagsq)
      
      #gperpmagsq = tf.where(tf.shape(gpll)[0]>0,tf.zeros_like(gperpmagsq),gperpmagsq)
      
      
      gradperpu = gradperp/gperpmag
      
      #gradperpurow = tf.reshape(gradperpu,[1,-1])
      
      #basis = tf.concat((tf.transpose(Q),gradperpurow),axis=0)
      
      #gradbasis = tf.concat([gpll,[tf.sqrt(gperpmagsq)]],axis=0)

      
      #def gradloop(i):
        #v = tf.gather(basis,i,axis=0)
        #return tf.gradients(grad*v,var,stop_gradients=v)[0]
      
      #niter = tf.shape(basis)[0]
      #niter = tf.Print(niter,[niter],message="niter")
      #Hbasis = control_flow_ops.for_loop(gradloop,basis.dtype, niter, parallel_iterations=None)
      
      #Hsub = tf.matmul(basis,Hbasis,transpose_b=True)
      
      #print("basis.shape")
      #print(basis.shape)
      
      
      #print("Hbasis.shape")
      #print(Hbasis.shape)
      
      #print("Hsub.shape")
      #print(Hsub.shape)
      
      #e,U = tf.self_adjoint_eig(Hsub)
      #QU = tf.matmul(basis,U,transpose_a=True)
      
      #gradbasiscol = tf.reshape(gradbasis,[-1,1])
      #a = tf.matmul(U,gradbasiscol,transpose_a=True)
      #a = tf.reshape(a,[-1])
      
      #print("U.shape")
      #print(U.shape)
      #print("gradbasiscol.shape")
      #print(gradbasiscol.shape)
      #print("a.shape")
      #print(a.shape)
      
      #a = tf.Print(a,[tf.shape(a)],message="shape(a)")
      
      #e = tf.Print(e,[e],message="e",summarize=1000)
      
      #lamperp = tf.reduce_sum(gradperpu*tf.gradients(grad*gradperpu,var,stop_gradients=gradperpu)[0])   
      lamperp = gamma
      #lamperp = gamma
      
      #gpll = tf.Print(gpll,[tf.shape(gpll)], message = "gpll shape:")
      #a = gpll
      a = tf.concat([gpll,[tf.sqrt(gperpmagsq)]],axis=0)
      lam = e + gamma
      lam = tf.concat((e+gamma,[lamperp]),axis=0)
      
      
      #a = a[:var.shape[0]]
      asq = tf.square(a)
      
      #lam = e + gamma
      #lam = tf.concat([lam,tf.reshape(gamma,[1])],axis=0)
      #lam = tf.pad(e,[[0,1]]) + gamma
      #lam = lam[:var.shape[0]]
      
      #padsize = tf.where(tf.shape(e)[0] < var.shape[0], 1, 0)
      #lam = tf.pad(e,[[0,padsize]]) + gamma

      #lampll = e + gamma

      #lam = tf.pad(e,[[0,1]]) + gamma
      #lam = e + gamma
      #lam = tf.concat([lam,[lamperp]],axis=0)

      #lam = e

      #lam = tf.Print(lam,[lam],message="lam",summarize=1000)

      

      #e0 = tf.reduce_min(e)
      e0 = tf.reduce_min(lam)
      #e0 = tf.reduce_min(tf.concat((lam,[gamma]),axis=0))
      


      #deal with null gradient components and repeated eigenvectors
      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))        
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      #TODO, add handling of additional cases here (singular and "hard" cases)
      
      phisigma0 = phif(sigma0)
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      
      def sigma():
        #tol = 1e-10
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit,phiprimeinit = phiphiprime(sigmainit)
                
        loop_vars = [sigmainit, phiinit,phiprimeinit, tf.constant(True), tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,phiprime,unconverged,j):
          return (unconverged) & (j<maxiter)
        
        def body(sigma,phi,phiprime,unconverged,j):   
          sigmaout = sigma - phi/phiprime
          phiout, phiprimeout = phiphiprime(sigmaout)
          unconverged = (phiout > phi) & (phiout < 0.)
          #phiout = tf.Print(phiout,[phiout],message="phiout")
          return (sigmaout,phiout,phiprimeout,unconverged,j+1)
          
        sigmaiter, phiiter,phiprimeiter,unconverged,jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        sigmaiter = tf.Print(sigmaiter,[phiiter],message="phiiter")
        return sigmaiter

      
      sigma = tf.cond(usesolu, lambda: sigma0, lambda: sigma())
      
      
      
      
      #tau = sigma + gamma
      
      ##print(var.shape[0])
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #innerinverse = tau*Minverse + tf.matmul(psi,psi,transpose_a=True)
      #innerpsiT = tf.matrix_solve(innerinverse,psiT)
      ##inner = tf.matrix_inverse(innerinverse)
      ##inner2 = tf.matmul(tf.matmul(psi,inner),psi, transpose_b=True)
      #inner2 = tf.matmul(psi,innerpsiT)
      #p = -tf.matmul(I-inner2, gradcol)/tau
      #p = tf.reshape(p,[-1])
      
      c = -a/(lam+sigma)
      
      #c = tf.Print(c,[tf.shape(lam)],message="shape(lam)")
      #c = tf.Print(c,[tf.shape(sigma)],message="shape(sigma)")
      #c = tf.Print(c,[tf.shape(c)],message="shape(c)")
      
      ccol = tf.reshape(c,[-1,1])
      gradperpucol = tf.reshape(gradperpu,[-1,1])

      
      p = tf.matmul(QU,ccol[:-1]) + ccol[-1]*gradperpucol
      p = tf.reshape(p,[-1])
      
      #p = tf.matmul(QU,ccol)
      #p = tf.reshape(p,[-1])
      
      #p = tf.matmul(QU,ccol[:-1]) + c[-1]*tf.reshape(gradperpu,[-1,1])
      #p = tf.reshape(p,[-1])
      
      #cpll = -a/(lam+sigma)
      
      #cpll = -gpll/lampll
      #cpll = -a/lam
      #cpll = tf.clip_by_value(cpll,-trustradius_out,trustradius_out)
      #cpll = tf.where(tf.less(lam,0.),-tf.sign(gpll)*trustradius_out,cpll)
      ##TODO handle special case below
      #cpll = tf.where(tf.logical_and(tf.equal(gpll,0.),tf.less(lam,0.)),trustradius_out*tf.ones_like(cpll),cpll)
      
      predicted_reduction_out = tf.reduce_sum(-(a*c + 0.5*lam*tf.square(c)))
      
      #reductionpll = -(a*cpll + 0.5*lam*tf.square(cpll))
      ##reductionpll = tf.Print(reductionpll,[reductionpll],message="reductionpll",summarize=1000)
      #delidx = tf.cond(tf.shape(reductionpll)[0]>0, lambda: tf.argmin(reductionpll), lambda: tf.cast(0,tf.int64))
      
      
      ##gradperpu = gradperp/gperpmag
      ###gammaperp = tf.reduce_sum(gradperpu*tf.gradients(grad*gradperpu,var,stop_gradients=gradperpu)[0])
      ##gammaperp = gamma
      ##cperp = -gperpmag/gammaperp
      ##cperp = tf.clip_by_value(cperp,-trustradius_out,trustradius_out)
      ##cperp = tf.where(tf.less(gammaperp,0.),-trustradius_out,cperp)
      ###cperp = tf.where(tf.equal(gperp,0.) & tf.less(gammaperp,0.), trustradius_out*tf.ones_like(cperp),cperp)
      
      ##cperp = tf.Print(cperp,[gperpmag],message="gperpmag")
      ##cperp = tf.Print(cperp,[gammaperp],message="gammaperp")
      ##cperp = tf.Print(cperp,[cperp],message="cperp")
      
      ##deltaperp = 1e-3*trustradius_out
      ##maggperp = grad - gpll
      ##maggperp = tf.sqrt(tf.reduce_sum(tf.square(gperp)))
      #t = tf.where(tf.less_equal(gperpmag,gamma*trustradius_out), 1./gamma, trustradius_out/gperpmag)
      ##t = 0.
      
      #cpllcol = tf.reshape(cpll,[-1,1])
      ##gradperpucol = tf.reshape(gradperpu,[-1,1])
      #gpllcol = tf.reshape(gpll,[-1,1])
      #p = tf.matmul(QU,cpllcol + t*gpllcol) - t*gradcol
      #p = tf.transpose(tf.matmul(cpllcol + t*gpllcol,QU,transpose_b=True,transpose_a=True)) - t*gradcol
      
      #p = tf.matmul(QU,cpllcol + t*gpllcol) - t*gradcol
      
      #print("cperp.shape")
      #print(cperp.shape)
      #print("gradperpcol.shape")
      #print(gradperpcol.shape)
      #p = tf.matmul(QU,cpllcol) + cperp*gradperpucol
      
      #ppll = tf.matmul(QU,cpllcol)
      #p = tf.where(tf.shape(cpll)[0]>0,ppll,p)
      
      
      #p = tf.reshape(tf.matmul(cpllcol + t*gpllcol,QU,transpose_a=True,transpose_b=True),[-1,1]) - t*gradcol
      #p = tf.cond(tf.shape(cpllcol)[0]>0, lambda: tf.matmul(QU,cpllcol + t*gpllcol,transpose_a=True) - t*gradcol, lambda: -t*gradcol)
      #p = tf.reshape(p,[-1])
      
      ##print(var.shape[0])
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #innerinverse = tau*Minverse + tf.matmul(psi,psi,transpose_a=True)
      #innerpsiT = tf.matrix_solve(innerinverse,psiT)
      ##inner = tf.matrix_inverse(innerinverse)
      ##inner2 = tf.matmul(tf.matmul(psi,inner),psi, transpose_b=True)
      #inner2 = tf.matmul(psi,innerpsiT)
      #p = -tf.matmul(I-inner2, gradcol)/tau
      #p = tf.reshape(p,[-1])
      
      #magpalt = tf.sqrt(tf.reduce_sum(tf.square(cpll)) + gperpmagsq)
      
      #magpalt = tf.sqrt(tf.reduce_sum(tf.square(cpll)) + tf.square(cperp))
      
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      #detMinverse = tf.matrix_determinant(Minverse)
      #detinnerinverse = tf.matrix_determinant(innerinverse)
      #p = tf.Print(p,[magp,detR,detMinverse,detinnerinverse],message = "magp, detR, detMinverse, detinnerinverse")
      magpdiff = tf.abs(magp-trustradius_out)/trustradius_out
      #p = tf.Print(p,[magp,magpdiff,e0,phisigma0],message="magp,magpdiff,e0,phisigma0")
      #p = tf.Print(p,[magp,magpalt,magpdiff,e0],message="magp,magpalt,magpdiff,e0")
      p = tf.Print(p,[magp,magpdiff,e0],message="magp,magpdiff,e0")

      #e0val = efull[0]
      #e0val = e0
      
      delidx = tf.cast(0,tf.int64)
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      #predicted_reduction_alt = -(tf.reduce_sum(gpll*cpll) + 0.5*tf.reduce_sum(cpll*cpll/e) + 

      #predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(Bvflat(gamma, psi, MpsiT, p)*p))
      
      nearboundary = tf.logical_not(usesolu)
      
      #nearboundary = tf.logical_or(tf.reduce_any(tf.greater(tf.abs(cpll),0.95*trustradius_out)), tf.greater(gperpmag,0.95*gamma*trustradius_out))
      #nearboundary = tf.logical_or(tf.logical_not(usesolu), tf.greater(gperpmag,0.95*gamma*trustradius_out))
      #nearboundary = tf.reduce_any(tf.greater(tf.abs(cpll),0.95*trustradius_out))
      
      #return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]
      return [var+p, predicted_reduction_out, nearboundary, grad, lamperp, delidx]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old,gamma,self.delidx])
    var_out, predicted_reduction_out, atboundary_out, grad_out, gamma, delidx = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.ST,ST,validate_shape=False))
      alist.append(tf.assign(self.YT,YT,validate_shape=False))
      alist.append(tf.assign(self.psi,psi,validate_shape=False))
      #alist.append(tf.assign(self.M,M,validate_shape=False))
      alist.append(tf.assign(self.MpsiT,MpsiT,validate_shape=False))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.gamma,gamma))
      alist.append(tf.assign(self.delidx,delidx))
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.assign(varorig, var_out/self.scale)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]


class TrustNCG:
  
  def __init__(self, loss, var, grad, initialtrustradius = 1., doSR1 = False):

    self.doSR1 = doSR1
    self.initialtrustradius = initialtrustradius
    
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
      #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
  def initialize(self, loss, var, grad):
        
    alist = []
    alist.append(tf.assign(self.trustradius,self.initialtrustradius))
    alist.append(tf.assign(self.loss_old, loss))
    alist.append(tf.assign(self.predicted_reduction, 0.))
    alist.append(tf.assign(self.var_old, var))
    alist.append(tf.assign(self.atboundary_old,False))
    alist.append(tf.assign(self.doiter_old,False))
    alist.append(tf.assign(self.isfirstiter,True))
    if self.doSR1:
      alist.append(tf.assign(self.B,tf.zeros([var.shape[0],var.shape[0]],dtype=var.dtype)))
      alist.append(tf.assign(self.grad_old,grad))

    return tf.group(alist)
  
  def setHessApprox(self,b,sess):
    self.B.load(b,sess)
  
  def minimize(self, loss, var,grad, hessian=None):
    
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

      #grad = self.grad
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
    
    trustradius_out = tf.Print(trustradius_out,[self.loss_old, loss, actual_reduction,self.predicted_reduction,isnull,rho,trustradius_out,isconvergedxtol,isconvergededmtol,doiter])
    
    @function.Defun(var.dtype)
    def hesspexact(v):
      #v.set_shape(var.shape)
      #hv = tf.gradients(grad*tf.stop_gradient(v),var, gate_gradients=True)[0]
      hv = tf.gradients(grad*v,var, gate_gradients=True,stop_gradients=v)[0]
      #hv.set_shape(var.shape)
      return hv
      #return tf.gradients(grad*tf.stop_gradient(v),var, gate_gradients=True)[0]
      #return tf.gradients(grad,var, gate_gradients=True)[0]
      #vj = tf.argmin(v)
      #return tf.gradients(tf.gather(grad,vj),var, gate_gradients=True)[0]
      #return tf.gradients(grad*v,1.*var, gate_gradients=True,stop_gradients=v)[0]
      
    def hesspapprox(v):
      return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])
      
    if self.doSR1:
      hessp = hesspapprox
    else:
      hessp = hesspexact
      
    def dfval_quad(v):
      return tf.reduce_sum(grad*v) + 0.5*tf.reduce_sum(hessp(v)*v)      
    
    gradmag = tf.sqrt(tf.reduce_sum(tf.square(grad)))
    tolerance = tf.minimum(0.5*tf.ones_like(loss), tf.sqrt(gradmag)) * gradmag
    #tolerance = 0.5*gradmag
    #tolerance = -tf.ones_like(loss)
    #tolerance = tf.minimum(0.5*tf.ones_like(loss), gradmag) * gradmag    
    
    def build_loop():
      #if self.doSR1:
        #grad = self.grad
      #else:
        #grad = tf.gradients(loss,var, gate_gradients=True)[0]
      
      #u = tf.ones_like(var)
      #hesspart = tf.gradients(grad,var,gate_gradients=True,stop_gradients=u)[0]
      
      #def hesspexact(v):
        #return tf.reshape(tf.matmul(hessian,tf.reshape(v,[-1,1])),[-1])
      
      

      
      def cond(z,r,d,atboundary,attol,isneg,j):
        return tf.logical_and(tf.logical_not(atboundary),tf.logical_not(attol))
      
      def body(z,r,d,atboundary,attol,isneg,j):              
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
        Bd.set_shape(var.shape)
        #print(Bd.shape)
        #Bd = tf.gradients(grad*d,var,stop_gradients=[d])[0]
        #Bd = tf.gradients(grad[j],var,stop_gradients=j)[0]
        dBd = tf.reduce_sum(d*Bd)
        
        #compute quadratic approximation to function value at boundaries
        #(do it conditionally, since the cost is non-trivial)
        negativecurvature = tf.less_equal(dBd,0.)
        #negativecurvature = tf.Print(negativecurvature,[negativecurvature],message = 'negativecurvature')
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

        return [z_next, r_next, d_next, atboundary_next, attol_next,negativecurvature, j+1]
      
      z = tf.zeros_like(var)
      r = grad
      d = -grad
      atboundary = tf.constant(False)
      attol = tf.constant(False)
      isneg = tf.constant(False)
      

      
      loop_vars = [z,r,d,atboundary,attol,isneg,tf.zeros([],dtype=tf.int32)]
      #print([z.shape,r.shape,d.shape,atboundary.shape,attol.shape])

      #solve subproblem iteratively
      z_out,r_out,d_out,atboundary_out,attol_out,isneg_out,niter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False, maximum_iterations=var.shape[0])
    
      #e0 = tf.linalg.eigvalsh(hessian)[0]
      #z_out = tf.Print(z_out,[niter,atboundary_out,attol_out,isneg_out,e0],message="niter,atboundary,attol,isneg,e0")
      z_out = tf.Print(z_out,[niter,atboundary_out,attol_out,isneg_out],message="niter,atboundary,attol,isneg")
    
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
        alist.append(tf.assign(self.grad_old,grad))
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
