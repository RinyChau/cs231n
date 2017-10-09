from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_batch_norm_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    out, ba_cache = batchnorm_forward(a, gamma, beta, bn_param)
    cache = (fc_cache, ba_cache)
    return out, cache

def affine_batch_norm_backward(dout, cache):
    fc_cache, ba_cache = cache
    da, dgamma, dbeta = batchnorm_backward_alt(dout, ba_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_batch_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    b, ba_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (fc_cache, ba_cache, relu_cache)
    return out, cache

def affine_batch_norm_relu_backward(dout, cache):
    fc_cache, ba_cache, relu_cache = cache
    db = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(db, ba_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_param):
    a, fc_cache = affine_forward(x, w, b)
    b, relu_cache = relu_forward(a)
    out, dropout_cache = dropout_forward(b, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache

def affine_relu_dropout_backward(dout, cache):
    fc_cache, relu_cache, dropout_cache = cache
    db = dropout_backward(dout, dropout_cache)
    da = relu_backward(db, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_batch_norm_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    a, fc_cache = affine_forward(x, w, b)
    b, ba_cache = batchnorm_forward(a, gamma, beta, bn_param)
    c, relu_cache = relu_forward(b)
    out, dropout_cache = dropout_forward(c, dropout_param)
    cache = (fc_cache, ba_cache, relu_cache, dropout_cache)
    return out, cache

def affine_batch_norm_relu_dropout_backward(dout, cache):
    fc_cache, ba_cache, relu_cache, dropout_cache = cache
    dc = dropout_backward(dout, dropout_cache)
    db = relu_backward(dc, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(db, ba_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta



def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def residual_forward(x, fw, fb, sw, sb, conv_param):
    f_out, f_cache = conv_relu_forward(x, fw, fb, conv_param)
    s_out, s_cache = conv_relu_forward(f_out, sw, sb, conv_param)
    out = x + s_out
    cache = f_cache, s_cache
    return out, cache

def residual_backward(dout, cache):
    f_cache, s_cache = cache
    dsx, dsw, dsb= conv_relu_backward(dout, s_cache)
    dfx, dfw, dfb = conv_relu_backward(dsx, f_cache)
    dx =dfx + dout
    return dx, dfw, dfb, dsw, dsb

def residual_spatial_batchnorm_forward(x, fw, fb, sw, sb, conv_param, fgamma, fbeta, sgamma, sbeta, fbn_param, sbn_param):
    f_out, f_cache = conv_spatial_batchnorm_relu_forward(x, fw, fb, conv_param, fgamma, fbeta, fbn_param)
    s_out, s_cache = conv_spatial_batchnorm_relu_forward(f_out, sw, sb, conv_param, sgamma, sbeta, sbn_param)
    out = x + s_out
    cache = f_cache, s_cache
    return out, cache

def residual_spatial_batchnorm_backward(dout, cache):
    f_cache, s_cache = cache
    dsx, dsw, dsb, dsgamma, dsbeta = conv_spatial_batchnorm_relu_backward(dout, s_cache)
    dfx, dfw, dfb, dfgamma, dfbeta = conv_spatial_batchnorm_relu_backward(dsx, f_cache)
    dx =dfx + dout
    return dx, dfw, dfb, dsw, dsb, dfgamma, dfbeta, dsgamma, dsbeta

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  s, conv_relu_cache = conv_relu_forward(x, w, b, conv_param)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_relu_cache[0], conv_relu_cache[1], pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_spatial_batchnorm_forward(x, w, b, conv_param, gamma, beta, bn_param):
    out, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, spa_ba_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
    return out, (conv_cache, spa_ba_cache)

def conv_spatial_batchnorm_backward(dout, cache):
    conv_cache, spa_ba_cache = cache
    dx, dgamma, dbeta = spatial_batchnorm_backward(dout, spa_ba_cache)
    dx, dw, db = conv_backward_fast(dx, conv_cache)
    return dx, dw, db, dgamma, dbeta

def conv_spatial_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    out, cache = conv_spatial_batchnorm_forward(x, w, b, conv_param, gamma, beta, bn_param)
    out, relu_cache = relu_forward(out)
    cache += (relu_cache, )
    return out, cache

def conv_spatial_batchnorm_relu_backward(dout, cache):
    conv_cache, spa_bat_cache, relu_cache = cache
    dout = relu_backward(dout, relu_cache)
    dx, dw, db, dgamma, dbeta = conv_spatial_batchnorm_backward(dout, (conv_cache, spa_bat_cache))
    return dx, dw, db, dgamma, dbeta

def conv_spatial_batchnorm_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):
    out, cache = conv_spatial_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)
    out, pool_cache = max_pool_forward_fast(out, pool_param)
    cache = (cache, pool_cache)
    return out, cache

def conv_spatial_batchnorm_relu_pool_backward(dout, cache):
    cache, pool_cache = cache
    dout = max_pool_backward_fast(dout, pool_cache)
    dx, dw, db, dgamma, dbeta = conv_spatial_batchnorm_relu_backward(dout, cache)
    return dx, dw, db, dgamma, dbeta




   
    
