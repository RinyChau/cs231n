import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.layer_helper import *
from initializations import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params["W1"] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    self.params["b1"] = np.zeros(num_filters)
    self.params["W2"] = np.random.randn(num_filters * (input_dim[1]/2) * (input_dim[2]/2), hidden_dim) * weight_scale
    self.params["b2"] = np.zeros(hidden_dim)
    self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b3"] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    reg = self.reg
    # W1, b1 = self.params['W1'], self.params['b1']
    # W2, b2 = self.params['W2'], self.params['b2']
    # W3, b3 = self.params['W3'], self.params['b3']
    
    N = X.shape[0]
    
    con_relu_out, conv_relu_cache = conv_relu_forward(X, W1, b1, conv_param)
    pool_out, pool_cache = max_pool_forward_fast(con_relu_out, pool_param)
    affine_relu_out, affine_relu_cache =  affine_relu_forward(pool_out, W2, b2)
    scores, affine_cache = affine_forward(affine_relu_out, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 *reg* (np.sum(np.minimum(W1,1000000)**2) + np.sum(np.minimum(W2,1000000)**2) + np.sum(np.minimum(W3,1000000)**2))
    loss = data_loss + reg_loss
    
    d_hidden_layer, dW3, dB3 = affine_backward(dscores, affine_cache)
    dW3 += reg * W3
    grads["W3"] = dW3
    grads["b3"] = dB3
    
    daffine_relu_out, dW2, dB2 = affine_relu_backward(d_hidden_layer, affine_relu_cache)
                          
    dW2 += reg * W2
    grads["W2"] = dW2
    grads["b2"] = dB2
    
    dpool_out = max_pool_backward_fast(daffine_relu_out, pool_cache)
    dx1, dW1, dB1 = conv_relu_backward(dpool_out, conv_relu_cache)
    
    dW1 += reg * W1
    grads["W1"] = dW1
    grads["b1"] =dB1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class FullyConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=((16, 16), (32, 32), (64, 64), ), filter_size=3,
               hidden_dims=(4096,4096, 1000), num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
      """
      Initialize a new network.
      
      Inputs:
      - input_dim: Tuple (C, H, W) giving size of input data
      - num_filters: Number of filters to use in the convolutional layer
      - filter_size: Size of filters to use in the convolutional layer
      - hidden_dim: Number of units to use in the fully-connected hidden layer
      - num_classes: Number of scores to produce from the final affine layer.
      - weight_scale: Scalar giving standard deviation for random initialization
        of weights.
      - reg: Scalar giving L2 regularization strength
      - dtype: numpy datatype to use for computation.
      """
      self.params = {}
      self.reg = reg
      self.dtype = dtype
      self.input_dim = input_dim
      self.num_filters = num_filters
      self.filter_size = filter_size
      self.hidden_dims =  hidden_dims
      self.num_classes = num_classes
      self.weight_scale = weight_scale
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
      i = 1
      num_channel = input_dim[0]
      width = input_dim[1]
      height = input_dim[2]
      for filter_pair in num_filters:
        for num_filter in filter_pair:
          w = "W" + str(i)
          b = "b" + str(i)
          self.params[w] = np.random.randn(num_filter, num_channel, filter_size, filter_size) * weight_scale
          self.params[b] = np.zeros(num_filter)
          num_channel = num_filter
          i += 1
        width = width/2
        height = height/2
      input_dim = num_channel * width * height
      for hidden_dim in hidden_dims:
        w = "W" + str(i)
        b = "b" + str(i)
        self.params[w] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params[b] = np.zeros(hidden_dim)
        input_dim = hidden_dim
        i += 1
      w = "W" + str(i)
      b = "b" + str(i)
      self.params[w] = np.random.randn(input_dim, num_classes) * weight_scale
      self.params[b] = np.zeros(num_classes)

    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)
    

    def loss(self, X, y=None):
      """
      Evaluate loss and gradient for the three-layer convolutional network.
      
      Input / output: Same API as TwoLayerNet in fc_net.py.
      """
    
      # pass conv_param to the forward pass for the convolutional layer
      filter_size = self.filter_size
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
      
      # pass pool_param to the forward pass for the max-pooling layer
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      
      scores = None
      ############################################################################
      # TODO: Implement the forward pass for the three-layer convolutional net,  #
      # computing the class scores for X and storing them in the scores          #
      # variable.                                                                #
      ############################################################################
      reg = self.reg
    
      N = X.shape[0]
    
      
      num_filters = self.num_filters
      hidden_dims = self.hidden_dims
      num_channel = self.input_dim[0]
      w = self.input_dim[1]
      h = self.input_dim[2]
      layer = X
      layers = {}
      i = 1
      for filter_pair in num_filters:
        for num_filter in filter_pair:
          w = "W" + str(i)
          b = "b" + str(i)
          W = self.params[w]
          B = self.params[b]
          layer, cache = conv_relu_forward(layer, W, B, conv_param)
          layers[i] = {"cache": cache}
          i += 1
        layer, cache = max_pool_forward_fast(layer, pool_param)
        if w == 'W1':
              self.layer1 = layer
        layers[i-1]['pool'] = {"cache":cache}
    
      for hidden_dim in hidden_dims:
        w = "W" + str(i)
        b = "b" + str(i)
        W = self.params[w]
        B = self.params[b]
        layer, cache = affine_relu_forward(layer, W, B)
        layers[i] = {"cache": cache}
        i += 1
        
      w = "W" + str(i)
      b = "b" + str(i)
      W = self.params[w]
      B = self.params[b]
      layer, cache = affine_forward(layer, W, B)
      layers[i] = {"cache": cache}
      scores = layer
    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################
    
      if y is None:
        return scores
    
      loss, grads = 0, {}
      ############################################################################
      # TODO: Implement the backward pass for the three-layer convolutional net, #
      # storing the loss and gradients in the loss and grads variables. Compute  #
      # data loss using softmax, and make sure that grads[k] holds the gradients #
      # for self.params[k]. Don't forget to add L2 regularization!               #
      ############################################################################
      data_loss, dscores = softmax_loss(scores, y)
      reg_loss = 0
      reg_loss += 0.5 * reg * np.sum(np.minimum(W,1000000)**2)
      i = 1
      for filter_pair in num_filters:
        for _ in filter_pair:
          w = "W" + str(i)
          reg_loss += 0.5 * reg * np.sum(np.minimum(self.params[w],1000000)**2)
          i += 1
        
      for _ in hidden_dims:
         w = "W" + str(i)
         reg_loss += 0.5 * reg * np.sum(np.minimum(self.params[w],1000000)**2)
         i += 1
      w = "W" + str(i)      
      reg_loss += 0.5 * reg * np.sum(np.minimum(self.params[w],1000000)**2)
      loss = data_loss + reg_loss
    
      w = "W" + str(i)
      b = "b" + str(i)
      dout, dW, dB = affine_backward(dscores, layers[i]['cache'])
      dW += reg * self.params[w]
      grads[w] = dW
      grads[b] = dB
      i -= 1
    
      for _ in hidden_dims:
        w = "W" + str(i)
        b = "b" + str(i)
        dout, dW, dB = affine_relu_backward(dout, layers[i]['cache'])
        dW += reg * self.params[w]
        grads[w] = dW
        grads[b] = dB
        i -= 1
    
      for filter_pair in num_filters:
        dout = max_pool_backward_fast(dout, layers[i]['pool']['cache'])
        for _ in filter_pair:
          w = "W" + str(i)
          b = "b" + str(i)
          dout, dW, dB = conv_relu_backward(dout, layers[i]['cache'])
          dW += reg * self.params[w]
          grads[w] = dW
          grads[b] = dB
          i -=1
    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################
      self.grads = grads
      return loss, grads
    
    
class FlexibleConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), layers={}, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_xavier=True):
      """
      Initialize a new network.
      
      Inputs:
      - input_dim: Tuple (C, H, W) giving size of input data
      - num_filters: Number of filters to use in the convolutional layer
      - filter_size: Size of filters to use in the convolutional layer
      - hidden_dim: Number of units to use in the fully-connected hidden layer
      - num_classes: Number of scores to produce from the final affine layer.
      - weight_scale: Scalar giving standard deviation for random initialization
        of weights.
      - reg: Scalar giving L2 regularization strength
      - dtype: numpy datatype to use for computation.
      """
      self.params = {}
      self.reg = reg
      self.dtype = dtype
      self.input_dim = input_dim
      self.num_classes = num_classes
      self.weight_scale = weight_scale
      self.layers = layers
      self.use_xavier = use_xavier
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
      pre_dim = input_dim
      for i, layer in enumerate(layers):
        pre_dim, gradient_params, func_params = self.get_params(layer, pre_dim, dtype)
        for k in gradient_params:
            self.params["{}{}".format(k, i+1)] = gradient_params[k]
        layer['gradient_params'] = gradient_params
        layer['func_params'] = func_params
      i = len(layers) + 1
      w = "W" + str(i)
      b = "b" + str(i)
      _weight_scale = weight_scale
      if self.use_xavier:
            _weight_scale = xavier_weight_scale(pre_dim)
      self.params[w] = np.random.randn(pre_dim, num_classes) * _weight_scale
      self.params[b] = np.zeros(num_classes)

      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      #for k, v in self.params.iteritems():
        #self.params[k] = v.astype(dtype)
    

    def loss(self, X, y=None):
      """
      Evaluate loss and gradient for the three-layer convolutional network.
      
      Input / output: Same API as TwoLayerNet in fc_net.py.
      """
      
      
      scores = None
      ############################################################################
      # TODO: Implement the forward pass for the three-layer convolutional net,  #
      # computing the class scores for X and storing them in the scores          #
      # variable.                                                                #
      ############################################################################
      reg = self.reg
      mode = 'test' if y is None else 'train'
      pre_input = X
      layers = self.layers
      self.layer1 = None
      # print 'layer mean %f' % np.mean(pre_input)
      # print 'layer std %f' % np.std(pre_input)
      for layer in layers:
        pre_input, cache = layer_forward(pre_input, layer['func'], layer['func_params'], mode)
        # print 'layer mean %f' % np.mean(pre_input)
        # print 'layer std %f' % np.std(pre_input)
        if self.layer1 is None:
            self.layer1 = pre_input
        layer['cache'] = cache
      
      
      i = len(layers) + 1  
      w = "W" + str(i)
      b = "b" + str(i)
      W = self.params[w]
      B = self.params[b]
      layer, cache = layer_forward(pre_input,'affine', {'w':W,'b':B}, mode)
      # print 'layer mean %f' % np.mean(pre_input)
      # print 'layer std %f' % np.std(pre_input)
      scores = layer
    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################
    
      if y is None:
        return scores
    
      loss, grads = 0, {}
      ############################################################################
      # TODO: Implement the backward pass for the three-layer convolutional net, #
      # storing the loss and gradients in the loss and grads variables. Compute  #
      # data loss using softmax, and make sure that grads[k] holds the gradients #
      # for self.params[k]. Don't forget to add L2 regularization!               #
      ############################################################################
      data_loss, dscores = softmax_loss(scores, y)
      dout, dparams = layer_backward('affine', dscores, cache)
      grads[w] = dparams[0]
      grads[b] = dparams[1]
      
      reg_loss =0
      reg_loss += weight_reg_loss(W, reg)
      for i, layer in reversed(list(enumerate(layers))):
        idx = i+1
        dout, d_params = layer_backward(layer['func'], dout, layer['cache'])
        if 'residual' not in layer['func']:
          reg_loss += weight_reg_loss(layer['gradient_params']['W'], reg)
          grads['W' + str(idx)], grads['b' + str(idx)] = d_params[:2]
          grads['W' + str(idx)] += reg * self.params['W' + str(idx)]
          if len(d_params) == 4:
            grads['gamma'+ str(idx)], grads['beta'+ str(idx)] = d_params[2:]
        else:
          reg_loss += weight_reg_loss(layer['gradient_params']['FW'], reg)
          reg_loss += weight_reg_loss(layer['gradient_params']['SW'], reg)
          grads['FW' + str(idx)], grads['Fb' + str(idx)] = d_params[:2]
          grads['FW' + str(idx)] += reg * self.params['FW' + str(idx)]
          grads['SW' + str(idx)], grads['Sb' + str(idx)] = d_params[2:4]
          grads['SW' + str(idx)] += reg * self.params['SW' + str(idx)]
          if len(d_params) > 4:
              grads['Fgamma' + str(idx)], grads['Fbeta' + str(idx)] = d_params[4:6]
              grads['Sgamma' + str(idx)], grads['Sbeta' + str(idx)] = d_params[6:8]
        
        
        
      loss = data_loss + reg_loss
    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################
      self.grads = grads
      return loss, grads

    def get_params(self, layer, input_dim, dtype):
        weight_scale = self.weight_scale
        func = layer['func'] 
        param = layer['params']
        if self.use_xavier:
            shape = input_dim
            if  'num_filters' in param:
                shape = param['num_filters'] * param['filter_size'] * param['filter_size']
            elif 'affine' in func:
                shape = input_dim if isinstance(input_dim, int) else np.prod(input_dim) 
            if 'relu' in func or 'residual' in func:
                weight_scale = xavier_relu_weight_scale(shape)
            else:
                weight_scale = xavier_weight_scale(shape)
        gradient_params = {}
        func_params = {}
        if 'affine' in func:
            dst_dim = param['hidden_dim']
            src_dim = input_dim
            if isinstance(input_dim, tuple):
                src_dim = 1
                for d in input_dim:
                    src_dim *= d
            w = np.random.randn(src_dim, dst_dim) * weight_scale
            b = np.zeros(dst_dim)
            w = w.astype(dtype)
            b = b.astype(dtype)
            gradient_params["W"] = w
            gradient_params['b'] = b
            func_params['w'] = w
            func_params['b'] = b
            if 'batchnorm' in func or 'batch_norm' in func:
                batch_dim = dst_dim[0] if 'spatial' in func else dst_dim
                gamma = np.ones(batch_dim)
                beta = np.zeros(batch_dim)
                gamma = gamma.astype(dtype)
                beta = beta.astype(dtype)
                gradient_params["gamma"] = gamma
                gradient_params["beta"] = beta
                func_params['gamma'] = gamma
                func_params['beta'] = beta
                func_params['bn_param'] = {}
        elif 'conv' in func:
            num_filters = param['num_filters']
            filter_size = param['filter_size']
            stride = param.get('stride', 1)
            pad = param.get('pad', (filter_size - 1) / 2)
            dst_dim = (num_filters, 1 + (input_dim[1] + 2 * pad - filter_size) / stride, 1 + (input_dim[2] + 2 * pad - filter_size) / stride)
            w = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
            b = np.zeros(num_filters)
            w = w.astype(dtype)
            b = b.astype(dtype)
            gradient_params["W"] = w
            gradient_params['b'] = b
            func_params['w'] = w
            func_params['b'] = b
            func_params['conv_param'] = {'pad': pad, 'stride': stride}
            if 'batchnorm' in func or 'batch_norm' in func:
                batch_dim = dst_dim[0] if 'spatial' in func else dst_dim
                gamma = np.ones(batch_dim)
                beta = np.zeros(batch_dim)
                gamma = gamma.astype(dtype)
                beta = beta.astype(dtype)
                gradient_params["gamma"] = gamma
                gradient_params["beta"] = beta
                func_params['gamma'] = gamma
                func_params['beta'] = beta
                func_params['bn_param'] = {}
        elif 'residual' in func:
            num_filters = param['num_filters']
            filter_size = param['filter_size']
            stride = 1
            pad = (filter_size - 1) / 2
            dst_dim = input_dim
            fw = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
            fb = np.zeros(num_filters)
            sw = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
            sb = np.zeros(num_filters)
            fw = fw.astype(dtype)
            fb = fb.astype(dtype)
            sw = sw.astype(dtype)
            sb = sb.astype(dtype)
            gradient_params["FW"] = fw
            gradient_params['Fb'] = fb
            gradient_params["SW"] = sw
            gradient_params['Sb'] = sb
            func_params['fw'] = fw
            func_params['fb'] = fb
            func_params['sw'] = sw
            func_params['sb'] = sb
            func_params['conv_param'] = {'pad': pad, 'stride': stride}
            if 'batchnorm' in func:
                batch_dim = input_dim[0]
                fgamma = np.ones(batch_dim, dtype=dtype)
                fbeta = np.zeros(batch_dim, dtype=dtype)
                sgamma = np.ones(batch_dim, dtype=dtype)
                sbeta = np.zeros(batch_dim, dtype=dtype)
                gradient_params["Fgamma"] = fgamma
                gradient_params["Fbeta"] = fbeta
                gradient_params["Sgamma"] = sgamma
                gradient_params["Sbeta"] = sbeta
                func_params['fgamma'] = fgamma
                func_params['fbeta'] = fbeta
                func_params['sgamma'] = sgamma
                func_params['sbeta'] = sbeta
                func_params['fbn_param'] = {}
                func_params['sbn_param'] = {}
        else:
            raise ValueError("{}: is not supportted".format(func))




        if 'pool' in func:
            pool_height = param.get('pool_height', 2)
            pool_width = param.get('pool_width', 2)
            pool_stride = param.get('pool_stride', 2)
            func_params['pool_param'] = {'pool_height':pool_height, 'pool_width':pool_width, 'stride': pool_stride }
            dst_dim = (dst_dim[0], (dst_dim[1] - pool_height) / pool_stride + 1, (dst_dim[2] - pool_width) / pool_stride + 1)

        return dst_dim, gradient_params, func_params




