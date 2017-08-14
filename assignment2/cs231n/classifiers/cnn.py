import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


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
    
      input_dim = num_filter * width * height
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
    
      i = 1
      num_filters = self.num_filters
      hidden_dims = self.hidden_dims
      num_channel = self.input_dim[0]
      w = self.input_dim[1]
      h = self.input_dim[2]
      layer = X
      layers = {}
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
        j = 0
        dout = max_pool_backward_fast(dout, layers[i]['pool']['cache'])
        for _ in filter_pair:
          w = "W" + str(i)
          b = "b" + str(i)
          dout, dW, dB = conv_relu_backward(dout, layers[i]['cache'])
          dW += reg * self.params[w]
          grads[w] = dW
          grads[b] = dB
          i -=1
          j += 1
    
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################
    
      return loss, grads