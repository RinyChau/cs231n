import numpy as np
import cs231n.layers as layers
import cs231n.layer_utils as layer_utils

def layer_forward(x, func, param, mode):
    func = load_function(func+'_forward')
    if 'bn_param' in param:
        param['bn_param']['mode'] = mode
    if 'fbn_param' in param:
        param['fbn_param']['mode'] = mode
    if 'sbn_param' in param:
        param['sbn_param']['mode'] = mode
    return func(x, **param)

def layer_backward(func, dout, cache):
    func = load_function(func+'_backward')
    result = func(dout, cache)
    result = (result[0], result[1:])
    return result

def load_function(func):
    func = getattr(layers, func) if hasattr(layers, func) else getattr(layer_utils, func)
    return func

def weight_reg_loss(w, reg):
    return 0.5 * reg * np.sum(np.minimum(w,1000000)**2)
    










