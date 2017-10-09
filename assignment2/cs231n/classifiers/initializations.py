import numpy as np

def get_fan_in(shape):
    fan_in = shape if isinstance(shape, int) else shape[0] if len(shape) == 2 else np.prod(shape[1:])
    return fan_in

def xavier_relu_weight_scale(shape):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in = get_fan_in(shape)
    s = np.sqrt(2. / fan_in)
    return s

def xavier_weight_scale(shape):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in = get_fan_in(shape)
    s = np.sqrt(1. / fan_in)
    return s