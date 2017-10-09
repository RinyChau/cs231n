import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      norm_scores = np.exp(X[i].dot(W))
      scores_sum = np.sum(norm_scores)
      probs = norm_scores / scores_sum
      for j in xrange(num_classes):
          prob = probs[j]
          if j == y[i]:
              dW[:,j] += (prob - 1) * X[i]
          else:
              dW[:,j] += prob * X[i]
      correct_prob = probs[y[i]]
      loss += -np.log(correct_prob)

  loss /= num_train
  dW /= num_train
  dW += reg * W
  loss += 0.5 * reg * np.sum(W**2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################
  scores =X.dot(W)
  scores = np.minimum(scores,500)
  scores_norm = np.exp(scores)
  scores_sum = np.sum(scores_norm,axis=1)
  probs = scores_norm / scores_sum[:,np.newaxis]
  loss += -np.sum(np.log(probs[range(num_train),y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  grad_prob = np.copy(probs)
  grad_prob[range(num_train),y] += -1
  dW = X.T.dot(grad_prob)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

