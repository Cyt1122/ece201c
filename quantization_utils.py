import numpy as np
import tensorflow as tf
import math
import struct
import functools
'''
utility function

with with block.suppress_stdout_stderr():
    your code

To hide stdout/stderr output i.e. from Tensorflow initialzation    
'''
from . import suppress_stdout_stderr as block




def tf_symbolic_convert(value, wl, fl):
    '''
    Convert float numpy array to wl-bit low precision data with Tensorflow API

    Inputs：
    - value : a numpy array of input data
    - wl : word length of the data format to convert
    - fl : fraction length (exponent length for floating-point)

    Returns:
    - val_fp : tf.Tensor as the symbolic expression for quantization
    '''
    # ================================================================ #
    # YOUR CODE HERE:
    #   tf.clip_by_value could be helpful
    # ================================================================ #
    ex = 0.0000001
    exponent = tf.math.ceil(tf.math.divide_no_nan(tf.math.log(tf.math.abs(value \
                    + tf.cast(ex, tf.float32))), tf.math.log(2.0)))
    mantissa = (tf.math.abs(value) / tf.cast(tf.math.pow(2.0, exponent), tf.float32))
    left = tf.cast(tf.math.round(mantissa / tf.math.pow(2.0, -fl)), tf.float32) \
                    * tf.cast(tf.math.pow(2.0, -fl), tf.float32)
    exponent = tf.clip_by_value(exponent, -(tf.math.pow(2.0, wl - fl - 1) - 1), \
                    tf.math.pow(2.0, wl - fl - 1))
    val_fp = left * tf.cast(tf.math.pow(2.0, exponent), tf.float32) * \
                    tf.math.divide_no_nan(tf.math.abs(value), value)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return val_fp


class Qnn:
    def __init__(self):
        pass

# dtype convertion: basic functions
    def find_best_sol (self, value, wl, fl):
        [m, e] = np.frexp(value)
        m = np.round(m / np.power(2.0, -fl)) * np.power(2.0, -fl)
        e = np.where(e < -(np.power(2.0, wl - fl - 1) - 1),\
                     -(np.power(2.0, wl - fl - 1) - 1), e)
        e = np.where(e > np.power(2.0, wl - fl - 1), np.power(2.0, wl - fl - 1), e)
        val_fp = m * np.power(2.0, e)
        return val_fp

    def to_fixedpoint(self, data_i, word_len, frac_len):

        return tf_symbolic_convert(data_i, word_len, frac_len)

    # utility function to convert symbolically or numerically
    def convert(self, data_i, word_len, frac_len, symbolic=False):
        if symbolic is True:
            data_q = self.to_fixedpoint(data_i, word_len, frac_len)
        else:
            with tf.Graph().as_default():
                data_q = self.to_fixedpoint(data_i, word_len, frac_len)
                with block.suppress_stdout_stderr():
                    with tf.Session() as sess:
                        data_q = sess.run(data_q)
        return data_q

        # error measurement

    def difference(self, data_q, data_origin, wl, fl):
        '''
        Compute the difference before and after quantization

        Inputs：
        - data_q: a numpy array of quantized data
        - data_origin: a numpy array of original data

        Returns:
        - dif : numerical value of quantization error
        '''

        # ================================================================ #
        # YOUR CODE HERE:
        #   implement mean squared error between data_q and data_origin
        # ================================================================ #
        dif = np.sum(np.power((data_q - data_origin), 2)) / data_origin.size
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return dif

    # search policy
    def search(self, data_i, word_len):
        '''
        Search for the optimal fraction length that leads to minimal quantization error for data_i

        Inputs：
        - data_i : a numpy array of original data
        - word_len : word length of quantized data

        Returns:
        - fl_opt : fraction length (python built-in int data type) that leads to minimal quantization error
        '''

        # ================================================================ #
        # YOUR CODE HERE:
        #   compute quantization error for different fraction lengths
        #   and determine fl_opt
        # ================================================================ #
        fl_err = []
        for i in range(word_len):
            data_quan = self.find_best_sol(data_i, word_len, i)
            fl_err.append(self.difference(data_quan, data_i, word_len, i))
        fl_opt = np.argmin(fl_err)
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return int(fl_opt)

    # granularity
    def apply(self, data_i, word_len):
        fl_opt = self.search(data_i, word_len)
        data_q = self.convert(data_i, word_len, fl_opt)
        return data_q, fl_opt
