
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import random
import sys

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging

import sys
print(sys.version)


sess = tf.Session()
with sess.as_default():
    tensor = tf.range(10)
    print_op = tf.print("tensors:", tensor, {2: tensor * 2},
                        output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
        tripled_tensor = tensor * 3
    sess.run(tripled_tensor)

# tf.enable_eager_execution()
# tensor = tf.range(10)
# tf.print(tensor, output_stream=sys.stderr)
