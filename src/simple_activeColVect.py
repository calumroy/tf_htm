import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_

# Create a tensorflow for calculating
# for each column a list of columns which that column can
# be inhibited by. Set the winning columns in this list as one.
# If a column is inhibited already then all those positions in
# the colwinners relating to that col are set as one. This means
# the inhibited columns don't determine the active columns


