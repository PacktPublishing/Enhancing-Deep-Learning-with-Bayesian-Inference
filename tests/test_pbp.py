import numpy as np

from bdl.ch05.pbp_b4tf.main import get_data
from bdl.ch05.pbp_theano.main import get_data as get_data_theano


def test_data_is_same():
    X_train, y_train, X_test, y_test, _, _ = get_data()
    X_train_theano, y_train_theano, X_test_theano, y_test_theano, _, _ = get_data_theano()
    np.testing.assert_array_equal(
        X_train, X_train_theano
    )
    np.testing.assert_array_equal(
        y_train, y_train_theano
    )
    np.testing.assert_array_equal(
        X_test, X_test_theano
    )
    np.testing.assert_array_equal(
        y_test, y_test_theano
    )
