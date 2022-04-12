from pathlib import Path

from bdl.pbp.main import get_data, fit, predict
import numpy as np
import tensorflow as tf

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def test_fit():
    X_train, y_train, X_test, y_test = get_data()
    model = fit(X_train, y_train, 1)
    m, v, rmse, test_ll, test_ll_with_noise = predict(model, X_test, y_test, X_train, y_train)
    print(f"{m=}, {v=}, {rmse=}")

    rmse_expected = 6.651303691787285
    test_ll_expected = -3.272592919535482
    test_ll_with_vnoise_expected = -3.235382224490954

    np.testing.assert_almost_equal(rmse, rmse_expected, decimal=5)
    np.testing.assert_almost_equal(test_ll, test_ll_expected, decimal=5)
    np.testing.assert_almost_equal(test_ll_with_noise, test_ll_with_vnoise_expected, decimal=5)

    m_expected = np.load(str(Path(__file__).parent / "resources" / "pbp" / "m_1_epoch_random_seed_0.npy"))
    v_expected = np.load(str(Path(__file__).parent / "resources" / "pbp" / "v_1_epoch_random_seed_0.npy"))

    np.testing.assert_array_almost_equal(m, m_expected, decimal=4)
    np.testing.assert_array_almost_equal(v, v_expected, decimal=4)
