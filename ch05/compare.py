import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ch05.pbp_b4tf.main import get_data
from ch05.pbp_b4tf.main import fit as fit_b4
from ch05.pbp_theano.main import fit as fit_theano
from ch05.pbp_b4tf.main import predict as predict_b4
from ch05.pbp_theano.main import predict as predict_theano
from ch05.pbp_b4tf.main import plot

def main():
    X_test, X_train, y_train, y_test, x_scaler, y_scaler = get_data()
    pbp_b4 = fit_b4(X_train, y_train, n_epochs=1)
    m_b4, v_b4 = predict_b4(pbp_b4, X_test, y_test, x_scaler, y_scaler)
    plot(X_test, m_b4, v_b4, y_scaler, y_test, title="B4")

    X_test, X_train, y_train, y_test, x_scaler, y_scaler = get_data()
    pbp_theano = fit_theano(X_train, y_train.squeeze(), n_epochs=1)
    m_theano, v_theano = predict_theano(pbp_theano, X_test, y_test, x_scaler, y_scaler)
    plot(X_test, m_theano, v_theano, y_scaler, y_test, title="Theano")


if __name__ == '__main__':
    main()