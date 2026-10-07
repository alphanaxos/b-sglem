from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_cnn_lstm(n_features, time_steps=24):
    """
    Builds and compiles the hybrid CNN-LSTM model.

    Args:
        n_features (int): Number of input features.
        time_steps (int): Length of the sequence window. Defaults to 24.

    Returns:
        Model: Compiled Keras Functional API model.
    """
    inp  = Input(shape=(time_steps, n_features))
    x    = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
    x    = MaxPooling1D(pool_size=2)(x)
    x    = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x    = LSTM(48)(x)
    x    = Dropout(0.15)(x)
    x    = Dense(16, activation='relu')(x)
    out  = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_cnn_lstm(X_tr_seq, y_tr_seq, n_features, time_steps=24, epochs=20, batch_size=64):
    """
    Builds and trains the CNN-LSTM model.

    Args:
        X_tr_seq (np.ndarray): Scaled train input sequences.
        y_tr_seq (np.ndarray): Scaled train targets.
        n_features (int): Number of features.
        time_steps (int): Time steps. Defaults to 24.
        epochs (int): Number of epochs. Defaults to 20.
        batch_size (int): Batch size. Defaults to 64.

    Returns:
        tuple: (trained_model, history_object)
    """
    model = build_cnn_lstm(n_features, time_steps)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_tr_seq, y_tr_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )
    return model, history
