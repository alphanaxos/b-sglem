import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

def build_bilstm(n_features, time_steps=24, lstm_units=64):
    """
    Builds and compiles the Bidirectional LSTM model.

    Args:
        n_features (int): Number of features in input sequence.
        time_steps (int): Length of each sequence window. Defaults to 24.
        lstm_units (int): Hidden units in first BiLSTM layer. Defaults to 64.

    Returns:
        Sequential: Compiled Keras Sequential model.
    """
    model = Sequential([
        Bidirectional(LSTM(lstm_units, return_sequences=True),
                      input_shape=(time_steps, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_bilstm(X_tr_seq, y_tr_seq, n_features, time_steps=24, lstm_units=64, epochs=20, batch_size=64):
    """
    Builds and trains the BiLSTM model.

    Args:
        X_tr_seq (np.ndarray): Scaled train input sequences.
        y_tr_seq (np.ndarray): Scaled train targets.
        n_features (int): Number of features.
        time_steps (int): Time steps. Defaults to 24.
        lstm_units (int): LSTM units. Defaults to 64.
        epochs (int): Number of epochs. Defaults to 20.
        batch_size (int): Batch size. Defaults to 64.

    Returns:
        tuple: (trained_model, history_object)
    """
    model = build_bilstm(n_features, time_steps, lstm_units)
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
