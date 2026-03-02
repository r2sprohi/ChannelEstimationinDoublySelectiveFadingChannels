import numpy as np
import sys
import os
import warnings
import pickle
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SNR_INDEX = np.arange(0, 45, 5)  # [0, 5, 10, 15, 20, 25, 30, 35, 40]
# IMPORTANT: nUSC = 52 (from MATLAB: nDSC + nPSC = 48 + 4)
# This means: input_dim = 6 * 52 = 312, output_dim = 2 * 52 = 104

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def load_dataset(path, mode):
    """
    Load dataset from MATLAB .mat file

    Args:
        path: Path to .mat file
        mode: 'train' or 'test'

    Returns:
        X, Y: Input and output arrays
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        mat = loadmat(path)
        data = mat['DNN_Datasets'][0, 0]

        if mode == 'train':
            X, Y = data['Train_X'], data['Train_Y']
        else:
            X, Y = data['Test_X'], data['Test_Y']

        # Validate shapes
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]} samples")

        return X, Y

    except Exception as e:
        raise RuntimeError(f"Error loading dataset from {path}: {str(e)}")


def build_model(input_dim, output_dim, hidden_layers):
    init = TruncatedNormal(mean=0.0, stddev=0.05)

    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu',
                    input_dim=input_dim,
                    kernel_initializer=init,
                    bias_initializer=init))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu',
                        kernel_initializer=init,
                        bias_initializer=init))

    model.add(Dense(output_dim,
                    kernel_initializer=init,
                    bias_initializer=init))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mse']
    )

    return model


# --------------------------------------------------
# Mode selection
# --------------------------------------------------
if len(sys.argv) == 12:
    # ==================================================
    # TRAINING MODE
    # ==================================================
    mobility        = sys.argv[1]
    channel_model   = sys.argv[2]
    modulation      = sys.argv[3]
    training_snr    = sys.argv[4]

    input_dim       = int(sys.argv[5])   # = 6 * nUSC
    output_dim      = int(sys.argv[6])   # = 2 * nUSC

    h1              = int(sys.argv[7])
    h2              = int(sys.argv[8])
    h3              = int(sys.argv[9])

    epochs          = int(sys.argv[10])
    batch_size      = int(sys.argv[11])

    print("=" * 60)
    print("Running TRAINING mode")
    print("=" * 60)
    print(f"Mobility: {mobility}, Channel: {channel_model}, Modulation: {modulation}")
    print(f"Training SNR: {training_snr} dB")
    print(f"Expected dimensions - Input: {input_dim}, Output: {output_dim}")
    print(f"Hidden layers: [{h1}, {h2}, {h3}]")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("=" * 60)

    dataset_path = f'./{mobility}_{channel_model}_{modulation}_DNN_training_dataset_{training_snr}.mat'
    X, Y = load_dataset(dataset_path, mode='train')

    print(f"Training data X: {X.shape}")
    print(f"Training data Y: {Y.shape}")

    # Validate dimensions match expected values
    if X.shape[1] != input_dim:
        raise ValueError(f"Input dimension mismatch! Expected {input_dim}, got {X.shape[1]}")
    if Y.shape[1] != output_dim:
        raise ValueError(f"Output dimension mismatch! Expected {output_dim}, got {Y.shape[1]}")

    # --------------------------------------------------
    # Normalization (FIT ON TRAINING ONLY)
    # --------------------------------------------------
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    Xs = scaler_x.fit_transform(X)
    Ys = scaler_y.fit_transform(Y)

    # Save scalers (using pickle for proper StandardScaler serialization)
    scaler_x_path = f'./scaler_x_{training_snr}.pkl'
    scaler_y_path = f'./scaler_y_{training_snr}.pkl'

    with open(scaler_x_path, 'wb') as f:
        pickle.dump(scaler_x, f)

    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)

    print(f"Saved scalers: {scaler_x_path}, {scaler_y_path}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=[h1, h2, h3]
    )

    print(model.summary())

    model_path = f'./{mobility}_{channel_model}_{modulation}_DNN_{training_snr}.h5'

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True,
                        monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=15, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        Xs, Ys,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.25,
        callbacks=callbacks,
        verbose=1
    )

    print("=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)

else:
    # ==================================================
    # TESTING MODE
    # ==================================================
    mobility        = sys.argv[1]
    channel_model   = sys.argv[2]
    modulation      = sys.argv[3]
    training_snr    = sys.argv[4]

    print("=" * 60)
    print("Running TESTING mode")
    print("=" * 60)
    print(f"Mobility: {mobility}, Channel: {channel_model}, Modulation: {modulation}")
    print(f"Model trained at SNR: {training_snr} dB")
    print("=" * 60)

    # Load trained model
    model_path = f'./{mobility}_{channel_model}_{modulation}_DNN_{training_snr}.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model = load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # Load scalers
    scaler_x_path = f'./scaler_x_{training_snr}.pkl'
    scaler_y_path = f'./scaler_y_{training_snr}.pkl'

    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_x_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_y_path}")

    with open(scaler_x_path, 'rb') as f:
        scaler_x = pickle.load(f)

    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    print(f"Loaded scalers from: {scaler_x_path}, {scaler_y_path}")

    # --------------------------------------------------
    # Loop over SNRs
    # --------------------------------------------------
    print("\nTesting over SNR range:", SNR_INDEX)
    print("=" * 60)

    for snr in SNR_INDEX:

        dataset_path = f'./{mobility}_{channel_model}_{modulation}_DNN_testing_dataset_{snr}.mat'

        if not os.path.exists(dataset_path):
            print(f"WARNING: Dataset not found for SNR={snr} dB, skipping...")
            continue

        X, Y = load_dataset(dataset_path, mode='test')

        print(f"Testing at SNR = {snr} dB | X shape = {X.shape}, Y shape = {Y.shape}")

        # Validate dimensions
        expected_input_dim = model.input_shape[1]
        expected_output_dim = model.output_shape[1]

        if X.shape[1] != expected_input_dim:
            raise ValueError(f"Input dimension mismatch at SNR={snr}! Expected {expected_input_dim}, got {X.shape[1]}")
        if Y.shape[1] != expected_output_dim:
            raise ValueError(f"Output dimension mismatch at SNR={snr}! Expected {expected_output_dim}, got {Y.shape[1]}")

        # Normalize, predict, denormalize
        Xs = scaler_x.transform(X)
        Ys = scaler_y.transform(Y)

        Y_pred_s = model.predict(Xs, verbose=0)
        Y_pred   = scaler_y.inverse_transform(Y_pred_s)

        # Calculate MSE for this SNR
        mse = np.mean((Y - Y_pred) ** 2)
        print(f"  MSE = {mse:.6f}")

        # Save results
        result_path = f'./{mobility}_{channel_model}_{modulation}_DNN_results_{snr}.mat'
        savemat(result_path, {
            'DNN_input_X': X,
            'DNN_true_Y': Y,
            'DNN_predicted_Y': Y_pred
        })

        print(f"  Saved results to {result_path}")

    print("=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)
