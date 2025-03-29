# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import keras
# from keras import Model
# from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM
# from keras.losses import MeanSquaredError as MSE, MeanAbsoluteError as MAE, Huber
# from keras.metrics import MeanAbsoluteError, MeanSquaredError
# from tcn import TCN
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# import joblib

# # Load the MinMaxScaler from the 'scaler.save' file
# scaler = joblib.load('corrector-ae.scaler.joblib')

# batch_size = 1024
# epochs = 1000
# window_size = 32

# def windowed_dataset_single(data, window_size, batch_size):
#     """
#     Create a windowed dataset from the input data.

#     Parameters:
#     data (np.array): The input data array.
#     window_size (int): The size of each window.
#     batch_size (int): The size of each batch.

#     Returns:
#     tf.data.Dataset: The windowed dataset.
#     """
#     dataset = tf.data.Dataset.from_tensor_slices(data)
#     dataset = dataset.window(window_size, shift=window_size, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(window_size))
#     dataset = dataset.map(lambda window: tf.expand_dims(window, axis=-1))
#     dataset = dataset.batch(batch_size).prefetch(1)
#     return dataset

# class RootMeanSqauredError(MeanSquaredError):
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         return super().update_state(y_true, y_pred, sample_weight)

#     def result(self):
#         return tf.sqrt(super().result())

# def build_model():
#     """
#     Build a CONV-LSTM model for error correction.
    
#     Returns:
#     keras.Model: The compiled model
#     """
#     # Input layer
#     inputs = Input(shape=(window_size, 1))
    
#     # # Encoder: progressively extract features with additional regularization
#     x = TCN(
#         nb_filters=256,
#         return_sequences=True,
#         dilations=[1, 2, 4, 8],
#         use_layer_norm=True,
#         use_skip_connections=True,
#         dropout_rate=0.15
#     )(inputs)
#     x = BatchNormalization()(x)
    
#     # Bottleneck layer with stronger regularization
#     bottleneck = Dense(
#         64, 
#         activation=None,
#         kernel_regularizer=keras.regularizers.l2(1e-4)
#     )(x)
#     bottleneck = Dropout(0.15)(bottleneck)
    
#     # Decoder with additional regularization
#     x = TCN(
#         nb_filters=256,
#         return_sequences=True,
#         dilations=[1, 2, 4, 8],
#         use_layer_norm=True,
#         dropout_rate=0.15
#     )(x)
#     x = BatchNormalization()(x)
#     # x = Dense(
#     #     16, 
#     #     activation='relu',
#     #     kernel_regularizer=keras.regularizers.l2(1e-4)
#     # )(x)
#     # x = Dense(
#     #     8,
#     #     activation='relu',
#     #     kernel_regularizer=keras.regularizers.l2(1e-4)
#     # )(x)
#     # Output layer with L2 regularization
#     # x = LSTM(
#     #     units=32,
#     #     return_sequences=True,
#     #     kernel_regularizer=keras.regularizers.l2(1e-4),
#     #     recurrent_regularizer=keras.regularizers.l2(1.2e-4)
#     # )(inputs)
#     # x = Dense(
#     #     16,
#     #     activation=None,
#     #     kernel_regularizer=keras.regularizers.l2(1e-4)
#     # )(x)
#     # x = LSTM(
#     #     units=32, 
#     #     return_sequences=True,
#     #     kernel_regularizer=keras.regularizers.l2(1e-4),
#     #     recurrent_regularizer=keras.regularizers.l2(1.2e-4)
#     # )(x)
#     outputs = Dense(
#         1
#     )(x)
    
#     # Modified learning rate schedule with lower initial rate
#     lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=1e-3,  # Lower initial learning rate
#         decay_steps=1000,            # More frequent decay
#         decay_rate=0.9              # More aggressive decay
#     )

#     # Create and compile model with Huber loss
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),  # Adam instead of Nadam
#         loss=MSE(),  # Huber loss is more robust than MAE
#         metrics=[
#             MeanSquaredError(name='mse'),
#             MeanAbsoluteError(name='mae'),
#             RootMeanSqauredError(name='rmse')
#         ]
#     )
    
#     return model

# model = build_model()
# model.summary()
# model.load_weights('./corrector-ae.weights.h5')

# # Load the 'beirut-daily-precipitation.csv' file
# beirut_df = pd.read_csv('data/beirut-hourly-precipitation.csv')

# # Create a windowed dataset from df['value'] without shuffling
# beirut_values = beirut_df['value'].values
# beirut_values = scaler.transform(beirut_values.reshape(-1, 1)).flatten()
# beirut_dataset = windowed_dataset_single(beirut_values, window_size, batch_size)

# # Pass the windowed dataset to the model for inference
# predictions = model.predict(beirut_dataset)

# # Unwrap the windows
# corrected_values = predictions.flatten()

# # Inverse transform the corrected values
# corrected_values = scaler.inverse_transform(corrected_values.reshape(-1, 1)).flatten()

# # corrected_values = scaler.inverse_transform(corrected_values.reshape(-1, 2)).flatten()
# print(len(corrected_values))
# print(len(beirut_df))

# # Save the outputs to 'beirut-daily-corrected.csv'
# beirut_df = beirut_df[:len(corrected_values)]
# beirut_df['value'] = corrected_values
# beirut_df.to_csv('data/beirut-hrly-corrected-full.csv', index=False)
# print("Corrected values have been saved to 'beirut-hrly-corrected-full.csv'.")