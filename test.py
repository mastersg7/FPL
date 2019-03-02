import keras
from load_data import load_dataset_rnn
import numpy as np
import tensorflow as tf
import pickle


###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Assign the type of memory allocator
config.gpu_options.allocator_type = 'BFC'

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.65

# check version of keras
print(keras.__version__)


window_len = 30
test_size = 0.2
n_flights = 100

with open('sample_data.pkl', "rb") as f:
    x = pickle.load(f)

x_train, x_test, y_train, y_test = x
n_features = x_train.shape[2]
nflights = x_train.shape[0]

dropout = 0.
epochs = 150
batch_size = 30
optimizer = 'adam'
hl1 = 512
hl2 = 512


def reshape_test(window_len, data):
    """
    splits targets into 3d multi-output matrices
    :param data:    multi-dimensional matrix (m, n, 3 ) of the multi-targets
    :return:        multiple multidimensional matrices of the multi-targetsL: 3 x (m, n, 1)
    """

    target_CTE_train = data[:, :, 0].reshape(data.shape[0], window_len, -1)
    target_ATE_train = data[:, :, 1].reshape(data.shape[0], window_len, -1)
    target_PAE_train = data[:, :, 2].reshape(data.shape[0], window_len, -1)

    return target_CTE_train, target_ATE_train, target_PAE_train


def rmse(y_true, y_pred):

    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=1))


def mse(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=1)


input_ = keras.layers.Input(
            shape=(window_len, n_features)
        )

# masking is to make sure the model doesn't fit the zero paddings
masking = keras.layers.Masking(mask_value=0.)(input_)

# hidden layer 1 with he_normal initializer. return sequence is to return the out of the layer in the same
# shape as the input layer
lstm_h1 = keras.layers.LSTM(hl1, dropout=dropout, kernel_initializer='he_normal',
                            return_sequences=True)(masking)

# hidden layer 1
lstm_h2 = keras.layers.LSTM(hl2, dropout=dropout, kernel_initializer='he_normal',
                            return_sequences=True)(lstm_h1)

# dense output layer of single output
cte = keras.layers.Dense(
    1,
    activation='linear',
    name='CTE',
)(lstm_h2)

ate = keras.layers.Dense(
    1,
    activation='linear',
    name='ATE',
)(lstm_h2)

pae = keras.layers.Dense(
    1,
    activation='linear',
    name='PAE',
)(lstm_h2)

model = keras.models.Model(
    inputs=input_,
    outputs=[cte, ate, pae]
)

pi = [rmse]
model.compile(loss=mse, optimizer=optimizer, metrics=pi)
model.summary()

y_train_reshaped = list(reshape_test(window_len, y_train))

model.fit(
            x_train,

            {
                'CTE': y_train_reshaped[0],
                'ATE': y_train_reshaped[1],
                'PAE': y_train_reshaped[2],
            },

            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle='batch',
            validation_data=(x_train, y_train_reshaped),

        )

history = model.history.history
print("---------------------------------------------------------------")
print('Training CTE RMSE Validation', history['CTE_rmse'])
print("---------------------------------------------------------------")
print('Test CTE RMSE Validation', history['val_CTE_rmse'])
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print('Training ATE RMSE Validation', history['ATE_rmse'])
print("---------------------------------------------------------------")
print('Test ATE RMSE Validation', history['val_ATE_rmse'])
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print('Training PAE RMSE Validation', history['PAE_rmse'])
print("---------------------------------------------------------------")
print('Test PAE RMSE Validation', history['val_PAE_rmse'])
print("---------------------------------------------------------------")

yhat_ = model.predict(x_train, batch_size=batch_size)
yhat = np.concatenate((yhat_[0], yhat_[1], yhat_[2]), axis=2)
print('end')