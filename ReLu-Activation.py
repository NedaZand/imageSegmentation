from keras.initializers import glorot_uniform
import keras
embeding_dtm = 100
model = Sequential()

model.add(Embedding(1000,embeding_dtm, input_length = max_length))
model.add(LSTM(units = 32, dropout = 0.2, recurrent_dropout = 0.2))
model.add(BatchNormalization())
model.add(Dense(3, activation = 'relu'))
