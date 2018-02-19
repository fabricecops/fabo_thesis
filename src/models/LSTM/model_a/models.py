#  hidden_neurons = 50
#  model = Sequential()
#  ##Encoder
#  model.add(LSTM(hidden_neurons,
#            batch_input_shape=(batch_size, n_prev, 1),
#            forget_bias_init='one',
#            return_sequences=False,
#            stateful=True))
# model.add(Dense(hidden_neurons))
# model.add(RepeatVector(n_nxt))
# ##Decoder
# model.add(LSTM(hidden_neurons,
#            batch_input_shape=(batch_size, n_prev, 1),
#            forget_bias_init='one',
#            return_sequences=True,
#            stateful=True))


 # model = Sequential()
 # model.add(LSTM(self.dimension, return_sequences=self.dict_c['pred_seq'],
 #                stateful=True,
 #                batch_input_shape=self.input_shape))
 # model.add(LSTM(self.dimension, return_sequences=self.dict_c['pred_seq'],
 #                stateful=True,
 #                batch_input_shape=self.input_shape))
