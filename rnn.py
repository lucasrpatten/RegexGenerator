# # RNN
# rnn = Model(inputs=[allowed_input, rejected_input], outputs=output)

from keras.layers import Input, LSTM, Flatten, Bidirectional, Masking, Attention, Dense


def create_rnn(max_input_length, output_size):
    # Input Shapes
    allowed_shape = (None, max_input_length)
    rejected_shape = (None, max_input_length)

    # Input Layers
    allowed_input = Input(shape=allowed_shape)
    rejected_input = Input(shape=rejected_shape)

    # Masking Layers
    allowed_masking = Masking()(allowed_input)
    rejected_masking = Masking()(rejected_input)

    # LSTM Layer
    def lstm_layer(neurons=64):
        return LSTM(neurons, return_sequences=True)

    # Bidirectional LSTM Layer
    def bidirectional_lstm(neurons=64):
        return Bidirectional(lstm_layer(neurons))

    # Pass Through LSTM
    allowed_output = bidirectional_lstm(128)(allowed_masking)
    rejected_output = bidirectional_lstm(128)(rejected_masking)

    # Apply Attention
    attentions = Attention([allowed_output, rejected_output])

    flattened_attentions = Flatten()(attentions)

    output = Dense(output_size, activation="sigmoid")(flattened_attentions)

    return output


MAX_INPUT_LENGTH = 100
OUTPUT_SIZE = 500

rnn = create_rnn(MAX_INPUT_LENGTH, OUTPUT_SIZE)
