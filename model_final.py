import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

def create_final_model(vocab_size, max_length):
    # Image Feature Extractor Branch
    inputs1 = Input(shape=(4096,), name="image_features")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu', name="image_dense")(fe1)

    # Sequence / Caption Branch
    inputs2 = Input(shape=(max_length,), name="caption_sequence")
    se1 = Embedding(vocab_size, 256, mask_zero=True, name="caption_embedding")(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256, name="caption_lstm")(se2)

    # Decoder / Merge Layer - MATCHING EXACT SAVED WEIGHT NAMES
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu', name="image_cell_state")(decoder1)
    outputs = Dense(vocab_size, activation='softmax', name="word_prediction")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model