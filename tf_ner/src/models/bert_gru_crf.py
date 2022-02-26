import tensorflow as tf
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow_addons.layers import CRF

import os
os.environ['TF_KERAS'] = '1'

from bert4keras.models import build_transformer_model
from bert4keras.backend import set_gelu
from bert4keras.layers import ConditionalRandomField


class BertGRUForTokenClassifier(object):
    def __init__(self, config):
        self.config = config

        self.bert = build_transformer_model(config_path=config.config_path,
                                            checkpoint_path=config.checkpoint_path,
                                            model='bert',
                                            return_keras_model=False)

        self.gru = Bidirectional(GRU(config.hidden_size, return_sequences=True), name='BRU')

        self.classifier = Dense(config.num_tags, name='output', activation='softmax')

        self.crf = CRF(config.crf_lr_multiplier)

        self.drop_out = Dropout(config.drop_out)

    def build_model(self):
        set_gelu('tanh')
        bert_output = self.bert.model.output

        drop_out = self.drop_out(bert_output)
        gru_output = self.gru(drop_out)

        classifier = self.classifier(gru_output)
        crf = self.crf(classifier)

        model = tf.keras.models.Model(inputs=self.bert.input, outputs=crf, name=self.config.model_name)
        return model, self.crf


