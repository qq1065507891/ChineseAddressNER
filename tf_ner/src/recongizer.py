import tensorflow as tf
import time

import os
os.environ['TF_KERAS'] = '1'

from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from src.models.bert_crf import BertForTokenClassifier
from src.models.bert_gru_crf import BertGRUForTokenClassifier
from src.utils.utils import log, get_time_idf, make_seed, load_pkl, map_example_to_dict, training_curve
from src.utils.process import Process, process_data


class Recongnizer(object):
    def __init__(self, config):
        self.config = config

        self.tokenizer = Tokenizer(config.dict_path, do_lower_case=True)

        if config.model_name == 'bert_crf':
            self.config.model_path = os.path.join(self.config.root_path, 'models/bert_ner.h5')
        else:
            self.config.model_path = os.path.join(self.config.root_path, 'models/bert_gru_ner.h5')

        if os.path.exists(config.model_path):
            log.info('************加载模型***********')
            self.model = self.load_model()
        else:
            log.info('************构建模型*************')
            self.model = self.init_model()

    def init_model(self):
        if self.config.model_name == 'bert_crf':
            log.info('***************构建bert_ner模型********************')
            model, crf = BertForTokenClassifier(self.config).build_model()


        else:
            log.info('********************构建bert_gru_ner模型******************')
            model, crf = BertGRUForTokenClassifier(self.config).build_model()

        optimizer = tf.keras.optimizers.Adamax(learning_rate=self.config.learning_rate)
        loss = crf.sparse_loss
        acc = crf.sparse_accuracy
        model.compile(optimizer=optimizer, loss=loss, metrics=acc)
        return model

    def load_model(self):
        model = self.init_model()
        model.load_weights(self.config.model_path)
        return model

    def fit(self):
        make_seed(1001)

        log.info('**********数据预处理************')

        start_time = time.time()

        process = Process(self.config)

        train_examples = process.get_train_examples()
        dev_examples = process.get_dev_examples()
        label_list = process.get_labels()

        train_data = process_data(self.config.out_path, train_examples, self.tokenizer,
                                  self.config.max_len, label_list, 'train')
        dev_data = process_data(self.config.out_path, dev_examples, self.tokenizer,
                                self.config.max_len, label_list, 'dev')

        train_loader = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1], train_data[2]))\
            .shuffle(200).batch(self.config.batch_size, drop_remainder=True).map(map_example_to_dict)

        dev_loader = tf.data.Dataset.from_tensor_slices((dev_data[0], dev_data[1], dev_data[2]))\
            .batch(self.config.batch_size, drop_remainder=True).map(map_example_to_dict)

        end_time = get_time_idf(start_time)

        log.info(f'*********数据预处理完成， 用时{end_time}**********')

        log.info("***** Running training *****")

        start_time = time.time()

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='max'),
            ModelCheckpoint(self.config.model_path, monitor='val_loss', verbose=1, save_best_only=True,
                            mode='min', period=1, save_weights_only=True)
        ]

        self.model.summary()

        tf.keras.backend.clear_session()

        history = self.model.fit(train_loader,
                                 epochs=self.config.epochs,
                                 validation_data=dev_loader,
                                 callbacks=callbacks)
        end_time = get_time_idf(start_time)
        log.info(f'训练完成， 用时: {end_time}')

        training_curve(history.history['loss'], history.history['sparse_accuracy'],
                       history.history['val_loss'], history.history['val_sparse_accuracy'])

    def predict(self, texts):
        """
        模型预测
        :param texts:
        :return:
        """
        preds = []
        label_map = load_pkl(self.config.label_list_path, 'label_map')
        label_map = {i: label for label, i in label_map.items()}
        if not isinstance(texts, list):
            texts = [texts]

        for text in tqdm(texts):
            input_ids, token_type_ids = self.tokenizer.encode(text, maxlen=self.config.max_len)
            input_ids = sequence_padding([input_ids])
            token_type_ids = sequence_padding([token_type_ids])
            predict = self.model.predict((input_ids, token_type_ids))[0]
            predict = tf.argmax(predict, axis=-1)
            predict = [label_map.get(x) for x in predict.numpy()]
            preds.append(predict)
        log.info(preds)
