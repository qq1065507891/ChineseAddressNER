import codecs
import os
import torch

from tqdm import tqdm

from src.utils.utils import ensure_dir, save_pkl, load_pkl, log


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, seq_length, attention_mask=None, label_id=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.seq_length = seq_length
        self.label_id = label_id


class Process(object):
    def __init__(self, config):
        self.config = config
        self.label = []

    def read_file(self, path, name='train'):
        with codecs.open(path, 'r', encoding='utf-8') as f:
            all_data = []
            data_line, label_line = '', ''
            for line in tqdm(f.readlines()):
                data = line.strip().split(' ')
                if len(data) == 1:
                    label_line = 'O' + '' + label_line + 'O'
                    all_data.append(InputExample(text_a=data_line, label=label_line, guid=name))
                    data_line, label_line = '', ''
                    continue
                data_line += data[0]
                label_line += data[1] + ''
                self.label.append(data[1])

        return all_data

    def get_train_examples(self):
        """See base class."""
        return self._create_examples("train")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples("test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples("dev")

    def get_labels(self):
        """See base class."""
        if not os.path.exists(self.config.label_list_path):
            label_map = {}
            self.label = list(set(self.label))
            for (i, label) in enumerate(self.label):
                label_map[label] = i
            save_pkl(self.config.label_list_path, label_map, 'label_map', use_bert=True)
        else:
            label_map = load_pkl(self.config.label_list_path, 'label_map')

        self.config.num_classes = len(label_map)
        return label_map

    def _create_examples(self, name='train'):
        ensure_dir(self.config.out_path)
        if name == 'train':
            path = self.config.train_examples_path
        elif name == 'test':
            path = self.config.test_examples_path
        else:
            path = self.config.dev_examples_path

        if os.path.exists(path):
            examples = load_pkl(path, name)
        else:
            if name == 'train':
                examples = self.read_file(self.config.train_path, name)
            elif name == 'dev':
                examples = self.read_file(self.config.dev_path, name)
            else:
                examples = self.read_file(self.config.test_path, name)

            save_pkl(path, examples, name, use_bert=True)
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_map=None):
    """Loads a data file into a list of `InputBatch`s."""

    log.info(f"#examples {len(examples)}")

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        label_ids = []
        data_encoding = tokenizer.encode_plus(text=example.text_a, truncation=True, max_length=max_seq_length,
                                              padding='max_length')

        input_ids = data_encoding['input_ids']
        token_type_ids = data_encoding['token_type_ids']
        attention_mask = data_encoding['attention_mask']
        y = example.label.split('')
        for i in range(len(y)):
            label_ids.append(label_map[y[i]])
        seq_length = len(label_ids)

        if len(label_ids) >= max_seq_length:
            label_ids = label_ids[:max_seq_length]
        else:
            label_ids = label_ids + [-1] * (max_seq_length - len(label_ids))

        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 2:
            log.info("*** Example ***")
            log.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [tokenization.printable_text(x) for x in tokens]))
            log.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            log.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            log.info(f"label: {example.label} (id = {' '.join([str(x) for x in label_ids])})")

        features.append(
            InputFeatures(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                seq_length=seq_length,
                label_id=label_ids))

    log.info(f'#features {len(features)}')
    return features


def process_data(data_dir, examples,  tokenizer, max_seq_length, label_list=None, name='train'):

    feature_dir = os.path.join(data_dir, '{}_{}.pkl'.format(name, max_seq_length))
    if os.path.exists(feature_dir):
        process_features = load_pkl(feature_dir, name)
    else:
        process_features = convert_examples_to_features(examples, max_seq_length, tokenizer, label_list)
        save_pkl(feature_dir, process_features, name, use_bert=True)

    log.info(f" Num examples = {len(process_features)}")

    input_ids = []
    token_type_ids = []
    attention_mask = []
    seq_length = []
    label_id = []
    for f in tqdm(process_features):
        input_ids.append(f.input_ids)
        token_type_ids.append(f.token_type_ids)
        attention_mask.append(f.attention_mask)
        seq_length.append(f.seq_length)
        label_id.append(f.label_id)

    return [input_ids, token_type_ids, attention_mask, seq_length, label_id]

