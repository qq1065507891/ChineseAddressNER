import os


class Config(object):
    curPath = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(os.path.split(curPath)[0])[0]

    MODEL_NAME = 'ernie-1.0'

    model_name = 'Bert_ner'

    data_path = os.path.join(root_path, 'data')
    train_path = os.path.join(data_path, 'CCKS2021/train.conll')
    dev_path = os.path.join(data_path, 'CCKS2021/dev.conll')
    test_path = os.path.join(data_path, 'CCKS2021/final_test.txt')

    log_folder_path = os.path.join(root_path, 'log')
    log_path = os.path.join(log_folder_path, 'log.txt')

    out_path = os.path.join(data_path, 'out')

    label_list_path = os.path.join(out_path, 'label_list.pkl')

    train_out_path = os.path.join(out_path, 'train.pkl')
    test_out_path = os.path.join(out_path, 'test.pkl')
    dev_out_path = os.path.join(out_path, 'dev.pkl')

    train_examples_path = os.path.join(out_path, 'train_examples.pkl')
    test_examples_path = os.path.join(out_path, 'test_examples.pkl')
    dev_examples_path = os.path.join(out_path, 'dev_examples.pkl')

    model_path = '' 

    max_len = 50
    drop_out = 0.3
    num_classes = 55

    epochs = 30
    batch_size = 32
    hidden_size = 512

    learning_rate = 5e-5
    require_improvement = 5000

config = Config()