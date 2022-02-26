import os


class Config(object):
    model_name = 'bert_GRU_ner'

    curPath = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(os.path.split(curPath)[0])[0]

    bert_base_path = os.path.join(root_path, 'bert-base-chinese')

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

    train = True

    num_tags = 57

    drop_out = 0.3
    max_len = 50

    epochs = 30
    num_layers = 8
    batch_size = 32
    hidden_size = 128

    learning_rate = 2e-5
    require_improvement = 5000


config = Config()
