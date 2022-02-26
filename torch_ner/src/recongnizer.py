import os
import torch
import time
import numpy as np

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from tqdm import tqdm

from src.models.bert_crf import BertForTokenClassifier
from src.models.bert_gru import BertGRUForTokenClassifier
from src.utils.utils import log, get_time_idf, make_seed, load_pkl
from src.utils.process import Process, process_data
from src.utils.metircs import ChunkEvaluator
from src.utils.dataset import CustomDataset, collate_fn


class Recongnizer(object):
    def __init__(self, config):
        self.config = config

        log.info('************构建模型*************')

        self.tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)

        if config.model_name == 'Bert_ner':
            log.info('***************构建bert_ner模型********************')
            self.model = BertForTokenClassifier(config)
            self.config.model_path = os.path.join(config.root_path, 'models/bert_ner.bin')
        else:
            log.info('********************构建bert_gru_ner模型******************')
            self.model = BertGRUForTokenClassifier(config)
            self.config.model_path = os.path.join(config.root_path, 'models/bert_gru_ner.bin')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        if os.path.exists(config.model_path):
            log.info('************加载模型***********')

            state_dict = torch.load(config.model_path)
            self.model.load_state_dict(state_dict)

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

        train_dataset = CustomDataset(train_data)
        dev_dataset = CustomDataset(dev_data)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=self.config.batch_size,
                                       drop_last=True,
                                       shuffle=True,
                                       collate_fn=collate_fn)
        dev_data_loader = DataLoader(dev_dataset,
                                     batch_size=self.config.batch_size,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=collate_fn)

        end_time = get_time_idf(start_time)

        log.info(f'*********数据预处理完成， 用时{end_time}**********')

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # 优化的参数
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy': 0.0}
        ]
        # 优化器
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.config.learning_rate)

        self.model.train()

        total_batch = 0  # 记录进行多少batch
        dev_best_loss = float('inf')  # 记录上次最好的验证集loss
        last_improve = 0  # 记录上次提升的batch
        flag = False  # 停止位的标志, 是否很久没提升

        metric = ChunkEvaluator(label_list=label_list, suffix=True)
        log.info("***** Running training *****")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            log.info('Epoch [{}/{}]'.format(epoch + 1, self.config.epochs))
            for i, batch in enumerate(train_data_loader):
                torch.cuda.empty_cache()
                *x, y = [data.to(self.device) for data in batch]
                loss, pred, seq_len = self.model(x, y)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                n_infer, n_label, n_correct = metric.compute(seq_len, pred, y)
                metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

                if total_batch % 100 == 0:  # 每训练50次输出在训练集和验证集上的效果
                    precision, recall, f1_score = metric.accumulate()

                    dev_precision, dev_recall, dev_f1_score, dev_loss = self.evaluate(self.model, dev_data_loader,
                                                                                      label_list)

                    if dev_best_loss > dev_loss:
                        dev_best_loss = dev_loss

                        torch.save(self.model.state_dict(), self.config.model_path)
                        improve = '+'
                        last_improve = total_batch
                    else:
                        improve = '-'

                    time_idf = get_time_idf(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.6}, Train precision: {2:>6.2%}, Train recall: {3:>6.2%}, ' \
                          'Train f1_score: {4:6.2%}, Val Loss: {5:>5.6}, Val precision: {6:>6.2%}, ' \
                          'Val recall: {7:6.2%}, Val f1_score: {8:6.2%}, Time: {9}  {10}'
                    log.info(msg.format(total_batch, loss.item(), precision, recall, f1_score,
                                        dev_loss, dev_precision, dev_recall, dev_f1_score, time_idf, improve))
                    self.model.train()

                total_batch = total_batch + 1

                if total_batch - last_improve > self.config.require_improvement:
                    # 在验证集上loss超过1000batch没有下降, 结束训练
                    log.info('在验证集上loss超过10000次训练没有下降, 结束训练')
                    flag = True
                    break

            if flag:
                break

    def evaluate(self, model, dev_iter, label_list):
        """
        模型评估:
        :param model:
        :param dev_iter:
        :param label_list:
        :return: precision, recall, f1_score, loss
        """
        model.eval()
        loss_total = []
        metric = ChunkEvaluator(label_list=label_list, suffix=True)
        for batch in dev_iter:
            *x, y = [data.to(self.device) for data in batch]
            loss, pred, seq_len = model(x, y)

            loss_total.append(loss.item())

            n_infer, n_label, n_correct = metric.compute(seq_len, pred, y)
            metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

        precision, recall, f1_score = metric.accumulate()

        return precision, recall, f1_score, np.mean(loss_total)

    def predict(self, texts):
        """
        模型预测
        :param texts:
        :return:
        """
        preds = []
        label_map = load_pkl(self.config.label_list_path, 'label_map')
        label_map = {i: label for label, i in label_map.items()}
        self.model.eval()
        if not isinstance(texts, list):
            texts = [texts]
        input_ids, token_type_ids, seq_length = [], [], []
        attention_mask = []
        for text in tqdm(texts):
            data_encoding = self.tokenizer.encode_plus(text=text, truncation=True, max_length=self.config.max_len,
                                                       padding='max_length')

            input_ids.append(data_encoding['input_ids'])
            token_type_ids.append(data_encoding['token_type_ids'])
            attention_mask.append(data_encoding['attention_mask'])
            seq_length.append(len(text)+2)

        dataset = CustomDataset((input_ids, token_type_ids, attention_mask, seq_length))
        data_loader = DataLoader(dataset, collate_fn=collate_fn)
        for batch in tqdm(data_loader):
            pred_str = ''
            x = [data.to(self.device) for data in batch]
            _, pred, seq_len = self.model(x)
            seq_len = seq_len.item()
            pred = pred.squeeze(0).cpu().numpy()
            pred = pred[:seq_len]
            for p in pred:
                pred_str = pred_str + label_map[p] + ""
            preds.append(pred_str)


