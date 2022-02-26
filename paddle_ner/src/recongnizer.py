import os
import paddle
import time
import numpy as np

from paddlenlp.transformers import ErnieTokenizer, LinearDecayWithWarmup
from paddlenlp.layers import LinearChainCrfLoss
from paddle.io import DataLoader, TensorDataset
from paddlenlp.metrics import ChunkEvaluator
from tqdm import tqdm

from src.models.bert_ner import Bert_ner
from src.models.bert_gru import Bert_gru_ner
from src.utils.utils import log, get_time_idf, make_seed, load_pkl
from src.utils.process import Process, process_data


class Recongnizer(object):
    def __init__(self, config):
        self.config = config
        log.info('************构建模型*************')
        self.tokenizer = ErnieTokenizer.from_pretrained(config.MODEL_NAME)
        if config.model_name == 'Bert_ner':
            log.info('***************构建bert_ner模型********************')
            self.model = Bert_ner(config)
            self.config.model_path = os.path.join(config.root_path, 'models/bert_ner.pdparams')

        else:
            log.info('********************构建bert_gru_ner模型******************')
            self.model = Bert_gru_ner(config)
            self.config.model_path = os.path.join(config.root_path, 'models/bert_gru_ner.pdparams')

        if os.path.exists(config.model_path):
            log.info('************加载模型***********')
            state_dict = paddle.load(config.model_path)
            self.model.set_state_dict(state_dict)

    def fit(self):
        make_seed(1001)

        log.info('**********数据预处理************')

        start_time = time.time()

        process = Process(self.config)
        
        train_examples = process.get_train_examples()
        dev_examples = process.get_dev_examples()
        label_list = process.get_labels()

        train_data = process_data(self.config.out_path, self.config, train_examples,
                                  self.tokenizer, self.config.max_len, label_list, 'train')
        dev_data = process_data(self.config.out_path, self.config, dev_examples,
                                self.tokenizer, self.config.max_len, label_list, 'dev')

        train_data_loader = DataLoader(train_data,
                                       batch_size=self.config.batch_size,
                                       drop_last=True,
                                       shuffle=True,
                                       num_workers=0)
        dev_data_loader = DataLoader(dev_data,
                                     batch_size=self.config.batch_size,
                                     drop_last=True,
                                     shuffle=True,
                                     num_workers=0)

        end_time = get_time_idf(start_time)

        log.info(f'*********数据预处理完成， 用时{end_time}**********')

        num_training_steps = len(train_data_loader) * self.config.epochs
        # 学习率
        lr_scheduler = LinearDecayWithWarmup(self.config.learning_rate, num_training_steps, 0.0)
        # 衰减的参数
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        # 梯度剪切
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        # 优化器
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=clip
        )
        self.model.train()
        loss_fn = LinearChainCrfLoss(self.model.crf)
        metric = ChunkEvaluator(label_list=label_list, suffix=True)
        total_batch = 0  # 记录进行多少batch
        dev_best_loss = float('inf')  # 记录上次最好的验证集loss
        last_improve = 0  # 记录上次提升的batch
        flag = False  # 停止位的标志, 是否很久没提升

        log.info("***** Running training *****")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            log.info('Epoch [{}/{}]'.format(epoch + 1, self.config.epochs))
            for i, batch in enumerate(train_data_loader):
                *x, y = batch

                outputs, pred, seq_len = self.model(x)

                loss = loss_fn(outputs, seq_len, y)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                n_infer, n_label, n_correct = metric.compute(seq_len, pred, y)
                metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

                if total_batch % 100 == 0:  # 每训练50次输出在训练集和验证集上的效果
                    precision, recall, f1_score = metric.accumulate()

                    dev_precision, dev_recall, dev_f1_score, dev_loss = self.evaluate(self.model, dev_data_loader, label_list)

                    if dev_best_loss > dev_loss:
                        dev_best_loss = dev_loss

                        paddle.save(self.model.state_dict(), self.config.model_path)
                        improve = '+'
                        last_improve = total_batch
                    else:
                        improve = '-'

                    time_idf = get_time_idf(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.6}, Train precision: {2:>6.2%}, Train recall: {3:>6.2%}, ' \
                          'Train f1_score: {4:6.2%}, Val Loss: {5:>5.6}, Val precision: {6:>6.2%}, ' \
                          'Val recall: {7:6.2%}, Val f1_score: {8:6.2%}, Time: {9}  {10}'
                    log.info(msg.format(total_batch, paddle.mean(loss).item(), precision, recall, f1_score,
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

    @paddle.no_grad()
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
        loss_fn = LinearChainCrfLoss(model.crf)
        metric = ChunkEvaluator(label_list=label_list, suffix=True)
        for batch in dev_iter:
            *x, y = batch
            outputs, pred, seq_len = model(x)
            loss = loss_fn(outputs, seq_len, y)
            loss_total.append(paddle.mean(loss))
            
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
        for text in tqdm(texts):
            data_encoding = self.tokenizer.encode(text, max_seq_len=self.config.max_len,
                                                  pad_to_max_seq_len=True, return_length=True)
            input_ids.append(data_encoding['input_ids'])
            token_type_ids.append(data_encoding['token_type_ids'])
            seq_length.append(data_encoding['seq_len'])

        input_ids = paddle.to_tensor(input_ids, dtype='int64')
        token_type_ids = paddle.to_tensor(token_type_ids, dtype='int64')
        seq_length = paddle.to_tensor(seq_length, dtype='int64')

        data = TensorDataset([input_ids, token_type_ids, seq_length])
        data_loader = DataLoader(data)
        for batch in tqdm(data_loader):
            pred_str = ''
            _, pred, _ = self.model(batch)
            pred = pred.squeeze(0)
            for p in pred.tolist():
                pred_str = pred_str + label_map[p] + ""
            preds.append(pred_str)
        print(preds)
