import torch

from transformers import BertModel

from src.models.crf import CRF


class BertGRUForTokenClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertGRUForTokenClassifier, self).__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.bert_base_path)

        self.gru = torch.nn.GRU(self.bert.config.hidden_size, config.hidden_size,
                                num_layers=config.num_layers, bidirectional=True)
        self.fc = torch.nn.Linear(config.hidden_size*2, config.num_tags)
        self.drop_out = torch.nn.Dropout(config.drop_out)

        self.crf = CRF(config.num_tags, batch_first=True)

    def forward(self, batch, num_tags=None):
        """

        :param batch:【input_ids, token_type_ids, attention_mask, seq_length】
        :return:
        """
        input_ids, token_type_ids, attention_mask, seq_length = batch

        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        output = self.drop_out(output)
        gru, _ = self.gru(output)
        classifier = self.fc(gru)

        if num_tags is not None:
            loss = -self.crf(classifier, num_tags)
        else:
            loss = 0

        pred = self.crf.decode(classifier)
        pred = torch.tensor(pred)

        return loss, pred, seq_length
