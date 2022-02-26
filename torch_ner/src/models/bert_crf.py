import torch

from transformers import BertModel

from src.models.crf import CRF


class BertForTokenClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertForTokenClassifier, self).__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.bert_base_path)

        self.fc = torch.nn.Linear(self.bert.config.hidden_size, config.num_tags)
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

        classifier = self.fc(output)

        if num_tags is not None:
            loss = self.crf(classifier, self.config.num_tags)
        else:
            loss = 0

        pred = self.crf.decode(classifier, mask=seq_length)
        pred = torch.tensor(pred)

        return loss, pred, seq_length
