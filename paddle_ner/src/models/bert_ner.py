import paddle.nn as nn

from paddlenlp.transformers import ErnieModel
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder


class Bert_ner(nn.Layer):
    def __init__(self, config):
        super(Bert_ner, self).__init__()
        self.config = config
        self.ernie = ErnieModel.from_pretrained(config.MODEL_NAME)

        self.drop_out = nn.layer.Dropout(config.drop_out)
        self.classifier = nn.layer.Linear(self.ernie.config['hidden_size'], config.num_classes+2)
        self.crf = LinearChainCrf(config.num_classes)
        self.decode = ViterbiDecoder(self.crf.transitions)

    def forward(self, batch):
        """
        前向转播
        :param batch:
        :return:
        """

        input_ids, token_type_ids, seq_len = batch
        # seq_len = seq_len.reshape([self.config.batch_size,])
        seq_len = seq_len.squeeze(1)

        output = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids)[0]

        drop_out = self.drop_out(output)

        classifier = self.classifier(drop_out)
        _, pred = self.decode(classifier, seq_len)
        return classifier, pred, seq_len
