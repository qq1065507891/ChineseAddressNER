import tensorflow as tf
import tensorflow_addons as tfa


# class ModelWithCRFLoss(tf.keras.Model):
#     """把CRFloss包装成模型，容易扩展各种loss"""
#
#     def __init__(self, base, return_scores=False, **kwargs):
#         super(ModelWithCRFLoss, self).__init__(**kwargs)
#         self.base = base
#         self.return_scores = return_scores
#         self.accuracy_fn = tf.keras.metrics.Accuracy(name="accuracy")
#
#     def call(self, inputs):
#         return self.base(inputs)
#
#     def summary(self):
#         self.base.summary()
#
#     def train_step(self, data):
#         x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
#         with tf.GradientTape() as tape:
#             viterbi_tags, lengths, crf_loss = self.compute_loss(
#                 x, y, sample_weight, training=True
#             )
#         grads = tape.gradient(crf_loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#         mask = tf.sequence_mask(lengths, y.shape[1])
#         self.accuracy_fn.update_state(y, viterbi_tags, mask)
#         results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
#         return results
#
#     def test_step(self, data):
#         x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
#         viterbi_tags, lengths, crf_loss = self.compute_loss(
#             x, y, sample_weight, training=False
#         )
#         mask = tf.sequence_mask(lengths, y.shape[1])
#         self.accuracy_fn.update_state(y, viterbi_tags, mask)
#         results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
#         return results
#
#     def predict_step(self, data):
#         # 预测阶段，模型只返回viterbi tags即可
#         x, *_ = tf.keras.utils.unpack_x_y_sample_weight(data)
#         viterbi_tags, *_ = self(x, training=False)
#         return viterbi_tags
#
#     def compute_loss(self, x, y, sample_weight, training):
#         viterbi_tags, potentials, lengths, trans = self(x, training=training)
#         crf_loss, _ = tfa.text.crf_log_likelihood(potentials, y, lengths, trans)
#         if sample_weight is not None:
#             crf_loss = crf_loss * sample_weight
#         return viterbi_tags, lengths, tf.reduce_mean(-crf_loss)
#
#     def accuracy(self, y_true, y_pred):
#         viterbi_tags, potentials, lengths, trans = y_pred
#         mask = tf.sequence_mask(lengths, y_true.shape[1])
#         return self.accuracy_fn(y_true, viterbi_tags, mask)
#

def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.accuracy_fn = tf.keras.metrics.Accuracy(name="accuracy")

    def call(self, inputs):
        return self.base_model(inputs)

    def summary(self):
        self.base_model.summary()

    def accuracy(self, y_true, y_pred):
        viterbi_tags, potentials, lengths, trans = y_pred
        mask = tf.sequence_mask(lengths, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_tags, mask)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        viterbi_tags, potentials, sequence_length, chain_kernel = y_pred

        # we now add the CRF loss:
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), sum(self.losses), sequence_length, viterbi_tags

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses, sequence_length, viterbi_tags = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        mask = tf.sequence_mask(sequence_length, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)

        return {"crf_loss": crf_loss, "internal_losses": internal_losses, "accuracy": self.accuracy_fn.result()}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses, sequence_length, viterbi_tags = self.compute_loss(x, y, sample_weight)
        mask = tf.sequence_mask(sequence_length, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)
        return {"crf_loss_val": crf_loss, "internal_losses_val": internal_losses, "accuracy": self.accuracy_fn.result()}

    def predict_step(self, data):
        # 预测阶段，模型只返回viterbi tags即可
        x, y, sample_weight = unpack_data(data)
        viterbi_tags, *_ = self(x, training=False)
        return viterbi_tags
