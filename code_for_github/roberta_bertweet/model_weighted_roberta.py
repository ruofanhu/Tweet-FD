import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaForTokenClassification, RobertaForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead, RobertaModel
except ModuleNotFoundError:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead, \
        RobertaModel


class BaseBiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for key in config:
            if key not in ['shape', '_ipython_display_', '_repr_mimebundle_']:
                setattr(self, key, config[key])


class BiLSTMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.rnn_hidden_dimension * 2, config.rnn_hidden_dimension * 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.rnn_hidden_dimension * 2, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BiLSTMForWeightedTokenClassification(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                class_weight=None,
                **kwargs
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        sequence_output = self.dropout(output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (logits,)
        return ((loss,) + outputs) if loss is not None else outputs


class BiLSTMForTokenClassificationWithCRF(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                class_weight=None,
                **kwargs
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        sequence_output = self.dropout(output)
        logits = self.classifier(sequence_output)

        new_logits_list, new_labels_list = [], []
        for seq_logits, seq_labels in zip(logits, labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_logits_list.append(seq_logits[seq_labels >= 0])
            new_labels_list.append(seq_labels[seq_labels >= 0])

        new_logits = pad_sequence(new_logits_list).transpose(0, 1)
        new_labels = pad_sequence(new_labels_list, padding_value=-999).transpose(0, 1)
        prediction_mask = new_labels >= 0
        active_labels = torch.where(
            prediction_mask, new_labels, torch.tensor(0).type_as(new_labels)
        )

        loss = -torch.mean(self.crf(new_logits, active_labels, prediction_mask, reduction='token_mean'))
        output_tags = self.crf.decode(new_logits, prediction_mask)

        return loss, logits, output_tags


class BiLSTMForWeightedSequenceClassification(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = BiLSTMClassificationHead(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                class_weight=None,
                **kwargs
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        out_forward = output[range(len(output)), text_len - 1, :self.rnn_hidden_dimension]
        out_reverse = output[:, 0, self.rnn_hidden_dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        logits = self.classifier(out_reduced)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=class_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (logits,)
        return ((loss,) + outputs) if loss is not None else outputs


class BiLSTMForWeightedTokenAndSequenceClassification(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.seq_classifier = BiLSTMClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_token_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                seq_labels=None,
                token_labels=None,
                token_class_weight=None,
                seq_class_weight=None,
                token_lambda=1,
                **kwargs
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        out_forward = output[range(len(output)), text_len - 1, :self.rnn_hidden_dimension]
        out_reverse = output[:, 0, self.rnn_hidden_dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        token_sequence_output = self.dropout(output)
        token_logits = self.token_classifier(token_sequence_output)
        seq_logits = self.seq_classifier(out_reduced)

        loss = None
        if token_labels is not None:
            token_loss_fct = CrossEntropyLoss(weight=token_class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)
                active_labels = torch.where(
                    active_loss, token_labels.view(-1), torch.tensor(token_loss_fct.ignore_index).type_as(token_labels)
                )
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(token_logits.view(-1, self.num_token_labels), token_labels.view(-1))
            loss = token_loss
        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        outputs = (token_logits, seq_logits)
        return ((loss,) + outputs) if loss is not None else outputs


class BiLSTMForWeightedTokenAndSequenceClassificationWithCRF(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.seq_classifier = BiLSTMClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_token_labels)
        self.crf = CRF(num_tags=config.num_token_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                seq_labels=None,
                token_labels=None,
                token_class_weight=None,
                seq_class_weight=None,
                token_lambda=1,
                **kwargs
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        out_forward = output[range(len(output)), text_len - 1, :self.rnn_hidden_dimension]
        out_reverse = output[:, 0, self.rnn_hidden_dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        token_sequence_output = self.dropout(output)
        token_logits = self.token_classifier(token_sequence_output)
        seq_logits = self.seq_classifier(out_reduced)

        loss = None
        if token_labels is not None:
            new_token_logits_list, new_token_labels_list = [], []
            for t_logits, t_labels in zip(token_logits, token_labels):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                new_token_logits_list.append(t_logits[t_labels >= 0])
                new_token_labels_list.append(t_labels[t_labels >= 0])

            new_token_logits = pad_sequence(new_token_logits_list).transpose(0, 1)
            new_token_labels = pad_sequence(new_token_labels_list, padding_value=-999).transpose(0, 1)
            token_prediction_mask = new_token_labels >= 0
            active_token_labels = torch.where(
                token_prediction_mask, new_token_labels, torch.tensor(0).type_as(new_token_labels)
            )

            loss = -torch.mean(
                self.crf(new_token_logits, active_token_labels, token_prediction_mask, reduction='token_mean'))
            output_tags = self.crf.decode(new_token_logits, token_prediction_mask)

        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        if token_labels is not None:
            outputs = (token_logits, seq_logits, output_tags)
        else:
            outputs = (token_logits, seq_logits)
        return ((loss,) + outputs) if loss is not None else outputs


class BiLSTMForWeightedTwoTokenClassification(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout_0 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_labels)
        self.se_classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_se_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                se_labels=None,
                class_weight=None,
                se_class_weight=None,
                se_lambda=1,
                **kwargs,
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        sequence_output_fr = self.dropout_0(output)
        sequence_output_se = self.dropout_1(output)
        logits = self.classifier(sequence_output_fr)
        se_logits = self.se_classifier(sequence_output_se)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if se_labels is not None:
            se_loss_fct = CrossEntropyLoss(weight=se_class_weight)
            if attention_mask is not None:
                active_se_loss = attention_mask.view(-1) == 1
                active_se_logits = se_logits.view(-1, self.num_se_labels)
                active_se_labels = torch.where(
                    active_se_loss, se_labels.view(-1), torch.tensor(se_loss_fct.ignore_index).type_as(se_labels)
                )
                loss = se_lambda * se_loss_fct(active_se_logits,
                                               active_se_labels) + loss if loss is not None else se_loss_fct(
                    active_se_logits, active_se_labels)

        output = (logits, se_logits)
        return ((loss,) + output) if loss is not None else output


class BiLSTMForTwoTokenClassificationWithCRF(BaseBiLSTM):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.rnn_hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout_0 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_labels)
        self.se_classifier = nn.Linear(self.rnn_hidden_dimension * 2, self.num_se_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.se_crf = CRF(num_tags=config.num_se_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                se_labels=None,
                class_weight=None,
                se_class_weight=None,
                se_lambda=1,
                **kwargs,
                ):
        text_len = torch.sum(attention_mask, dim=1).detach().cpu().long()
        packed_input = pack_padded_sequence(input_ids, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])

        sequence_output_fr = self.dropout_0(output)
        sequence_output_se = self.dropout_1(output)
        logits = self.classifier(sequence_output_fr)
        se_logits = self.se_classifier(sequence_output_se)

        new_logits_list, new_labels_list = [], []
        for seq_logits, seq_labels in zip(logits, labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_logits_list.append(seq_logits[seq_labels >= 0])
            new_labels_list.append(seq_labels[seq_labels >= 0])

        new_logits = pad_sequence(new_logits_list).transpose(0, 1)
        new_labels = pad_sequence(new_labels_list, padding_value=-999).transpose(0, 1)
        prediction_mask = new_labels >= 0
        active_labels = torch.where(
            prediction_mask, new_labels, torch.tensor(0).type_as(new_labels)
        )

        loss = -torch.mean(self.crf(new_logits, active_labels, prediction_mask, reduction='token_mean'))
        output_tags = self.crf.decode(new_logits, prediction_mask)

        new_se_logits_list, new_se_labels_list = [], []
        for seq_se_logits, seq_se_labels in zip(se_logits, se_labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_se_logits_list.append(seq_se_logits[seq_se_labels >= 0])
            new_se_labels_list.append(seq_se_labels[seq_se_labels >= 0])

        new_se_logits = pad_sequence(new_se_logits_list).transpose(0, 1)
        new_se_labels = pad_sequence(new_se_labels_list, padding_value=-999).transpose(0, 1)
        se_prediction_mask = new_se_labels >= 0
        active_se_labels = torch.where(
            se_prediction_mask, new_se_labels, torch.tensor(0).type_as(new_se_labels)
        )

        se_loss = -torch.mean(self.se_crf(new_se_logits, active_se_labels, se_prediction_mask, reduction='token_mean'))
        output_se_tags = self.se_crf.decode(new_se_logits, se_prediction_mask)
        loss += se_lambda * se_loss

        return loss, logits, se_logits, output_tags, output_se_tags


class RobertaForWeightedTokenClassification(RobertaForTokenClassification):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            class_weight=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForWeightedTwoTokenClassification(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_se_labels = config.num_se_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout_0 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.se_classifier = nn.Linear(config.hidden_size, config.num_se_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            se_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            class_weight=None,
            se_class_weight=None,
            se_lambda=1,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output_fr = self.dropout_0(sequence_output)
        sequence_output_se = self.dropout_1(sequence_output)
        logits = self.classifier(sequence_output_fr)
        se_logits = self.se_classifier(sequence_output_se)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if se_labels is not None:
            se_loss_fct = CrossEntropyLoss(weight=se_class_weight)
            if attention_mask is not None:
                active_se_loss = attention_mask.view(-1) == 1
                active_se_logits = se_logits.view(-1, self.num_se_labels)
                active_se_labels = torch.where(
                    active_se_loss, se_labels.view(-1), torch.tensor(se_loss_fct.ignore_index).type_as(se_labels)
                )
                loss = se_lambda * se_loss_fct(active_se_logits,
                                               active_se_labels) + loss if loss is not None else se_loss_fct(
                    active_se_logits, active_se_labels)

        output = (logits, se_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class RobertaForTokenClassificationWithCRF(RobertaForTokenClassification):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        new_logits_list, new_labels_list = [], []
        for seq_logits, seq_labels in zip(logits, labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_logits_list.append(seq_logits[seq_labels >= 0])
            new_labels_list.append(seq_labels[seq_labels >= 0])

        new_logits = pad_sequence(new_logits_list).transpose(0, 1)
        new_labels = pad_sequence(new_labels_list, padding_value=-999).transpose(0, 1)
        prediction_mask = new_labels >= 0
        active_labels = torch.where(
            prediction_mask, new_labels, torch.tensor(0).type_as(new_labels)
        )

        loss = -torch.mean(self.crf(new_logits, active_labels, prediction_mask, reduction='token_mean'))
        output_tags = self.crf.decode(new_logits, prediction_mask)

        return loss, logits, output_tags


class RobertaForTwoTokenClassificationWithCRF(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_se_labels = config.num_se_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout_0 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.se_classifier = nn.Linear(config.hidden_size, config.num_se_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.se_crf = CRF(num_tags=config.num_se_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            se_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            class_weight=None,
            se_class_weight=None,
            se_lambda=1,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output_fr = self.dropout_0(sequence_output)
        sequence_output_se = self.dropout_1(sequence_output)
        logits = self.classifier(sequence_output_fr)
        se_logits = self.se_classifier(sequence_output_se)

        new_logits_list, new_labels_list = [], []
        for seq_logits, seq_labels in zip(logits, labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_logits_list.append(seq_logits[seq_labels >= 0])
            new_labels_list.append(seq_labels[seq_labels >= 0])

        new_logits = pad_sequence(new_logits_list).transpose(0, 1)
        new_labels = pad_sequence(new_labels_list, padding_value=-999).transpose(0, 1)
        prediction_mask = new_labels >= 0
        active_labels = torch.where(
            prediction_mask, new_labels, torch.tensor(0).type_as(new_labels)
        )

        loss = -torch.mean(self.crf(new_logits, active_labels, prediction_mask, reduction='token_mean'))
        output_tags = self.crf.decode(new_logits, prediction_mask)

        new_se_logits_list, new_se_labels_list = [], []
        for seq_se_logits, seq_se_labels in zip(se_logits, se_labels):
            # Index logits and labels using prediction mask to pass only the
            # first subtoken of each word to CRF.
            new_se_logits_list.append(seq_se_logits[seq_se_labels >= 0])
            new_se_labels_list.append(seq_se_labels[seq_se_labels >= 0])

        new_se_logits = pad_sequence(new_se_logits_list).transpose(0, 1)
        new_se_labels = pad_sequence(new_se_labels_list, padding_value=-999).transpose(0, 1)
        se_prediction_mask = new_se_labels >= 0
        active_se_labels = torch.where(
            se_prediction_mask, new_se_labels, torch.tensor(0).type_as(new_se_labels)
        )

        se_loss = -torch.mean(self.se_crf(new_se_logits, active_se_labels, se_prediction_mask, reduction='token_mean'))
        output_se_tags = self.se_crf.decode(new_se_logits, se_prediction_mask)
        loss += se_lambda * se_loss

        return loss, logits, se_logits, output_tags, output_se_tags


class RobertaForWeightedSequenceClassification(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            class_weight=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=class_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForTokenAndSequenceClassification(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.seq_classifier = RobertaClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:
            token_loss_fct = CrossEntropyLoss(weight=token_class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)
                active_labels = torch.where(
                    active_loss, token_labels.view(-1), torch.tensor(token_loss_fct.ignore_index).type_as(token_labels)
                )
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(token_logits.view(-1, self.num_token_labels), token_labels.view(-1))
            loss = token_loss
        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class RobertaForTokenAndSequenceClassificationWithCRF(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.seq_classifier = RobertaClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.crf = CRF(num_tags=self.num_token_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:

            new_token_logits_list, new_token_labels_list = [], []
            for t_logits, t_labels in zip(token_logits, token_labels):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                new_token_logits_list.append(t_logits[t_labels >= 0])
                new_token_labels_list.append(t_labels[t_labels >= 0])

            new_token_logits = pad_sequence(new_token_logits_list).transpose(0, 1)
            new_token_labels = pad_sequence(new_token_labels_list, padding_value=-999).transpose(0, 1)
            token_prediction_mask = new_token_labels >= 0
            active_token_labels = torch.where(
                token_prediction_mask, new_token_labels, torch.tensor(0).type_as(new_token_labels)
            )

            loss = -torch.mean(
                self.crf(new_token_logits, active_token_labels, token_prediction_mask, reduction='token_mean'))
            output_tags = self.crf.decode(new_token_logits, token_prediction_mask)

        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        if token_labels is not None:
            output = (token_logits, seq_logits, output_tags) + outputs[2:]
        else:
            output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
