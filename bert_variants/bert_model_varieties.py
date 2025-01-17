import torch
from torch import nn
from transformers import BertModel, DistilBertModel, XLMRobertaModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class BertConcatMultiPassageClassifier(nn.Module):

    def __init__(self, passage_count, class_count, dropout):
        super(BertConcatMultiPassageClassifier, self).__init__()
        self.passage_count = passage_count
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.concat_dropout = nn.Dropout(dropout)
        self.concat_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.concat_relu = nn.ReLU()
        self.mixer_dropout = nn.Dropout(dropout)
        self.mixer_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.mixer_relu = nn.ReLU()
        self.classif_dropout = nn.Dropout(dropout)
        self.classif_linear = nn.Linear(768 * passage_count, class_count)
        self.classif_relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, masks):
        concat_outputs_bert = self.bert(input_ids=input_ids[:,0], attention_mask=masks[:,0]).pooler_output
        for i in range(1, self.passage_count):
            pooler_output = self.bert(input_ids=input_ids[:,i], attention_mask=masks[:,i]).pooler_output
            concat_outputs_bert = torch.cat((concat_outputs_bert, pooler_output), dim=1)
        concat_dropout_output = self.concat_dropout(concat_outputs_bert)
        concat_linear_output = self.concat_linear(concat_dropout_output)
        concat_relu_output = self.concat_relu(concat_linear_output)
        mixer_dropout_output = self.mixer_dropout(concat_relu_output)
        mixer_linear_output = self.mixer_linear(mixer_dropout_output)
        mixer_relu_output = self.mixer_relu(mixer_linear_output)
        classif_dropout_output = self.classif_dropout(mixer_relu_output)
        classif_linear_output = self.classif_linear(classif_dropout_output)
        final_layer = self.classif_relu(classif_linear_output)
        prob = self.softmax(final_layer)
        return prob

class DistilBertConcatMultiPassageClassifier(nn.Module):

    def __init__(self, passage_count, class_count, dropout):
        super(DistilBertConcatMultiPassageClassifier, self).__init__()
        self.passage_count = passage_count
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.concat_dropout = nn.Dropout(dropout)
        self.concat_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.concat_relu = nn.ReLU()
        self.mixer_dropout = nn.Dropout(dropout)
        self.mixer_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.mixer_relu = nn.ReLU()
        self.classif_dropout = nn.Dropout(dropout)
        self.classif_linear = nn.Linear(768 * passage_count, class_count)
        self.classif_relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, masks):
        ## if shape of input_ids is [2,512] which is 2 dimension, it is converted into [2,1,512] where 1 is the passage_count
        if input_ids.ndimension()==2:
           input_ids = torch.reshape(input_ids, [input_ids.size()[0], self.passage_count, input_ids.size()[1]])
        distilbert_output = self.distilbert(input_ids=input_ids[:,0], attention_mask=masks[:,0])
        hidden_state = distilbert_output[0]
        concat_outputs_bert = hidden_state[:, 0]
        for i in range(1, self.passage_count):
            distilbert_output = self.distilbert(input_ids=input_ids[:,i], attention_mask=masks[:,i])
            hidden_state = distilbert_output[0]
            pooler_output = hidden_state[:, 0]
            concat_outputs_bert = torch.cat((concat_outputs_bert, pooler_output), dim=1)
        concat_dropout_output = self.concat_dropout(concat_outputs_bert)
        concat_linear_output = self.concat_linear(concat_dropout_output)
        concat_relu_output = self.concat_relu(concat_linear_output)
        mixer_dropout_output = self.mixer_dropout(concat_relu_output)
        mixer_linear_output = self.mixer_linear(mixer_dropout_output)
        mixer_relu_output = self.mixer_relu(mixer_linear_output)
        classif_dropout_output = self.classif_dropout(mixer_relu_output)
        classif_linear_output = self.classif_linear(classif_dropout_output)
        final_layer = self.classif_relu(classif_linear_output)
        prob = self.softmax(final_layer)
        return prob


class BertLSTMMultiPassageClassifier(nn.Module):

    def __init__(self, class_count, dropout):
        super(BertLSTMMultiPassageClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(768, 768)
        self.linear = nn.Linear(768, class_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, masks):
        num_input_pieces = input_ids.size(dim=1)
        pooler_output = self.bert(input_ids=input_ids[:,0], attention_mask=masks[:,0]).pooler_output
        dropout_output = self.dropout(pooler_output)
        outputs_to_stack = [dropout_output]
        for i in range(1, num_input_pieces):
            pooler_output = self.bert(input_ids=input_ids[:,i], attention_mask=masks[:,i]).pooler_output
            dropout_output = self.dropout(pooler_output)
            outputs_to_stack.append(dropout_output)
        stacked_bert_outputs = torch.stack(outputs_to_stack)
        lstm_output, _ = self.lstm(stacked_bert_outputs)
        linear_output = self.linear(lstm_output[-1])
        #final_layer = self.relu(linear_output)
        #prob = self.softmax(final_layer)
        #return linear_output
        return self.sigmoid(linear_output)
    
    
class MultiPassageXlmRobertaClassifier(torch.nn.Module):
    def __init__(self, model_name, number_labels):
        super(MultiPassageXlmRobertaClassifier, self).__init__()
        self.roberta_model = XLMRobertaModel.from_pretrained(model_name, return_dict=True)
        self.dropout = torch.nn.Dropout(0.25)
        if model_name == "xlm-roberta-base":
            self.linear = torch.nn.Linear(768, number_labels)
        else:
            self.linear = torch.nn.Linear(1024, number_labels)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.roberta_model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

 
class MultilabelXlmRobertaClassifier(torch.nn.Module):
    def __init__(self, model_name, cache_dir, number_labels, dropout):
        super(MultilabelXlmRobertaClassifier, self).__init__()
        self.roberta_model = XLMRobertaModel.from_pretrained(model_name, cache_dir=cache_dir, return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        # Define a minear layer for multilabe classification
        if model_name == "xlm-roberta-base":
            self.linear = torch.nn.Linear(768, number_labels)
        else:
            self.linear = torch.nn.Linear(1024, number_labels)
        #self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, input_ids, masks):
        output = self.roberta_model(input_ids, attention_mask=masks)
        # extract the output embd from the last layer
        last_hidden_states = output.pooler_output
        logits = self.linear(self.dropout(last_hidden_states))
        # appliy sigmoid
        #proba = self.sigmoid(logits)
        return logits
    
    
class MultilabelLSTMXlmRobertaClassifier(torch.nn.Module):
    def __init__(self, model_name, number_labels):
        super(MultilabelLSTMXlmRobertaClassifier, self).__init__()
        self.roberta_model = XLMRobertaModel.from_pretrained(model_name, return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.lstm = nn.LSTM(768, 768)
        # Define a minear layer for multilabe classification
        if model_name == "xlm-roberta-base":
            self.linear = torch.nn.Linear(768, number_labels)
        else:
            self.linear = torch.nn.Linear(1024, number_labels)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, input_ids, masks):
        num_input_pieces = input_ids.size(dim=1)
        output = self.roberta_model(input_ids=input_ids[:,0], attention_mask=masks[:,0])
        last_hidden_states = output.pooler_output
        dropout_output = self.dropout(last_hidden_states)
        outputs_to_stack = [dropout_output]
        for i in range(1, num_input_pieces):
            output = self.roberta_model(input_ids=input_ids[:,i], attention_mask=masks[:,i])
            last_hidden_states = output.pooler_output
            dropout_output = self.dropout(last_hidden_states)
            outputs_to_stack.append(dropout_output)
        stacked_bert_outputs = torch.stack(outputs_to_stack)
        lstm_output, _ = self.lstm(stacked_bert_outputs)
        logits = self.linear(lstm_output[-1])
        # proba = self.sigmoid(logits) # appliy sigmoid
        return logits

class TransformerEncoder(nn.Module):
    # YR: As described in "Attention is All You Need"
    def __init__(self, v_dim):
        super(TransformerEncoder, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=v_dim, num_heads=3, dropout=0.1)
        self.norm1 = nn.BatchNorm1d(v_dim)
        self.feed_forward = nn.Sequential(
                                nn.Linear(v_dim, 3 * v_dim),
                                nn.ReLU(),
                                nn.Linear(3 * v_dim, v_dim)
                            )
        self.norm2 = nn.BatchNorm1d(v_dim)
    def forward(self, v):
        attn_output, _ = self.self_attention(v, v, v)
        attn_output_plus_residual = v.add(attn_output)
        norm_attn_output_plus_residual = self.norm1(attn_output_plus_residual)
        ff_output = self.feed_forward(norm_attn_output_plus_residual)
        ff_output_plus_residual = ff_output.add(norm_attn_output_plus_residual)
        norm_ff_output_plus_residual = self.norm2(ff_output_plus_residual)
        return norm_ff_output_plus_residual

class MultilabelSelfAttentionXlmRobertaClassifier(torch.nn.Module):
    def __init__(self, doc_count, class_count):
        super(MultilabelSelfAttentionXlmRobertaClassifier, self).__init__()
        self.doc_count = doc_count
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        # YR: For this experiment I'm using a stack of 4 encoders, more or fewer may be adequate for specific problems
        self.encoder_stack = nn.Sequential(
                                TransformerEncoder(768 * doc_count),
                                TransformerEncoder(768 * doc_count),
                                TransformerEncoder(768 * doc_count),
                                #TransformerEncoder(768 * doc_count)
                            )
        self.final_linear = nn.Linear(768 * doc_count, class_count)

    def forward(self, input_ids, masks):
        tensors_to_concat = [self.xlm_roberta(input_ids=input_ids[:,i], attention_mask=masks[:,i]).pooler_output for i in range(self.doc_count)]
        concat_outputs_xlm_roberta = torch.cat(tensors_to_concat, dim=1)
        enc_concat_otputs_xlm_roberta = self.encoder_stack(concat_outputs_xlm_roberta)
        model_output = self.final_linear(enc_concat_otputs_xlm_roberta)

        return model_output
