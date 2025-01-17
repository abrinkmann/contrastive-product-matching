import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from transformers import AutoModel, AutoConfig
from src.finetuning.open_book.contrastive_product_matching.src.contrastive.models.loss import SupConLoss, BarlowTwins


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseEncoder(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        
        output = self.transformer(input_ids, attention_mask)

        return output

# self-supervised contrastive model
class ContrastiveSelfSupervisedPretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj='mlp', temperature=0.07):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask, input_ids_right, attention_mask_right):

        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = F.normalize(output, dim=-1)

        loss = self.criterion(output)

        return loss, output


# self-supervised contrastive model
class BarlowTwinsPretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj='mlp',
                 model_dim=768, lambd=0.0051):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.criterion = BarlowTwins(lambd, model_dim)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask, input_ids_right, attention_mask_right):

        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = F.normalize(output, dim=-1)

        loss = self.criterion(output)

        return loss, output


# supervised contrastive model
class ContrastivePretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj='mlp', temperature=0.07):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = F.normalize(output, dim=-1)

        loss = self.criterion(output, labels)

        return loss, output


class ContrastiveModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj='mlp',
                 temperature=0.07):
        """Model used for application"""
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask):

        if self.pool:
            output = self.encoder(input_ids, attention_mask)
            output = mean_pooling(output, attention_mask)

            #output_right = self.encoder(input_ids_right, attention_mask_right)
            #output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output = self.encoder(input_ids, attention_mask)['pooler_output']
            #output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        #output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = F.normalize(output, dim=-1)

        # Do not calculate loss - only for application
        loss = 0

        return (loss, output)

class ContrastivePretrainHead(nn.Module):

    def __init__(self, hidden_size, proj='mlp'):
        super().__init__()
        if proj == 'linear':
            self.proj = nn.Linear(hidden_size, hidden_size)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, hidden_states):
        x = self.proj(hidden_states)
        return x

# cross-entropy fine-tuning model
class ContrastiveClassifierModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, comb_fct='concat-abs-diff-mult', frozen=True, pos_neg=False):
        super().__init__()

        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path
        self.comb_fct = comb_fct
        self.pos_neg = pos_neg

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        if self.pos_neg:
            self.criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_neg]))
        else:
            self.criterion = BCEWithLogitsLoss()
        self.classification_head = ClassificationHead(self.config, self.comb_fct)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        if self.comb_fct == 'concat-abs-diff':
            output = torch.cat((output_left, output_right, torch.abs(output_left - output_right)), -1)
        elif self.comb_fct == 'concat-mult':
            output = torch.cat((output_left, output_right, output_left * output_right), -1)
        elif self.comb_fct == 'concat':
            output = torch.cat((output_left, output_right), -1)
        elif self.comb_fct == 'abs-diff':
            output = torch.abs(output_left - output_right)
        elif self.comb_fct == 'mult':
            output = output_left * output_right
        elif self.comb_fct == 'abs-diff-mult':
            output = torch.cat((torch.abs(output_left - output_right), output_left * output_right), -1)
        elif self.comb_fct == 'concat-abs-diff-mult':
            output = torch.cat((output_left, output_right, torch.abs(output_left - output_right), output_left * output_right), -1)

        proj_output = self.classification_head(output)

        if labels is not None:
            loss = self.criterion(proj_output.view(-1), labels.float())
        else:
            loss = 0

        proj_output = torch.sigmoid(proj_output)

        return (loss, proj_output)

class ClassificationHead(nn.Module):

    def __init__(self, config, comb_fct):
        super().__init__()

        if comb_fct in ['concat-abs-diff', 'concat-mult']:
            self.hidden_size = 3 * config.hidden_size
        elif comb_fct in ['concat', 'abs-diff-mult']:
            self.hidden_size = 2 * config.hidden_size
        elif comb_fct in ['abs-diff', 'mult']:
            self.hidden_size = config.hidden_size
        elif comb_fct in ['concat-abs-diff-mult']:
            self.hidden_size = 4 * config.hidden_size

        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x