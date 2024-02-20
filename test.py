import math
import copy
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor, optim
from torch.nn.utils.rnn import pad_sequence

# 从JSON文件中读取数据
with open("D:/研究生/硕士毕业论文/课程推荐数据整理/users_data_emb.json", 'r') as file1:
    data = json.load(file1)

with open("D:/研究生/硕士毕业论文/课程推荐数据整理/item_emb.json", 'r') as file2:
    item_emb = json.load(file2)
item_emb = torch.tensor(item_emb).unsqueeze(0)

train_data = data[:627]
test_data = data[627:]

print("Finish data loading!")

maxlen = 160
nhead = 4
nlayers = 2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.5, max_len: int = 160):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TokenTypeEmbedding(nn.Module):

    def __init__(self, d_model: int, n_token_types: int):
        super().__init__()
        self.token_type_embeddings = nn.Embedding(n_token_types, d_model)

    def forward(self, x, token_type_ids):
        return x + self.token_type_embeddings(token_type_ids)
    
def _get_clones(module, N):#创建特定神经网络模块的多个副本
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):#根据提供的激活函数名称返回相应的激活函数
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.5, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None,src_key_padding_mask = None):
        attn_output, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn_output_weights  

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None,src_key_padding_mask= None):
        output = src
        for mod in self.layers:
            output,att = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class BehaviorTransformer(nn.Module):
    def __init__(self, d_model, output_size):
        super(BehaviorTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model=d_model,dropout=0.5,max_len=maxlen)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.5, activation="relu")
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=nlayers, norm=None)
        self.fc = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, user_item_emb, item_emb):
        # 添加位置编码
        user_behavior_emb = self.pos_encoding(user_item_emb)
        
        # 创建掩码
        src_key_padding_mask = torch.zeros(user_item_emb.shape[1], user_item_emb.shape[0], dtype=torch.bool)
        for i, length in enumerate([len(sequence) for sequence in user_item_emb]):
            src_key_padding_mask[length:, i] = True

        # 通过Transformer层
        user_behavior_transformed = self.encoder(user_behavior_emb, src_key_padding_mask=src_key_padding_mask)

        # 用户历史记录表征的平均值作为用户表征
        # (bz, seq_len, emb_sz) -> (bz, 1, emb_sz)
        user_behavior_emb = torch.mean(user_behavior_transformed, axis=1, keepdim=True)
        user_behavior_emb = F.relu(self.fc(user_behavior_emb))
        
        cross = torch.mul(user_behavior_emb, item_emb)  # 计算用户表征和物品表征的逐元素相乘
        output = self.fc_out(cross)  # 使用fc_out将交叉项映射到输出空间
        output = torch.sigmoid(output)  # 使用sigmoid函数将输出映射到0到1范围内
        output = output.squeeze()  # 压缩维度
        
        return output
    
def get_batch(i, batch_size, data, data_length):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, data_length)
    batch = data[start_index:end_index]
    user_item_emb = [torch.tensor(item[1]) for item in batch]
    target = [item[3] for item in batch]
    return user_item_emb,target

def recall_at_k(y_true, y_pred, k):
    y_pred_at_k = y_pred[:k]
    num_relevant_at_k = sum(y_true[item] for item in y_pred_at_k)  
    num_relevant = sum(y_true)
    return num_relevant_at_k / num_relevant

def ndcg_at_k(y_true, y_pred, k):
    y_pred_at_k = y_pred[:k]
    y_true_at_k = [y_true[item] for item in y_pred_at_k]
    gains = np.power(2, y_true_at_k) - 1
    discounts = np.log2(np.arange(len(y_pred_at_k)) + 2)
    dcg = np.sum(gains / discounts)
    ideal_sorted_y_true = sorted(y_true, reverse=True)
    ideal_gains = np.power(2, ideal_sorted_y_true) - 1
    ideal_gains_at_k = ideal_gains[:k]
    ideal_discounts = np.log2(np.arange(len(ideal_gains_at_k)) + 2)
    ideal_dcg = np.sum(ideal_gains_at_k / ideal_discounts)
    return dcg / ideal_dcg

def evaluate(results):
    print("== evaluating ==")

    num_users = len(results)
    total_recall10 = 0
    total_ndcg10 = 0
    total_recall20 = 0
    total_ndcg20 = 0

    for result in results:
        recall10 = recall_at_k(result['truth'], result["predict"], 10)
        ndcg10 = ndcg_at_k(result['truth'], result["predict"], 10)
        total_recall10 += recall10
        total_ndcg10 += ndcg10
        recall20 = recall_at_k(result['truth'], result["predict"], 20)
        ndcg20 = ndcg_at_k(result['truth'], result["predict"], 20)
        total_recall20 += recall20
        total_ndcg20 += ndcg20

    overall_recall10 = total_recall10 / num_users
    overall_ndcg10 = total_ndcg10 / num_users
    overall_recall20 = total_recall20 / num_users
    overall_ndcg20 = total_ndcg20 / num_users

    return overall_recall10, overall_ndcg10, overall_recall20, overall_ndcg20

def set_random_seed(seed = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(12345)

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()
    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()
    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

d_model = 64
output_size = 1
trainepoch = 500
batch_size = 128
train_data_length = 627
test_data_length = 269

model = BehaviorTransformer(d_model, output_size)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
criterion = nn.BCELoss()

for epoch in tqdm(range(trainepoch), desc='Training Epochs'):
    loss_total=0
    for i in tqdm(range(train_data_length//batch_size+1), desc='Training Progress'):
        b_user_item_emb, target = get_batch(i, batch_size, train_data, train_data_length)
        # 使用pad_sequence进行填充
        padded_data = pad_sequence(b_user_item_emb, batch_first=True)
        outputs= model(padded_data,item_emb)
        result=torch.FloatTensor(outputs)
        zero = torch.zeros_like(result)
        one = torch.ones_like(result)
        result=torch.where(result > 0.5, one, zero)
        target = torch.Tensor(target)
        loss = criterion(outputs,target)
        loss_total=loss_total+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print ("loss",loss_total)
    if epoch % 10 != 0:
        continue
    else:#test
        results = []
        for j in tqdm(range(test_data_length//batch_size+1), desc='Testing Progress'):
            (b_user_item_emb, target) = get_batch(j, batch_size, test_data, test_data_length)
            # 使用pad_sequence进行填充
            padded_data = pad_sequence(b_user_item_emb, batch_first=True)
            test_output = model(padded_data,item_emb)
            # 排序
            (output_sort, position) = torch.sort(test_output)
            position.detach().tolist()

            for i in range(len(target)):
                result = {"truth": target[i],"predict": position[i]}
                results.append(result)
                
        recall10, ndcg10, recall20, ndcg20 = evaluate(results)

        print('recall10: ', recall10)
        print('ndcg10: ', ndcg10)
        print('recall20: ', recall20)
        print('ndcg20: ', ndcg20)
        print("== done ==")

print("over")