import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_candles

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, os, math
from PIL import Image


class CandleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, embed_dim, num_heads, num_layers, dropout=0.1, ff_mult=4):
        super(CandleTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim, bias=False)
        self.bnorm = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Embedding(seq_len, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, embed_dim*ff_mult, dropout, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, output_dim)
        self.embed_dim = embed_dim
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.05)
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, src):
        positions = torch.arange(0, src.size(1)).unsqueeze(0).to(src.device)
        x = self.fc(src).transpose(1, 2)
        x = self.bnorm(x).transpose(1, 2)
        x = nn.GELU()(x)
        x = self.dropout(x) 
        x = x + self.pos_encoder(positions) * self.embed_dim ** 0.5
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))    
        tgt_mask = tgt_mask.to(src.device)
        output = self.decoder(x, x, tgt_mask=tgt_mask)
        output = self.output_layer(output)
    

        return output

class SeqWeightedMSELoss(nn.Module):
    def __init__(self, weight = None):
        super(SeqWeightedMSELoss, self).__init__()
        self.weight = weight
        self.base_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        if self.weight is not None and self.weight != 1.0:    
            main_pred = y_pred[:, :-1]
            main_true = y_true[:, :-1]
            last_pred = y_pred[:, -1]
            last_true = y_true[:, -1]
            main_loss = self.base_loss(main_pred, main_true)
            last_loss = self.base_loss(last_pred, last_true) * self.weight
            return main_loss + last_loss
        else:
            return self.base_loss(y_pred, y_true)



#define writer for tensorboard
def init_writer(log_dir, continue_training = False, purge_step = None, writers = None):
 
    if not continue_training:
        #use python os package to delete logs including files in subfolders and subfolders itself
        for root, dirs, files in os.walk(log_dir):
            print(f"Deleting writer logs in {root}")
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                for fils in os.listdir(os.path.join(root, dir)):
                    os.remove(os.path.join(root, dir, fils))
                os.rmdir(os.path.join(root, dir))  
    if purge_step is not None and writers is not None:
        print(f"Initializing writer...purge step: {purge_step}")
        for wr in reversed(writers):
            writer = SummaryWriter(wr+'/', purge_step = purge_step)    
    else:
        print(f"Initializing writer...")
        writer = SummaryWriter(log_dir)
    return writer


def init_model(hyperparams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    #if CUDA enable TF32
    if device == torch.device('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.mem_efficient_sdp_enabled = True
        torch.backends.cuda.flash_sdp_enabled = True


    #device = torch.device('cpu')

    print(f"Device to be used: {device}")
    #Unpack hyperparameters and initialize model
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    seq_len = hyperparams['seq_len']
    embed_dim = hyperparams['embed_dim']
    num_heads = hyperparams['num_heads']
    num_layers = hyperparams['num_layers']
    dropout = hyperparams['dropout']
    ff_mult = hyperparams['ff_mult']

    torch.manual_seed(42)
    model = CandleTransformer(input_dim, output_dim, seq_len, embed_dim, num_heads, num_layers, dropout=dropout, ff_mult=ff_mult)

    model = model.to(device)
   
    return model, device

def store_traning_result(model, optimizer, scaler, epoch, model_path, hyperparameters):
    #store model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'hyperparameters': hyperparameters,
        }, model_path)

def load_model(model, optimizer, scaler, model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = model.to('cpu')                          
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    hyperparameters = checkpoint['hyperparameters']
    epoch = checkpoint['epoch']
    return model, optimizer, scaler, epoch, hyperparameters

def get_leaning_rates(embed_dim, warmup_steps, num_epochs):
    l_rates = []
    for epoch in range(num_epochs):
        lr = embed_dim ** (-1.5) * min((epoch+1) ** (-0.5), (epoch+1)*warmup_steps**(-1.5)) 
        l_rates.append(lr)
        
    return l_rates, max(l_rates)

def set_learning_rate(optimizer, lr, weight_decay = 0):
    
    for g in optimizer.param_groups:
        g['lr'] = lr 
        g['weight_decay'] = weight_decay
    return optimizer

def mape_loss(y_pred, y_true):
    
    mask = y_true != 0
    mape_non_zeros = torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    mape_zeros = torch.abs(y_pred[~mask])
    
    # consider each part ratio in the total sum
    total_elements = y_true.numel()
    mape =  ((mape_non_zeros.sum() + mape_zeros.sum()) / total_elements) * 100
    return mape



def write_charts_to_TB(name, writer, sequence, epoch, init_price = 1000, init_timestamp = None):
    
    
  
    candles = restore_candles(sequence, init_price, init_timestamp)
    

    fig2 = go.Figure(data=[go.Candlestick(x=candles.index,
                    open=candles['open'],
                    high=candles['high'],
                    low=candles['low'],
                    close=candles['close'])])
    
    fig2.update_layout(height=600, width=1200)
    fig2.update_layout(xaxis_rangeslider_visible=False)
    
    
    #convert figures to image and write to tensorboard
    fig2_bytes = fig2.to_image(format="png")

    # Преобразуем байтовые данные в массив NumPy

    fig2_image = np.array(Image.open(io.BytesIO(fig2_bytes)))

    # Преобразуем массивы NumPy в тензоры PyTorch
  

    # Добавляем изображения в TensorBoard

    writer.add_image(f'{name}', fig2_image, epoch, dataformats='HWC')
    writer.flush()