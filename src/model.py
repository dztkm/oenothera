import torch
import torch.nn.functional as F
import torch.nn as nn

from hparams import HParams

def mask_from_seq_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.BoolTensor:
    sequence_lengths = sequence_lengths.long()
    seq_range = torch.arange(max_length, device=sequence_lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(sequence_lengths.size(0), max_length)
    seq_length_expand = sequence_lengths.unsqueeze(-1)
    return seq_range_expand < seq_length_expand

class EncoderBlock(nn.Module):
    def __init__(self, hp: HParams):
        super(EncoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hp.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.dropout_1 = torch.nn.Dropout(0.1)

        self.norm_2 = nn.LayerNorm(normalized_shape=hp.embedding_size)

        self.linear_1 = nn.Linear(hp.embedding_size, hp.dim_feedforward)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.linear_2 = nn.Linear(hp.dim_feedforward, hp.embedding_size)
        self.dropout_3 = torch.nn.Dropout(0.1)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_out = self.norm_1(x)
        x_out, _ = self.attn(
            query=x_out, 
            key=x_out, 
            value=x_out,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x_out = self.dropout_1(x_out)
        x = x + x_out    

        x_out = self.norm_2(x) 
        x_out = self.linear_1(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout_2(x_out)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_3(x_out)
        x = x + x_out
        return x

class DecoderBlock(nn.Module):
    def __init__(self, hp: HParams):
        super(DecoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=hp.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.dropout_1 = torch.nn.Dropout(0.1)

        self.norm_2 = nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hp.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )    
        self.dropout_2 = torch.nn.Dropout(0.1)

        self.norm_3 = nn.LayerNorm(normalized_shape=hp.embedding_size)

        self.linear_1 = nn.Linear(hp.embedding_size, hp.dim_feedforward)
        self.dropout_3 = torch.nn.Dropout(0.1)
        self.linear_2 = nn.Linear(hp.dim_feedforward, hp.embedding_size)
        self.dropout_4 = torch.nn.Dropout(0.1)

    def forward(self, x, memory, x_attn_mask=None, x_key_padding_mask=None, memory_attn_mask=None, memory_key_padding_mask=None):
        x_out, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x,
            attn_mask=x_attn_mask,
            key_padding_mask=x_key_padding_mask
        )
        x_out = self.dropout_1(x_out)
        x = self.norm_1(x + x_out)
         
        x_out, _ = self.attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_attn_mask,
            key_padding_mask=memory_key_padding_mask
        )
        x_out = self.dropout_2(x_out)
        x = self.norm_2(x + x_out)

        x_out = self.linear_1(x)
        x_out = F.relu(x_out)
        x_out = self.dropout_3(x_out)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_4(x_out)
        x = self.norm_3(x + x_out)
        return x

class EncoderPreNet(nn.Module):
    def __init__(self, hp: HParams):
        super(EncoderPreNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=hp.text_num_embeddings,
            embedding_dim=hp.encoder_embedding_size
        )
        self.linear_1 = nn.Linear(hp.encoder_embedding_size, hp.encoder_embedding_size)
        self.linear_2 = nn.Linear(hp.encoder_embedding_size, hp.embedding_size)

        self.conv_1 = nn.Conv1d(hp.encoder_embedding_size, hp.encoder_embedding_size, kernel_size=hp.encoder_kernel_size, stride=1, padding=int((hp.encoder_kernel_size - 1) / 2), dilation=1)
        self.bn_1 = nn.BatchNorm1d(hp.encoder_embedding_size)
        self.dropout_1 = torch.nn.Dropout(0.5)

        self.conv_2 = nn.Conv1d(hp.encoder_embedding_size, hp.encoder_embedding_size, kernel_size=hp.encoder_kernel_size, stride=1, padding=int((hp.encoder_kernel_size - 1) / 2), dilation=1)
        self.bn_2 = nn.BatchNorm1d(hp.encoder_embedding_size)
        self.dropout_2 = torch.nn.Dropout(0.5)

        self.conv_3 = nn.Conv1d(hp.encoder_embedding_size, hp.encoder_embedding_size, kernel_size=hp.encoder_kernel_size, stride=1, padding=int((hp.encoder_kernel_size - 1) / 2), dilation=1)
        self.bn_3 = nn.BatchNorm1d(hp.encoder_embedding_size)
        self.dropout_3 = torch.nn.Dropout(0.5)    

    def forward(self, text):
        x = self.embedding(text) # (N, S, E)
        x = self.linear_1(x)
        x = x.transpose(2, 1) # (N, E, S) 

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        
        x = self.conv_3(x)
        x = self.bn_3(x)    
        x = F.relu(x)
        x = self.dropout_3(x)

        x = x.transpose(1, 2) # (N, S, E)
        x = self.linear_2(x)
        return x

class PostNet(nn.Module):
    def __init__(self, hp: HParams):
        super(PostNet, self).__init__()  
        self.conv_1 = nn.Conv1d(hp.n_mels, hp.postnet_embedding_size, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_1 = nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout_1 = torch.nn.Dropout(0.5)

        self.conv_2 = nn.Conv1d(hp.postnet_embedding_size, hp.postnet_embedding_size, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_2 = nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout_2 = torch.nn.Dropout(0.5)

        self.conv_3 = nn.Conv1d(hp.postnet_embedding_size, hp.postnet_embedding_size, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_3 = nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout_3 = torch.nn.Dropout(0.5)

        self.conv_4 = nn.Conv1d(hp.postnet_embedding_size, hp.postnet_embedding_size, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_4 = nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout_4 = torch.nn.Dropout(0.5)

        self.conv_5 = nn.Conv1d(hp.postnet_embedding_size, hp.postnet_embedding_size, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_5 = nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout_5 = torch.nn.Dropout(0.5)

        self.conv_6 = nn.Conv1d(hp.postnet_embedding_size, hp.n_mels, kernel_size=hp.postnet_kernel_size, stride=1, padding=int((hp.postnet_kernel_size - 1) / 2), dilation=1)
        self.bn_6 = nn.BatchNorm1d(hp.n_mels)
        self.dropout_6 = torch.nn.Dropout(0.5)

    def forward(self, x):
        # x - (N, TIME, FREQ)
        x = x.transpose(2, 1) # (N, FREQ, TIME)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.tanh(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = torch.tanh(x)
        x = self.dropout_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = torch.tanh(x)
        x = self.dropout_3(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = torch.tanh(x)
        x = self.dropout_4(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = torch.tanh(x)
        x = self.dropout_5(x)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.dropout_6(x)

        x = x.transpose(1, 2)
        return x


class DecoderPreNet(nn.Module):
    def __init__(self, hp: HParams):
        super(DecoderPreNet, self).__init__()
        self.linear_1 = nn.Linear(hp.n_mels, hp.embedding_size)
        self.linear_2 = nn.Linear(hp.embedding_size, hp.embedding_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear_2(x)
        x = F.relu(x)    
        x = F.dropout(x, p=0.5, training=self.training)
        return x    


class TransformerTTS(nn.Module):
    def __init__(self, hp: HParams):
        super(TransformerTTS, self).__init__()
        self.hp = hp

        self.encoder_prenet = EncoderPreNet(hp)
        self.decoder_prenet = DecoderPreNet(hp)
        self.postnet = PostNet(hp)

        self.pos_encoding = nn.Embedding(
            num_embeddings=hp.max_mel_time, 
            embedding_dim=hp.embedding_size
        )

        self.encoder_block_1 = EncoderBlock(hp)
        self.encoder_block_2 = EncoderBlock(hp)
        self.encoder_block_3 = EncoderBlock(hp)

        self.decoder_block_1 = DecoderBlock(hp)
        self.decoder_block_2 = DecoderBlock(hp)
        self.decoder_block_3 = DecoderBlock(hp)

        self.linear_1 = nn.Linear(hp.embedding_size, hp.n_mels) 
        self.linear_2 = nn.Linear(hp.embedding_size, 1)

        self.norm_memory = nn.LayerNorm(normalized_shape=hp.embedding_size)

    def forward(self, text, mel, stop_tokens):  
        '''
        text: (N, S) - zero padded text sequences
        mel: (N, n_mels, TIME) - zero padded mel spectrograms 
        stop_tokens: (N, TIME) - generated by dataset (0.0 for actual frame, 1.0 for padding)
        '''
        text_len = (text != 0).sum(dim=1)
        # Add 1 to mel_len so that the first padding frame is not masked,
        # allowing the model to predict stop_token=1 at that frame.
        mel_len = (stop_tokens == 0).sum(dim=1) + 1

        mel = mel.transpose(1, 2) # (N, TIME, n_mels)

        N = text.shape[0]
        S = text.shape[1]
        TIME = mel.shape[1]

        self.src_key_padding_mask = torch.zeros(
            (N, S), device=text.device
        ).masked_fill(
          ~mask_from_seq_lengths(text_len, max_length=S), float("-inf")
        )
        
        self.src_mask = torch.zeros(
          (S, S), device=text.device
        ).masked_fill(
          torch.triu(torch.full((S, S), True, dtype=torch.bool), diagonal=1).to(text.device),       
          float("-inf")
        )

        self.tgt_key_padding_mask = torch.zeros(
          (N, TIME), device=mel.device
        ).masked_fill(
          ~mask_from_seq_lengths(mel_len, max_length=TIME), float("-inf")
        )

        self.tgt_mask = torch.zeros(
          (TIME, TIME), device=mel.device
        ).masked_fill(
          torch.triu(torch.full((TIME, TIME), True, device=mel.device, dtype=torch.bool), diagonal=1),       
          float("-inf")
        )

        self.memory_mask = torch.zeros(
          (TIME, S), device=mel.device
        ).masked_fill(
          torch.triu(torch.full((TIME, S), True, device=mel.device, dtype=torch.bool), diagonal=1),       
          float("-inf")
        )    

        text_x = self.encoder_prenet(text) # (N, S, E)    
        
        pos_codes_time = self.pos_encoding(torch.arange(min(TIME, self.hp.max_mel_time)).to(mel.device))
        pos_codes_text = self.pos_encoding(torch.arange(min(S, self.hp.max_mel_time)).to(text.device))

        text_x = text_x + pos_codes_text[:S]

        text_x = self.encoder_block_1(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)
        text_x = self.encoder_block_2(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)    
        text_x = self.encoder_block_3(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)

        text_x = self.norm_memory(text_x)
            
        mel_x = self.decoder_prenet(mel) # (N, TIME, E)    
        mel_x = mel_x + pos_codes_time[:TIME]

        mel_x = self.decoder_block_1(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, x_key_padding_mask=self.tgt_key_padding_mask, memory_attn_mask=self.memory_mask, memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.decoder_block_2(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, x_key_padding_mask=self.tgt_key_padding_mask, memory_attn_mask=self.memory_mask, memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.decoder_block_3(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, x_key_padding_mask=self.tgt_key_padding_mask, memory_attn_mask=self.memory_mask, memory_key_padding_mask=self.src_key_padding_mask)

        mel_linear = self.linear_1(mel_x) # (N, TIME, FREQ)
        mel_postnet = self.postnet(mel_linear) # (N, TIME, FREQ)
        mel_postnet = mel_linear + mel_postnet # (N, TIME, FREQ)
        stop_token_out = self.linear_2(mel_x) # (N, TIME, 1)

        bool_mel_mask = self.tgt_key_padding_mask.ne(0).unsqueeze(-1).repeat(1, 1, self.hp.n_mels)

        mel_linear = mel_linear.masked_fill(bool_mel_mask, 0)
        mel_postnet = mel_postnet.masked_fill(bool_mel_mask, 0)
        # Give normal output for stop token so it can be learned
        stop_token_out = stop_token_out.squeeze(2)
        
        return mel_postnet, mel_linear, stop_token_out

    @torch.no_grad()
    def inference(self, text, max_length=800, stop_token_threshold=0.5, with_tqdm=True):
        self.eval()    
        self.train(False)
        N = text.shape[0]
        SOS = torch.zeros((N, self.hp.n_mels, 1), device=text.device)
        
        mel_padded = SOS
        stop_tokens = torch.zeros((N, 1), device=text.device)
        stop_token_outputs = torch.FloatTensor([]).to(text.device)

        if with_tqdm:
            from tqdm import tqdm
            iters = tqdm(range(max_length))
        else:
            iters = range(max_length)

        for _ in iters:
            mel_postnet, mel_linear, stop_token_out = self.forward(text, mel_padded, stop_tokens)

            last_frame = mel_postnet[:, -1:, :].transpose(1, 2)
            mel_padded = torch.cat([mel_padded, last_frame], dim=2)
            
            if torch.sigmoid(stop_token_out[:, -1]) > stop_token_threshold:      
                break
            else:
                stop_token_outputs = torch.cat([stop_token_outputs, stop_token_out[:,-1:]], dim=1)
                stop_tokens = torch.zeros((N, mel_padded.shape[2]), device=text.device)

        return mel_postnet, stop_token_outputs

