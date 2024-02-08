from flask import Flask, render_template, request
import torch
import torchtext, math
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import pythainlp
import pickle


app = Flask(__name__)

# Copy some neccessary code from A3.ipynb
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, attention_type): #pf_dim = feed forward dim
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim) # make number to smaller
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, attention_type)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]
        
        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]
        
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, attention_type, max_length = 700):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device,attention_type)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len    = src.shape[1]
        
        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]
        
        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)) # *scale to scale down
        #src: [batch_size, src_len, hid_dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, attention_type):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.attention_type = attention_type
        
        self.fc_q     = nn.Linear(hid_dim, hid_dim)
        self.fc_k     = nn.Linear(hid_dim, hid_dim)
        self.fc_v     = nn.Linear(hid_dim, hid_dim)
        self.v        = nn.Linear(self.head_dim, 1, bias = False)
        self.W        = nn.Linear(self.head_dim, self.head_dim) #for decoder input_
        self.U        = nn.Linear(self.head_dim, self.head_dim)  #for encoder_outputs
        self.W_mal    = nn.Linear(self.head_dim, self.head_dim)
        
        self.fc_o     = nn.Linear(hid_dim, hid_dim) #for output
        
        self.dropout  = nn.Dropout(dropout)
        
        # self.scale    = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
                
    def forward(self, query, key, value, mask = None):
        #src, src, src, src_mask
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        
        batch_size = query.shape[0]
        src_len = key.shape[1]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q=K=V: [batch_size, src len, hid_dim]
        
        #separate in many version
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch_size, n heads, query len, head_dim]
        if self.attention_type == "general":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) #[batch_size, n heads, query len, key len]
        elif self.attention_type == "multiplicative":
            energy = torch.matmul(Q,(self.W_mal(K)).permute(0, 1, 3, 2)) #[batch_size, n heads, query len, key len]

        elif self.attention_type == "additive":

            energy = self.v(torch.tanh(self.U(K) + self.W(Q))) #[batch_size, n heads, query len, 1]
                   
             # Ensure the energy tensor has the same shape as in other attention mechanisms
            energy = energy.expand(-1, -1, -1,src_len)  #[batch_size, n heads, query len, key len]
                
        #for making attention to padding to 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        attention = torch.softmax(energy, dim = -1)
        #attention = [batch_size, n heads, query len, key len]
        
        x = torch.matmul(self.dropout(attention), V)
        #[batch_size, n heads, query len, key len] @ [batch_size, n heads, value len, head_dim]
        #x = [batch_size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()  #we can perform .view
        #x = [batch_size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch_size, query len, hid dim]
        x = self.fc_o(x)
        #x = [batch_size, query len, hid dim]
        
        return x, attention
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, attention_type):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, attention_type)
        self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, attention_type)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg)) #trg_mask to make sure it does not attention future token
        #trg = [batch_size, trg len, hid dim]
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]
        
        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, 
                 pf_dim, dropout, device, attention_type,max_length = 700):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device,attention_type)
                                            for _ in range(n_layers)])
        self.fc_out        = nn.Linear(hid_dim, output_dim)
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, trg len]
        
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            
        #trg: [batch_size, trg len, hid dim]
        #attention: [batch_size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        #output = [batch_size, trg len, output_dim]
        
        return output, attention


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    
def translate_english_to_thai(model, english_sentence, device, SRC_LANGUAGE, TRG_LANGUAGE, SOS_IDX, EOS_IDX, token_transform, vocab_transform):
    # Tokenize the input English sentence
    tokenized_sentence = token_transform[SRC_LANGUAGE](english_sentence)
    
    # Convert tokens to indices using the English vocabulary
    indexed_sentence = [vocab_transform[SRC_LANGUAGE][token] for token in tokenized_sentence]
    num = len(indexed_sentence)
    
    
    # Convert indices to tensor and add batch dimension
    input_tensor = torch.LongTensor(indexed_sentence).unsqueeze(0).to(device)
    
    trg_tokens = []
    trg_token = SOS_IDX
    
    # Translate the English sentence to Thai
    with torch.no_grad():
        for _ in range(num):
            trg_tensor = torch.LongTensor([[trg_token]]).to(device)
            output, _ = model(input_tensor.reshape(1, -1), trg_tensor.reshape(1, -1))
            
            # Get the most likely next token
            trg_token = torch.argmax(output, dim=2)[:,-1].item()
            
            # Append the token to the list
            trg_tokens.append(trg_token)
            
    
    # Convert indices to tokens and then to words
    translated_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[index] for index in trg_tokens]
    
    # Concatenate tokens to form the translated sentence
    translated_sentence = ''.join(translated_tokens)
    
    return translated_sentence


@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    translated_thai = None

    SRC_LANGUAGE = 'en'
    TRG_LANGUAGE = 'th'
    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform[TRG_LANGUAGE] = lambda x: list(pythainlp.tokenize.word_tokenize(x, engine="newmm"))

    with open('vocab_transform.pkl', 'rb') as f:
        vocab_transform = pickle.load(f)

    input_dim   = len(vocab_transform[SRC_LANGUAGE])
    output_dim  = len(vocab_transform[TRG_LANGUAGE])
    hid_dim = 256
    enc_layers = 3
    dec_layers = 3
    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512
    dec_pf_dim = 512
    enc_dropout = 0.1
    device = 'cpu'

    SRC_PAD_IDX = 1 
    TRG_PAD_IDX = 1
    SOS_IDX, EOS_IDX = 2,3

    enc_general = Encoder(input_dim, 
              hid_dim, 
              enc_layers, 
              enc_heads, 
              enc_pf_dim, 
              enc_dropout, 
              device,'general')

    dec_general = Decoder(output_dim, 
                hid_dim, 
                dec_layers, 
                dec_heads, 
                dec_pf_dim, 
                enc_dropout, 
                device, 'general')


    # Load saved model
    loaded_model_general = Seq2SeqTransformer(enc_general, dec_general, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    loaded_model_general.load_state_dict(torch.load('Seq2SeqTransformer_with_general.pt'))

    if request.method == 'POST':
        # Clear the cache
        search_query = None
        translated_thai = None

        search_query = request.form['search_query']
        translated_thai = translate_english_to_thai(loaded_model_general, search_query, device, SRC_LANGUAGE, TRG_LANGUAGE, SOS_IDX, EOS_IDX, token_transform, vocab_transform)
        
    return render_template('index.html', search_query=search_query,translated_thai=translated_thai)

if __name__ == '__main__':
    app.run(debug=True)
