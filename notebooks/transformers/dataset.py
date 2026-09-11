import torch

from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, config):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config
        self.src_sos_id = tokenizer_src.token_to_id('<sos>')
        self.src_eos_id = tokenizer_src.token_to_id('<eos>')
        self.src_pad_id = tokenizer_src.token_to_id('<pad>')
        self.tgt_sos_id = tokenizer_tgt.token_to_id('<sos>')
        self.tgt_eos_id = tokenizer_tgt.token_to_id('<eos>')
        self.tgt_pad_id = tokenizer_tgt.token_to_id('<pad>')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.config['lang_src']]
        tgt_text = src_target_pair['translation'][self.config['lang_tgt']]

        seq_len = self.config['seq_len']
        src_tokens = self.tokenizer_src.encode(src_text).ids[:seq_len - 2]
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids[:seq_len - 1]

        enc_padding = seq_len - len(src_tokens) - 2
        dec_padding = seq_len - len(tgt_tokens) - 1

        encoder_input = torch.tensor(
            [self.src_sos_id, *src_tokens, self.src_eos_id]
            + [self.src_pad_id] * enc_padding,
            dtype=torch.long,
        )
        decoder_input = torch.tensor(
            [self.tgt_sos_id, *tgt_tokens] + [self.tgt_pad_id] * dec_padding,
            dtype=torch.long,
        )
        label = torch.tensor(
            [*tgt_tokens, self.tgt_eos_id] + [self.tgt_pad_id] * dec_padding,
            dtype=torch.long,
        )

        # DataLoader adds the batch dimension:
        # encoder_mask -> (batch, 1, 1, seq_len)
        # decoder_mask -> (batch, 1, seq_len, seq_len)
        encoder_mask = (encoder_input != self.src_pad_id).unsqueeze(0).unsqueeze(0)
        decoder_padding_mask = (decoder_input != self.tgt_pad_id).unsqueeze(0).unsqueeze(1)
        decoder_mask = decoder_padding_mask & causal_mask(seq_len)

        return encoder_input, decoder_input, label, encoder_mask, decoder_mask

def causal_mask(size):
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool))
