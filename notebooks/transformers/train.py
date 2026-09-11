from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from dataset import BilingualDataset
from config import get_config, get_weights_file_path
from pathlib import Path
from tqdm import tqdm
import torch


from model import BuildTransformer

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='<unk>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=['<pad>', '<unk>', '<sos>', '<eos>'],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(
            get_all_sentences(dataset, lang),
            trainer=trainer,
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def tokenize_batch(batch, tokenizer_src, tokenizer_tgt, config):
    """Create fixed-length encoder inputs, decoder inputs, and labels."""
    seq_len = config['seq_len']
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']

    pad_id = tokenizer_src.token_to_id('<pad>')
    src_sos_id = tokenizer_src.token_to_id('<sos>')
    src_eos_id = tokenizer_src.token_to_id('<eos>')
    tgt_sos_id = tokenizer_tgt.token_to_id('<sos>')
    tgt_eos_id = tokenizer_tgt.token_to_id('<eos>')

    special_ids = (pad_id, src_sos_id, src_eos_id, tgt_sos_id, tgt_eos_id)
    if any(token_id is None for token_id in special_ids):
        raise ValueError("Tokenizers must contain <pad>, <sos>, and <eos> tokens")

    encoder_inputs = []
    decoder_inputs = []
    labels = []
    source_texts = []
    target_texts = []

    for translation in batch['translation']:
        source_text = translation[lang_src]
        target_text = translation[lang_tgt]
        source_ids = tokenizer_src.encode(source_text).ids
        target_ids = tokenizer_tgt.encode(target_text).ids

        if len(source_ids) + 2 > seq_len or len(target_ids) + 1 > seq_len:
            raise ValueError(
                f"Sentence is too long for seq_len={seq_len}: "
                f"source={len(source_ids) + 2}, target={len(target_ids) + 1}"
            )

        encoder_inputs.append(
            [src_sos_id, *source_ids, src_eos_id]
            + [pad_id] * (seq_len - len(source_ids) - 2)
        )
        decoder_inputs.append(
            [tgt_sos_id, *target_ids]
            + [pad_id] * (seq_len - len(target_ids) - 1)
        )
        labels.append(
            [*target_ids, tgt_eos_id]
            + [pad_id] * (seq_len - len(target_ids) - 1)
        )
        source_texts.append(source_text)
        target_texts.append(target_text)

    return {
        'encoder_input': encoder_inputs,
        'decoder_input': decoder_inputs,
        'label': labels,
        'source_text': source_texts,
        'target_text': target_texts,
    }


def get_ds(config):
    ds_raw = load_dataset(
        'Helsinki-NLP/opus_books',
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split='train',
    )
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
   
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source text: {max_len_src}')
    print(f'Max length of target text: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_size, vocab_tgt_size, pad_token_id):
    model = BuildTransformer(
        src_vocab_size=vocab_src_size,
        tgt_vocab_size=vocab_tgt_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_layers=config['n_layers'],
        d_ff=config.get('d_ff'),
        max_seq_len=config['seq_len'],
        pad_token_id=pad_token_id,
    )
    return model

def train_model(model, config, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.NLLLoss(ignore_index=tokenizer_tgt.token_to_id('<pad>'))
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1}")
        train_loss = 0.0

        for batch in batch_iterator:
            encoder_input, decoder_input, labels, encoder_mask, decoder_mask = (
                tensor.to(device) for tensor in batch
            )

            optimizer.zero_grad(set_to_none=True)
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = loss_fn(
                output.reshape(-1, output.size(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_iterator.set_postfix(loss=f'{loss.item():.4f}')

        val_loss = validate_model(model, val_dataloader, loss_fn, device)
        average_train_loss = train_loss / max(len(train_dataloader), 1)
        print(
            f'Epoch {epoch + 1}: '
            f'train_loss={average_train_loss:.4f}, val_loss={val_loss:.4f}'
        )

        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': average_train_loss,
                'val_loss': val_loss,
            },
            get_weights_file_path(config, epoch + 1),
        )


@torch.no_grad()
def validate_model(model, val_dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(val_dataloader, desc='Validating', leave=False):
        encoder_input, decoder_input, labels, encoder_mask, decoder_mask = (
            tensor.to(device) for tensor in batch
        )
        output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
        loss = loss_fn(
            output.reshape(-1, output.size(-1)),
            labels.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / max(len(val_dataloader), 1)


def main():
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        tokenizer_tgt.token_to_id('<pad>'),
    )
    train_model(
        model,
        config,
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    )


if __name__ == '__main__':
    main()
