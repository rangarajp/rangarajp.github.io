from pathlib import Path


def get_config():
    return {
        'batch_size': 128,
        'seq_len': 100,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 10,
        'lang_src': 'en',
        'lang_tgt': 'fr',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'tokenizer_file': 'tokenizer_{lang}.json',

    }

def get_weights_file_path(config, epoch):

    model_path = Path(config['model_folder']) / f"{config['model_basename']}{epoch}.pth"
    return model_path
    