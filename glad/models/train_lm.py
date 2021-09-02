from pickle import load
import math
import torch
import torch.nn as nn

LEARN_RATE = 2e-4

def train(lm, data_loader, new_model_path, epoch_num, device='cuda'):
    lm.train()
    optimizer = torch.optim.Adam(lm.parameters(), lr=LEARN_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    for e_idx in range(0, epoch_num):
        for b_idx, (pad_x, lengths) in enumerate(data_loader):
            pad_x = pad_x.to(device)
            x_logits = lm(pad_x, lengths)

            x_logits_flat = x_logits.view(-1, lm.vocab_size)
            x_labels_flat = pad_x[1:].view(-1)
            dec_loss = loss_fn(x_logits_flat, x_labels_flat)
            loss = torch.mean(dec_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b_idx % 500 == 0:
                print(f'{b_idx}/{e_idx}: total {loss.item():.2f} = dec {dec_loss.item():.2f} (1/{math.e**dec_loss.item():.1f})')
        torch.save(lm.state_dict(), new_model_path)

if __name__ == '__main__':
    ## hyper-parameters
    from model import LanguageModel, CNNModel
    from data_loader import get_SrcMLLinear_loader, get_SmallLinear_loader
    train_epochs = 5
    layers = 1
    z_dim = 2048
    batch_size = 32
    bidirectional = False
    train_file = '../data/java-med-methods'
    device = 'cuda'
    vocab_file = '../etc_data/jmtrain_laxlit_v2_vocab_BPE.pkl'
    varpair_file = '../etc_data/jmtrain_laxlit_v2_varpairs_BPE.pkl'
    with open(vocab_file, 'rb') as f:
        char2idx = load(f)
        vocab_size = (max(char2idx.values())+1+2)+1 # +1 for pad
    print('Vocab size:', vocab_size)

    lm = LanguageModel(vocab_size, emb_dim=z_dim, hidden_size=z_dim, num_layers=layers)
    lm.to(device)

    train_data_loader = get_SrcMLLinear_loader(train_file, varpair_file, vocab_file, batch_size, num_workers=4)
    train_size = len(train_data_loader)
    print('Training data size:', train_size)
    print('Vocab size:', vocab_size)
    train(lm, train_data_loader, 'weights/jmBPELitON_GRU_L1W2048.pth', train_epochs, device=device)
