import os
import json
import random
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChatDataset, collate_fn, conversations, word_to_idx, idx_to_word, vocab_size
from model import Encoder, Decoder, Seq2Seq
from utils import save_model, load_model

def validate_model(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, src_lengths, trg, trg_lengths in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, optimizer, criterion, train_loader, val_loader, device, epochs, clip=1.0):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for src, src_lengths, trg, trg_lengths in train_loader:
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src, src_lengths, trg)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        val_loss = validate_model(model, criterion, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}, Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    return model

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = config["max_len"]
    emb_dim = config["emb_dim"]
    hid_dim = config["hid_dim"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_path = config["model_path"]
    vocab_path = config["vocab_path"]

    encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])

    # Split data into training and validation
    all_data = conversations[:]
    random.shuffle(all_data)
    split_index = int(0.8 * len(all_data))
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]
    train_dataset = ChatDataset(train_data, word_to_idx, max_len)
    val_dataset = ChatDataset(val_data, word_to_idx, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        model, optimizer = load_model(model, optimizer, model_path, device)
        print("Loaded existing model and vocabulary.")
    else:
        print("No existing model found. Training from scratch.")

    model = train(model, optimizer, criterion, train_loader, val_loader, device, epochs)
    save_model(model, optimizer, model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_idx, idx_to_word, vocab_size), f)
    print("Training complete. Model saved.")

if __name__ == '__main__':
    main()
