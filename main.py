import os
import re
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- NLP Preprocessing ---
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # Ensure the punkt_tab resource is available
from nltk.tokenize import word_tokenize

def preprocess(text):
    """Lowercase and remove unwanted characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]+", "", text)
    return text.strip()

# --- Save / Load Functions ---
def save_model(model, optimizer, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)

def load_model(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model, optimizer

# --- Build Vocabulary ---
# Reserve tokens: <pad>, <sos>, <eos>
word_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
idx_to_word = ['<pad>', '<sos>', '<eos>']
current_idx = 3

# Example conversations (feel free to expand)
conversations = [
    ("hello", "hi"),
    ("how are you", "i am doing well thank you"),
    ("what is your name", "i am a chatbot"),
    ("tell me a joke", "why did the scarecrow win an award because he was outstanding in his field"),
    ("good morning", "good morning to you too"),
    ("good afternoon", "good afternoon"),
    ("good night", "good night sleep well"),
    ("how is the weather", "i am sorry i cannot provide real time weather information"),
    ("thank you", "you are welcome"),
    ("what can you do", "i can chat with you and try to answer your questions"),
    ("nice to meet you", "nice to meet you too"),
    ("bye", "goodbye"),
    ("see you later", "see you later"),
    ("talk to you soon", "okay talk to you soon"),
    ("have a good day", "you too"),
    ("have a nice day", "you too have a nice day"),
    ("how do you work", "i work by using machine learning models to generate responses to your inputs"),
    ("are you intelligent", "i am designed to simulate conversation but i dont have consciousness or real intelligence"),
    ("who created you", "i was created by a programmer"),
    ("what is your purpose", "my purpose is to assist and chat with users"),
    ("what is the meaning of life", "the meaning of life is a question that has puzzled philosophers for centuries"),
    ("tell me something interesting", "did you know honey never spoils archaeologists have found pots of honey in ancient egyptian tombs that are thousands of years old and still perfectly edible"),
    ("i am bored", "how about we play a game or i can tell you a story"),
    ("can you help me", "i will try my best how can i help you today"),
    ("what are your hobbies", "as a chatbot i dont have hobbies in the way humans do but i enjoy learning new things and helping people"),
    ("do you like books", "yes i like books in the sense that i can process and understand information from them they are a great source of knowledge"),
    ("what is your favorite color", "as a chatbot i dont perceive colors but i understand that colors are an important aspect of human experience"),
    ("can you sing", "i cant sing in the way humans do but i can generate text that could be considered song lyrics"),
]

# Build the vocabulary using NLTK tokenization
for pair in conversations:
    for sentence in pair:
        sentence = preprocess(sentence)
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token not in word_to_idx:
                word_to_idx[token] = current_idx
                idx_to_word.append(token)
                current_idx += 1
vocab_size = len(word_to_idx)

# --- Dataset & Collate Function ---
class ChatDataset(Dataset):
    def __init__(self, conversations, word_to_idx, max_len=50):
        self.conversations = conversations
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        src_text, trg_text = self.conversations[idx]
        src_text = preprocess(src_text)
        trg_text = preprocess(trg_text)
        src_tokens = word_tokenize(src_text)
        trg_tokens = word_tokenize(trg_text)
        # Convert tokens to indices (using <pad> for unknowns)
        src_indices = [self.word_to_idx.get(token, self.word_to_idx['<pad>']) for token in src_tokens]
        # For target, add <sos> at the beginning and <eos> at the end.
        trg_indices = ([self.word_to_idx['<sos>']] +
                       [self.word_to_idx.get(token, self.word_to_idx['<pad>']) for token in trg_tokens] +
                       [self.word_to_idx['<eos>']])
        # Limit sequence length
        src_indices = src_indices[:self.max_len]
        trg_indices = trg_indices[:self.max_len]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    """Pads source and target sequences dynamically for the batch."""
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(s) for s in src_batch]
    trg_lengths = [len(t) for t in trg_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=word_to_idx['<pad>'])
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=word_to_idx['<pad>'])
    return src_padded, torch.tensor(src_lengths), trg_padded, torch.tensor(trg_lengths)

# --- Model: Encoder, Decoder (with Attention), and Seq2Seq ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=word_to_idx['<pad>'])
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch, src_len, hid_dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=word_to_idx['<pad>'])
        # The decoder receives the embedded token and the context vector (from attention)
        self.gru = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch] --> [batch, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]
        # Compute attention (dot-product between last hidden state and encoder outputs)
        hidden_last = hidden[-1].unsqueeze(2)  # [batch, hid_dim, 1]
        attn_energies = torch.bmm(encoder_outputs, hidden_last).squeeze(2)  # [batch, src_len]
        attn_weights = torch.softmax(attn_energies, dim=1)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hid_dim]
        # Concatenate embedded and context
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim+hid_dim]
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(1)   # [batch, hid_dim]
        context = context.squeeze(1) # [batch, hid_dim]
        embedded = embedded.squeeze(1) # [batch, emb_dim]
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch, output_dim]
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        # First input to the decoder is <sos>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

# --- Training Function ---
def train_model(model, optimizer, criterion, dataloader, device, epochs=10, clip=1.0):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for src, src_lengths, trg, trg_lengths in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output = model(src, src_lengths, trg)  # [batch, trg_len, output_dim]
            output_dim = output.shape[-1]
            # Exclude the first token (<sos>) for computing loss
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# --- Beam Search Decoding ---
def beam_search(model, src_tensor, src_length, beam_width, max_len, word_to_idx, idx_to_word, device, length_penalty=0.7):
    model.eval()
    with torch.no_grad():
        # Prepare the source (add batch dimension)
        src_tensor = src_tensor.unsqueeze(0).to(device)  # [1, src_len]
        src_length = torch.tensor([src_length])
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
        # Start with the <sos> token
        beams = [([word_to_idx['<sos>']], 0.0, hidden)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden_state in beams:
                if seq[-1] == word_to_idx['<eos>']:
                    completed.append((seq, score))
                    continue
                input_token = torch.tensor([seq[-1]], device=device)
                output, hidden_new, _ = model.decoder(input_token, hidden_state, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1)  # [1, vocab_size]
                top_log_probs, top_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    next_token = top_indices[0][i].item()
                    next_score = score + top_log_probs[0][i].item()
                    new_seq = seq + [next_token]
                    new_beams.append((new_seq, next_score, hidden_new))
            beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)[:beam_width]
            if all(seq[-1] == word_to_idx['<eos>'] for seq, score, _ in beams):
                break
        if not completed:
            completed = beams
        best_seq = sorted(completed, key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)[0][0]
        # Remove special tokens from the result.
        result = [idx_to_word[idx] for idx in best_seq if idx not in (word_to_idx['<sos>'], word_to_idx['<pad>'])]
        if '<eos>' in result:
            result = result[:result.index('<eos>')]
        return result

# --- Chat Function with Online Learning ---
def chat(model, word_to_idx, idx_to_word, device, max_len=50, beam_width=5):
    print("Chatbot is ready. Type 'exit' to quit.")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        user_input = preprocess(user_input)
        tokens = word_tokenize(user_input)
        src_indices = [word_to_idx.get(token, word_to_idx['<pad>']) for token in tokens]
        src_indices = src_indices[:max_len]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        generated_tokens = beam_search(model, src_tensor, len(src_indices), beam_width,
                                       max_len, word_to_idx, idx_to_word, device)
        response_text = ' '.join(generated_tokens)
        print(f"Bot: {response_text}")

        feedback = input("Did I say something wrong? (yes/no) ")
        if feedback.lower() == 'yes':
            correct_response = input("What should I have said? ")
            correct_response = preprocess(correct_response)
            conversation_history.append((user_input, correct_response))
            print("Thank you for your feedback. I'll learn from this.")

            # Fine-tune the model on the new example.
            new_conversation = [(user_input, correct_response)]
            new_dataset = ChatDataset(new_conversation, word_to_idx, max_len)
            new_dataloader = DataLoader(new_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
            train_model(model, optimizer, criterion, new_dataloader, device, epochs=10)
            save_model(model, optimizer, 'chatbot_seq2seq.pth')
            with open('vocab.pkl', 'wb') as f:
                pickle.dump((word_to_idx, idx_to_word, vocab_size), f)

# --- Hyperparameters and Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 50
emb_dim = 128
hid_dim = 256
n_layers = 2
dropout = 0.5
learning_rate = 0.001
batch_size = 32

# --- Instantiate the Model ---
encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
decoder = Decoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])

# --- Load Existing Model/Vocabulary if Available ---
if os.path.exists('chatbot_seq2seq.pth') and os.path.exists('vocab.pkl'):
    with open('vocab.pkl', 'rb') as f:
        word_to_idx_loaded, idx_to_word_loaded, vocab_size_loaded = pickle.load(f)
        if vocab_size_loaded == vocab_size:
            word_to_idx = word_to_idx_loaded
            idx_to_word = idx_to_word_loaded
            vocab_size = vocab_size_loaded
            model, optimizer = load_model(model, optimizer, 'chatbot_seq2seq.pth', device)
            print("Loaded existing model and vocabulary.")
        else:
            print("Vocabulary size mismatch. Training from scratch.")
else:
    print("No existing model found. Training from scratch.")

# --- Prepare Dataset and Train ---
dataset = ChatDataset(conversations, word_to_idx, max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Increase epochs if needed – initial training
train_model(model, optimizer, criterion, dataloader, device, epochs=100)
save_model(model, optimizer, 'chatbot_seq2seq.pth')
with open('vocab.pkl', 'wb') as f:
    pickle.dump((word_to_idx, idx_to_word, vocab_size), f)

# --- Start Chatting ---
chat(model, word_to_idx, idx_to_word, device, max_len, beam_width=5)
