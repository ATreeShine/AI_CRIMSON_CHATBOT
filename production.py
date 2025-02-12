import os
import pickle
import torch
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from model import Encoder, Decoder, Seq2Seq
from utils import load_model, preprocess
from dataset import word_to_idx, idx_to_word, vocab_size

def beam_search(model, src_tensor, src_length, beam_width, max_len, word_to_idx, idx_to_word, device, length_penalty=0.7):
    model.eval()
    with torch.no_grad():
        src_tensor = src_tensor.unsqueeze(0).to(device)
        src_length = torch.tensor([src_length])
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
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
                log_probs = torch.log_softmax(output, dim=1)
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
        result = [idx_to_word[idx] for idx in best_seq if idx not in (word_to_idx['<sos>'], word_to_idx['<pad>'])]
        if '<eos>' in result:
            result = result[:result.index('<eos>')]
        return result

def chat(model, word_to_idx, idx_to_word, device, max_len, beam_width):
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        user_input = preprocess(user_input)
        tokens = word_tokenize(user_input)
        src_indices = [word_to_idx.get(token, word_to_idx['<pad>']) for token in tokens]
        src_indices = src_indices[:max_len]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        generated_tokens = beam_search(model, src_tensor, len(src_indices), beam_width, max_len, word_to_idx, idx_to_word, device)
        response_text = ' '.join(generated_tokens)
        print(f"Bot: {response_text}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 50
    emb_dim = 128
    hid_dim = 256
    n_layers = 2
    dropout = 0.5
    learning_rate = 0.001
    model_path = "chatbot_seq2seq.pth"
    vocab_path = "vocab.pkl"
    encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        model, optimizer = load_model(model, optimizer, model_path, device)
        with open(vocab_path, 'rb') as f:
            global word_to_idx, idx_to_word, vocab_size
            word_to_idx, idx_to_word, vocab_size = pickle.load(f)
        print("Loaded existing model and vocabulary.")
    else:
        print("No trained model found. Please run train.py first.")
        return
    chat(model, word_to_idx, idx_to_word, device, max_len, beam_width=5)

if __name__ == '__main__':
    main()
