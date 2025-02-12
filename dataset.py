import pickle
from torch.utils.data import Dataset
import torch
from nltk.tokenize import word_tokenize
from utils import preprocess

# Reserve tokens
word_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
idx_to_word = ['<pad>', '<sos>', '<eos>']
current_idx = 3

# Example conversations – feel free to expand these!
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
    ("how do i write a loop in python", "you can write a loop in python using for i in range(n): print(i)"),
    ("how do i define a function", "in python you define a function using def keyword for example: def my_function(): pass"),
    ("how do i reverse a string", "you can reverse a string using slicing for example my_string[::-1]"),
    ("what is a list comprehension", "a list comprehension is a concise way to create lists for example: [x for x in iterable if condition]"),
    ("how do i create a dictionary", "you can create a dictionary using curly braces for example: my_dict = {'key': 'value'}"),
    ("how do i read a file", "you can read a file in python using with open('filename', 'r') as f: data = f.read()"),
    ("what is pip", "pip is a package manager for python that allows you to install and manage libraries"),
    ("how do i install a package", "you can install a package using pip install package_name")
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
        src_indices = [self.word_to_idx.get(token, self.word_to_idx['<pad>']) for token in src_tokens]
        trg_indices = ([self.word_to_idx['<sos>']] +
                       [self.word_to_idx.get(token, self.word_to_idx['<pad>']) for token in trg_tokens] +
                       [self.word_to_idx['<eos>']])
        src_indices = src_indices[:self.max_len]
        trg_indices = trg_indices[:self.max_len]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(s) for s in src_batch]
    trg_lengths = [len(t) for t in trg_batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=word_to_idx['<pad>'])
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=word_to_idx['<pad>'])
    return src_padded, torch.tensor(src_lengths), trg_padded, torch.tensor(trg_lengths)
