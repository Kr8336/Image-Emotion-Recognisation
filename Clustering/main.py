import os
import sys
import json
import time

from torch.utils.data import DataLoader
from Clustering.image_captioning import *
from Clustering.dataloader import collate_batch
from Clustering.model import TextClassificationModel
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset


def print_helper(num):
    print('Executing Line', num)


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text[0])



## Change these stupid arguments to kwargs***
def train(model, dataloader, criterion, optimizer, epoch):

    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            print(loss)
    return total_acc/total_count

def main():

    folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine ' \
                  r'Learning\datasets_coursework2\Flickr\Flickr'
    save_text = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Image-Emotion-Recognisation\captions.json'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Make a dictionary from image name --> string generated...

    if not os.path.isfile(save_text):
        strings, dataset = generate_image_captions(folder_name)
        with open(save_text, 'w') as file:
            print('Generated Image Captions!')
            json.dump(dataset, file)

    else:
        with open(save_text, 'r') as file:
            print('Found Image Captions Dataset')
            dataset = json.load(file)

    tokenizer = get_tokenizer('basic_english')
    print_helper(85)

    train_iter = iter(dataset[:int(len(dataset)*.9)])
    test_iter = iter(dataset[int(len(dataset)*.9):])
    print_helper(89)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print_helper(93)

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)
    print_helper(96)

    # dataloader = DataLoader(train_iter, batch_size=4, shuffle=False, collate_fn=collate_batch)

    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    print_helper(107)
    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 0.1  # learning rate
    BATCH_SIZE = 4 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    print_helper(117)

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    print_helper(123)

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)
    print_helper(131)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, criterion, optimizer, epoch)
        print_helper(136)

        accu_val = evaluate(model, valid_dataloader, criterion)
        print_helper(139)

        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader)
    print_helper(154)
    print('test accuracy {:8.3f}'.format(accu_test))

if __name__ == '__main__':
    main()