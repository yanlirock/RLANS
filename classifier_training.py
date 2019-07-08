# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import pickle
import discriminator_model as model
import data
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM classification Model')
parser.add_argument('--data', type=str, default=os.getcwd()+'/ag_news_csv/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--reduce_rate', type=float, default=0.95,
                    help='learning rate reduce rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--nclass', type=int, default=4,
                    help='number of class in classification')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout_em', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_rnn', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_cl', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--number_per_class', type=int, default=1000,
                    help='location of the data corpus')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--save', type=str,
                    default=os.getcwd()+'/classify_with_pre/',
                    help='path to save the final model')
parser.add_argument('--pre_train', type=str,
                    default=os.getcwd()+'/ag_lm_model/',
                    help='path to save the final model')

args = parser.parse_args()

# create the directory to save model if the directory is not exist
if not os.path.exists(args.save):
    os.makedirs(args.save)
resume = args.save+'resume_checkpoint/'
if not os.path.exists(resume):
    os.makedirs(resume)
result_dir = args.save+'result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
dic_exists = os.path.isfile(os.path.join(args.data, 'action_dictionary.pkl'))
if dic_exists:
    with open(os.path.join(args.data, 'action_dictionary.pkl'), 'rb') as input:
        Corpus_Dic = pickle.load(input)
else:
    Corpus_Dic = data.Dictionary()

train_data_name = os.path.join(args.data, str(args.number_per_class)+'_labeled_train.csv')
test_data_name = os.path.join(args.data, 'test.csv')

train_data = data.Csv_DataSet(train_data_name)
test_data = data.Csv_DataSet(test_data_name)
train_data.load(dictionary=Corpus_Dic)
test_data.load(dictionary=Corpus_Dic, train_mode=False)

# save the dictionary
if not dic_exists:
    with open(os.path.join(args.data, 'action_dictionary.pkl'), 'wb') as output:
        pickle.dump(Corpus_Dic, output, pickle.HIGHEST_PROTOCOL)
    print("load data and save the dictionary to '{}'".
          format(os.path.join(args.data, 'action_dictionary.pkl')))

bitch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=bitch_size,
                                           shuffle=True,
                                           collate_fn=data.collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=bitch_size,
                                          shuffle=True,
                                          collate_fn=data.collate_fn)

print('The size of the dictionary is', len(Corpus_Dic))

###############################################################################
# Build the model
###############################################################################
learning_rate = args.lr

ntokens = len(Corpus_Dic)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, args.nclass, args.dropout_em, 
                       args.dropout_rnn, args.dropout_cl, args.tied).to(device)

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.reduce_rate)

###############################################################################
# Training code
###############################################################################


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for i_batch, sample_batched in enumerate(train_loader):
        # the sample batched has the following information
        # {token_seqs, next_token_seqs, importance_seqs, labels, seq_lengths, pad_length}
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        token_seqs = torch.from_numpy(np.transpose(sample_batched[0])).to(device)
        labels = torch.from_numpy(np.transpose(sample_batched[3])).to(device)
        seq_lengths = np.transpose(sample_batched[4])
        hidden = model.init_hidden(token_seqs.shape[1])
        output = model(token_seqs, hidden, seq_lengths)
        element_loss = criterion(output, labels)
        loss = torch.mean(element_loss)
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called.
        optimizer.zero_grad()

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.4f} | ppl {:8.2f}'.format(
                epoch, i_batch, len(train_data) // args.batch_size,
                                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

###############################################################################
# Evaluate code
###############################################################################


def evaluate():
    # Turn on evaluate mode which disables dropout.
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            token_seqs = torch.from_numpy(np.transpose(sample_batched[0])).to(device)
            labels = torch.from_numpy(np.transpose(sample_batched[3])).to(device)
            seq_lengths = np.transpose(sample_batched[4])
            hidden = model.init_hidden(token_seqs.shape[1])
            output = model(token_seqs, hidden, seq_lengths)
            _, predict_class = torch.max(output, 1)
            total += labels.size(0)
            correct += (predict_class == labels).sum().item()
        print('Accuracy of the classifier on the test data is : {:5.4f}'.format(
                100 * correct / total))
        return correct / total


###############################################################################
# The learning process
###############################################################################
epoch = 0
resume_file = os.path.join(resume, 'classifier_checkpoint.pth.tar')
pre_trained_lm_model_file = os.path.join(args.pre_train, 'lm_model.pt')
result_file = os.path.join(result_dir, 'result.csv')

if os.path.isfile(result_file):
    all_result_df = pd.read_csv(result_file)
else:
    all_result_df = pd.DataFrame(columns=['batch', 'accuracy'])

# At any point you can hit Ctrl + C to break out of training early.
try:
    # first check if there is a checkpoint for resuming
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        scheduler = checkpoint['scheduler']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        # if no checkpoint for resuming then check is there is a pre_trained language model
        print("=> no checkpoint found at '{}'".format(resume_file))
        print("Now check if there is a pre_trained language model")
        if os.path.isfile(pre_trained_lm_model_file):
            print("=> Initialize the classification model with '{}'".
                  format(pre_trained_lm_model_file))
            pre_trained_lm_model = torch.load(pre_trained_lm_model_file)
            model.load_state_dict(pre_trained_lm_model.state_dict(), strict=False)
        else:
            print("=> No pretrained language model can be found at '{}'".
                  format(pre_trained_lm_model_file))
        start_epoch = 1

    best_accuracy = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        scheduler.step()
        train()
        current_accuracy = evaluate()
        cdf = pd.DataFrame([[epoch, current_accuracy]], columns=['batch', 'accuracy'])
        all_result_df = all_result_df.append(cdf, ignore_index=True)
        # Save the model if the validation loss is the best we've seen so far.
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            with open(os.path.join(args.save, 'classifier_model.pt'), 'wb') as f:
                torch.save(model, f)
    all_result_df.to_csv(result_file, index=False, header=True)


    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'scheduler': scheduler,
         'optimizer': optimizer.state_dict()
         }, resume_file)
    print('-' * 89)
    print("save the check point to '{}'".format(resume_file))

except KeyboardInterrupt:
    print('-' * 89)
    print("Exiting from training early")
    print("save the check point to '{}'".format(resume_file))
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'scheduler': scheduler,
         'optimizer': optimizer.state_dict()
         }, resume_file)
    print("save the current result to '{}'".format(result_file))
    all_result_df.to_csv(result_file, index=False, header=True)

print('=' * 89)
print('End of training and evaluation')
