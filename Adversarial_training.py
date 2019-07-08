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
import discriminator_model as discriminator
import judge_model as judge
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
parser.add_argument('--clip', type=float, default=5.0,
                    help='gradient clipping')
parser.add_argument('--nclass', type=int, default=4,
                    help='number of class in classification')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
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
                    default=os.getcwd()+'/ag_adv_model/',
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

labeled_train_data_name = os.path.join(args.data, str(args.number_per_class)+'_labeled_train.csv')
unlabeled_train_data_name = os.path.join(args.data, str(args.number_per_class)+'_unlabeled_train.csv')
test_data_name = os.path.join(args.data, 'test.csv')

labeled_train_data = data.Csv_DataSet(labeled_train_data_name)
unlabeled_train_data = data.Csv_DataSet(unlabeled_train_data_name)
test_data = data.Csv_DataSet(test_data_name)

labeled_train_data.load(dictionary=Corpus_Dic)
unlabeled_train_data.load(dictionary=Corpus_Dic)
test_data.load(dictionary=Corpus_Dic,train_mode=False)

# save the dictionary
if not dic_exists:
    with open(os.path.join(args.data, 'action_dictionary.pkl'), 'wb') as output:
        pickle.dump(Corpus_Dic, output, pickle.HIGHEST_PROTOCOL)
    print("load data and save the dictionary to '{}'".
          format(os.path.join(args.data, 'action_dictionary.pkl')))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
bitch_size = args.batch_size
labeled_train_loade = torch.utils.data.DataLoader(
    dataset=labeled_train_data,
    batch_size=bitch_size,
    shuffle=True,
    collate_fn=data.collate_fn)
labeled_train_loader = iter(cycle(labeled_train_loade))

unlabeled_train_loade = torch.utils.data.DataLoader(
    dataset=unlabeled_train_data,
    batch_size=bitch_size,
    shuffle=True,
    collate_fn=data.collate_fn)
unlabeled_train_loader = iter(cycle(unlabeled_train_loade))

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=bitch_size,
    shuffle=True,
    collate_fn=data.collate_fn)

print('The size of the dictionary is', len(Corpus_Dic))

###############################################################################
# Build the model
###############################################################################
dis_learning_rate = args.lr
judge_learning_rate = args.lr

ntokens = len(Corpus_Dic)
discriminator = discriminator.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, args.nclass, args.dropout_em, 
                       args.dropout_rnn, args.dropout_cl, args.tied).to(device)
judger = judge.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, args.nclass, args.dropout_em, 
                       args.dropout_rnn, args.dropout_cl, args.tied).to(device)

criterion = nn.CrossEntropyLoss(reduction='none')
criterion_judge = nn.BCELoss()
dis_optimizer = torch.optim.SGD(discriminator.parameters(), lr=dis_learning_rate, momentum=0.9)
dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size=10, gamma=args.reduce_rate)

judge_optimizer = torch.optim.SGD(judger.parameters(), lr=judge_learning_rate, momentum=0.9)
judge_scheduler = torch.optim.lr_scheduler.StepLR(judge_optimizer, step_size=5, gamma=args.reduce_rate)

###############################################################################
# Training code
###############################################################################


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def dis_pre_train_step():
    discriminator.train()
    lab_token_seqs, _, _, labels, lab_seq_lengths, _ = next(labeled_train_loader)
    lab_token_seqs = torch.from_numpy(np.transpose(lab_token_seqs)).to(device)
    labels = torch.from_numpy(np.transpose(labels)).to(device)
    num_lab_sample = lab_token_seqs.shape[1]
    lab_hidden = discriminator.init_hidden(num_lab_sample)
    lab_output = discriminator(lab_token_seqs, lab_hidden, lab_seq_lengths)
    lab_element_loss = criterion(lab_output, labels)
    lab_loss = torch.mean(lab_element_loss)
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called.
    dis_optimizer.zero_grad()

    lab_loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip)
    dis_optimizer.step()

    return lab_loss

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def adv_train_step(judge_only=True):
    discriminator.train()
    judger.train()

    # {token_seqs, next_token_seqs, importance_seqs, labels, seq_lengths, pad_length}
    # Sample m labeled instances from DL
    lab_token_seqs, _, _, labels, lab_seq_lengths, _ = next(labeled_train_loader)
    lab_token_seqs = torch.from_numpy(np.transpose(lab_token_seqs)).to(device)
    labels = torch.from_numpy(np.transpose(labels)).to(device)
    num_lab_sample = lab_token_seqs.shape[1]
    
    # Sample m labeled instances from DU and predict their corresponding label
    unl_token_seqs, _, _, _, unl_seq_lengths, _ = next(unlabeled_train_loader)
    unl_token_seqs = torch.from_numpy(np.transpose(unl_token_seqs)).to(device)
    num_unl_sample = unl_token_seqs.shape[1]
    unl_hidden = discriminator.init_hidden(num_unl_sample)
    unl_output = discriminator(unl_token_seqs, unl_hidden, unl_seq_lengths)
    _, fake_labels = torch.max(unl_output, 1)

    if judge_only:
        k = 1
    else:
        k = 3

    for _k in range(k):
        # Update the judge model
        ###############################################################################
        lab_judge_hidden = judger.init_hidden(num_lab_sample)
        one_hot_label = one_hot_embedding(labels, args.nclass).to(device)  # one hot encoder
        lab_judge_prob = judger(lab_token_seqs, lab_judge_hidden, lab_seq_lengths, one_hot_label)
        lab_labeled = torch.ones(num_lab_sample).to(device)

        unl_judge_hidden = judger.init_hidden(num_unl_sample)
        one_hot_unl = one_hot_embedding(fake_labels, args.nclass).to(device)  # one hot encoder
        unl_judge_prob = judger(unl_token_seqs, unl_judge_hidden, unl_seq_lengths, one_hot_unl)
        unl_labeled = torch.zeros(num_unl_sample).to(device)
        
        if_labeled = torch.cat((lab_labeled, unl_labeled))
        all_judge_prob = torch.cat((lab_judge_prob, unl_judge_prob))
        all_judge_prob = all_judge_prob.view(-1)
        judge_loss = criterion_judge(all_judge_prob, if_labeled)
        judge_optimizer.zero_grad()

        judge_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(judger.parameters(), args.clip)
        judge_optimizer.step()

        unl_loss_value = 0.0
        lab_loss_value = 0.0
        fake_labels = repackage_hidden(fake_labels)
        unl_judge_prob = repackage_hidden(unl_judge_prob)
        if not judge_only:
            # Update the predictor
            ###############################################################################
            lab_hidden = discriminator.init_hidden(num_lab_sample)
            lab_output = discriminator(lab_token_seqs, lab_hidden, lab_seq_lengths)
            lab_element_loss = criterion(lab_output, labels)
            lab_loss = torch.mean(lab_element_loss)

            # calculate loss for unlabeled instances
            unl_hidden = discriminator.init_hidden(num_unl_sample)
            unl_output = discriminator(unl_token_seqs, unl_hidden, unl_seq_lengths)
            unl_element_loss = criterion(unl_output, fake_labels)
            unl_loss = unl_element_loss.dot(unl_judge_prob.view(-1))/num_unl_sample
            # do not include this in version 1 
            if _k<int(k/2):
                lab_unl_loss = lab_loss+unl_loss
            else:
                lab_unl_loss = unl_loss
            dis_optimizer.zero_grad()
            lab_unl_loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip)
            dis_optimizer.step()
            
            unl_loss_value = unl_loss.item()
            lab_loss_value = lab_loss.item()

    return judge_loss, unl_loss_value, lab_loss_value


###############################################################################
# Training process
###############################################################################

def train(epoch=None):
    # 1. pre_train discriminator.
    if epoch < 30:#30
        num_iter = len(labeled_train_data) // args.batch_size
        start_time = time.time()
        total_loss = 0
        for i_iter in range(num_iter):
            dis_loss = dis_pre_train_step()
            total_loss += dis_loss.item()
        elapsed = time.time() - start_time
        cur_loss = total_loss/num_iter
        print('Pre_train discriminator labeled_data only | epoch {:3d} | ms/batch {:5.2f} | '
              'labeled loss {:5.4f} | ppl {:8.4f}'.format(
            epoch, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
    # 2. pre_train judger and adv train.
    else:
        judge_scheduler.step()
        if epoch < 205:#35
            judge_only = True
            current_process = 'Pre_train judger: '
        else:
            judge_only = False
            current_process = 'Adv train: '
        num_iter = len(unlabeled_train_data) // args.batch_size
        start_time = time.time()
        total_judge_loss = 0
        total_unl_loss = 0
        total_lab_loss = 0
        for i_iter in range(num_iter):
            judge_loss, unl_loss_value, lab_loss_value = adv_train_step(judge_only=judge_only)
            total_judge_loss += judge_loss.item()
            total_unl_loss += unl_loss_value
            total_lab_loss += lab_loss_value

            if i_iter % args.log_interval == 0 and i_iter > 0:
                cur_judge_loss = total_judge_loss / args.log_interval
                cur_unl_loss = total_unl_loss / args.log_interval
                cur_lab_loss = total_lab_loss / args.log_interval
                elapsed = time.time() - start_time
                print(current_process+'| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                      'judge_loss {:5.4f} | unlabel_loss {:5.4f} |label_loss {:5.4f}'.format(
                    epoch, i_iter, num_iter, elapsed * 1000 / args.log_interval, cur_judge_loss,
                    cur_unl_loss, cur_lab_loss))
                total_judge_loss = 0
                total_unl_loss = 0
                total_lab_loss = 0
                start_time = time.time()

###############################################################################
# Evaluate code
###############################################################################


def evaluate():
    # Turn on evaluate mode which disables dropout.
    correct = 0
    total = 0
    discriminator.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            token_seqs = torch.from_numpy(np.transpose(sample_batched[0])).to(device)
            labels = torch.from_numpy(np.transpose(sample_batched[3])).to(device)
            seq_lengths = np.transpose(sample_batched[4])
            hidden = discriminator.init_hidden(token_seqs.shape[1])
            output = discriminator(token_seqs, hidden, seq_lengths)
            _, predict_class = torch.max(output,1)
            total += labels.size(0)
            correct += (predict_class == labels).sum().item()
        print('Accuracy of the classifier on the test data is : {:5.4f}'.format(
                100 * correct / total))
        return correct / total


###############################################################################
# The learning process
###############################################################################
epoch = 0
dis_resume_file = os.path.join(resume, 'discriminator_checkpoint.pth.tar')
judge_resume_file = os.path.join(resume, 'judger_checkpoint.pth.tar')
pre_trained_lm_model_file = os.path.join(args.pre_train, 'lm_model.pt')
result_file = os.path.join(result_dir, 'result.csv')

if os.path.isfile(result_file):
    all_result_df = pd.read_csv(result_file)
else:
    all_result_df = pd.DataFrame(columns=['batch', 'accuracy'])

###############################################################################
# check if there is a chekpoint for resuming or there is a
# pretrained language model to update model
try:
    # first check if there is a checkpoint for resuming
    if os.path.isfile(dis_resume_file) and os.path.isfile(judge_resume_file):
        print("=> loading discriminator's checkpoint from '{}'".format(dis_resume_file))
        dis_checkpoint = torch.load(dis_resume_file)
        start_epoch = dis_checkpoint['epoch']
        dis_scheduler = dis_checkpoint['scheduler']
        discriminator.load_state_dict(dis_checkpoint['model_state_dict'])
        dis_optimizer.load_state_dict(dis_checkpoint['optimizer'])

        print("=> loading judger's checkpoint from '{}'".format(judge_resume_file))
        judge_checkpoint = torch.load(judge_resume_file)
        judge_scheduler = judge_checkpoint['scheduler']
        judger.load_state_dict(judge_checkpoint['model_state_dict'])
        judge_optimizer.load_state_dict(judge_checkpoint['optimizer'])

        print("=> loaded discriminator's checkpoint '{}' and judger's checkpoint '{}' (epoch {})"
              .format(dis_resume_file, judge_resume_file, dis_checkpoint['epoch']))
    else:
        # if no checkpoint for resuming then check if there is a pre_trained language model
        print("=> no checkpoint found at '{}'".format(dis_resume_file))
        print("=> no checkpoint found at '{}'".format(judge_resume_file))
        print("Now check if there is a pre_trained language model")
        if os.path.isfile(pre_trained_lm_model_file):
            print("=> Initialize the classification model with '{}'".
                  format(pre_trained_lm_model_file))
            pre_trained_lm_model = torch.load(pre_trained_lm_model_file)
            discriminator.load_state_dict(pre_trained_lm_model.state_dict(), strict=False)
            judger.load_state_dict(pre_trained_lm_model.state_dict(), strict=False)
        else:
            print("=> No pretrained language model can be found at '{}'".
                  format(pre_trained_lm_model_file))
        start_epoch = 1

###############################################################################
# this is the training loop and each loop run a batch
    best_accuracy = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        dis_scheduler.step()
        train(epoch=epoch)
        current_accuracy = evaluate()
        cdf = pd.DataFrame([[epoch, current_accuracy]], columns=['batch', 'accuracy'])
        all_result_df = all_result_df.append(cdf, ignore_index=True)
        # Save the model if the validation loss is the best we've seen so far.
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            with open(os.path.join(args.save, 'discriminator.pt'), 'wb') as f:
                torch.save(discriminator, f)
            with open(os.path.join(args.save, 'judger.pt'), 'wb') as f:
                torch.save(judger, f)

###############################################################################
# save the result and the final checkpoint
    all_result_df.to_csv(result_file, index=False, header=True)

    torch.save(
        {'epoch': epoch,
         'model_state_dict': discriminator.state_dict(),
         'scheduler': dis_scheduler,
         'optimizer': dis_optimizer.state_dict()
         }, dis_resume_file)
    torch.save(
        {'model_state_dict': judger.state_dict(),
         'scheduler': judge_scheduler,
         'optimizer': judge_optimizer.state_dict()
         }, judge_resume_file)

    print('-' * 89)
    print("save the check point to '{}' and '{}'".
          format(dis_resume_file, judge_resume_file))

###############################################################################
# At any point you can hit Ctrl + C to break out of training early.
except KeyboardInterrupt:
    print('-' * 89)
    print("Exiting from training early")
    print("save the check point to '{}' and '{}'".
          format(dis_resume_file, judge_resume_file))
    torch.save(
        {'epoch': epoch,
         'model_state_dict': discriminator.state_dict(),
         'scheduler': dis_scheduler,
         'optimizer': dis_optimizer.state_dict()
         }, dis_resume_file)
    torch.save(
        {'model_state_dict': judger.state_dict(),
         'scheduler': judge_scheduler,
         'optimizer': judge_optimizer.state_dict()
         }, judge_resume_file)
    print("save the current result to '{}'".format(result_file))
    all_result_df.to_csv(result_file, index=False, header=True)

print('=' * 89)
print('End of training and evaluation')
