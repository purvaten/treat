from arguments import parser

import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SubsetRandomSampler
from models import MyData

from models import MultiTask
import sys
import os.path

from logger import get_logger
logger = get_logger(__name__)


IMG_SIZE = 224

args = parser.parse_args()
args.data = args.data.split(',')
args.data.sort()

with open('args/%s_model_%s_alpha_%.2f_datalimit_%d.pickle' % (args.id, args.model, args.alpha, args.datalimit), 'wb') as f:
    pickle.dump(args, f)

logger.info(args)

# Initialize train dataset
logger.info("Starting Loading data")
dataset = MyData(args, img_size=IMG_SIZE)

dataset_len = len(dataset)
logger.info("Dataset obtained successfully")

# Split data as train and test
logger.info("Starting data splitting")

# If indices are already saved then load from file directly else peform split once and save
tv_split = int(np.floor(0.1 * len(dataset)))
split_name = "%s-%d-%d" % ("-".join(args.data), args.datalimit, len(dataset))
if os.path.isfile('splits/split-%s.pkl' % split_name):
    logger.info("Using saved split splits/split-%d.pkl" % len(dataset))
    with open('splits/split-%s.pkl' % split_name, 'rb') as f:
        myindices = pickle.load(f)
    train_idx, val_idx = myindices
else:
    logger.info("Splitting data for the first time")
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[tv_split:], indices[:tv_split]
    myindices = [train_idx, val_idx]
    with open('splits/split-%s.pkl' % split_name, 'wb') as f:
        pickle.dump(myindices, f)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

dataloader_train = DataLoader(dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=1)
dataloader_val = DataLoader(dataset, batch_size=args.batchsize, sampler=val_sampler, num_workers=1)

logger.info("Data split complete. Data loaded successfully")


learning_rate = 0.0001
num_epochs = 100

model = MultiTask(args)

if torch.cuda.is_available():
    logger.info("Using GPU")
    model.cuda()

rcriterion = torch.nn.MSELoss(reduction='none')
ccriterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

loss_history_train, loss_history_val = [], []
best_loss = sys.maxsize
e = 0
for epoch in range(num_epochs):
    # ===================train========================
    model.train()
    loss_total_train = lr_total_train = lc_total_train = 0
    total_letters = 0
    for data in dataloader_train:
        img, labels, weights, identity = data
        img = img.float().cuda()
        labels = labels.long().cuda()
        weights = weights.float().cuda()
        identity = identity.cuda()

        # Markers for letters which are used for classification loss
        letter_flags = (identity == 2).float()
        total_letters += letter_flags.sum()
        # ===================forward=====================
        routput, coutput, _ = model(img)

        lr = rcriterion(routput, img)
        # averaging across pixels + channels
        lr = lr.view(-1, IMG_SIZE * IMG_SIZE * 3).mean(-1)

        lc = ccriterion(coutput, labels)

        loss = weights * lr + (1 - weights) * lc
        loss = torch.sum(loss)

        lr_total_train += lr.sum().data
        lc_total_train += (letter_flags * lc).sum().data
        loss_total_train += loss.data
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================delete variables====================
        del loss, routput, coutput

    lr_train = lr_total_train / (dataset_len - tv_split)
    if total_letters == 0:
        lc_train = 0
    else:
        lc_train = lc_total_train / total_letters
    loss_train = loss_total_train / (dataset_len - tv_split)
    del lr_total_train, lc_total_train, loss_total_train

    # ===================val========================
    with torch.no_grad():
        model.eval()
        loss_total_val = lr_total_val = lc_total_val = 0
        total_letters = 0
        for data in dataloader_val:
            img, labels, weights, identity = data
            img = img.float().cuda()
            labels = labels.long().cuda()
            weights = weights.float().cuda()
            identity = identity.cuda()

            # Markers for letters which are used for classification loss
            letter_flags = (identity == 2).float()
            total_letters += letter_flags.sum()
            # ===================forward=====================
            routput, coutput, _ = model(img)
            lr = rcriterion(routput, img)
            # averaging across pixels + channels
            lr = lr.view(-1, IMG_SIZE * IMG_SIZE * 3).mean(-1)
            lc = ccriterion(coutput, labels)
            loss = weights * lr + (1 - weights) * lc
            loss = torch.sum(loss)

            lr_total_val += lr.sum().data
            lc_total_val += (letter_flags * lc).sum().data
            loss_total_val += loss.data
            # ===================delete variables====================
            del loss, routput, coutput

        lr_val = lr_total_val / tv_split
        if total_letters == 0:
            lc_val = 0
        else:
            lc_val = lc_total_val / total_letters
        loss_val = loss_total_val / tv_split
        del lr_total_val, lc_total_val, loss_total_val
    # ===================log========================
    output = 'epoch [{}/{}], train loss:{:.4f}, lr:{:.4f}, lc:{:.4f}, val loss:{:.4f}, lr:{:.4f}, lc:{:.4f}'.format(
        epoch + 1, num_epochs, loss_train, lr_train, lc_train, loss_val, lr_val, lc_val
    )
    logger.info(output)

    loss_history_train.append(loss_train)
    loss_history_val.append(loss_val)

    if loss_val < best_loss:
        save_path = 'save/%s_model_%s_alpha_%.2f_datalimit_%d.pth' % (args.id, args.model, args.alpha, args.datalimit)
        logger.info("Saving best model at epoch %d at %s ..." % (epoch, save_path))
        best_loss = loss_val
        torch.save(model.state_dict(), save_path)
        e = epoch
    del loss_train, loss_val
