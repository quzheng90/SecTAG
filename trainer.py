# Built-in libraries
import copy
import datetime
from tkinter import W
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import f1_score
from tqdm import tqdm
# Local files
from utils import save
from config import LABEL_DICT
from transformers import BertModel
from lstmclassifer import LSTMClassifier



class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''
    def __init__(
        self,
        # model: nn.Module,
        model:LSTMClassifier(768,100,3,2),
        epochs: int,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        loss_weights: List[float],
        clip: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        patience: int,
        task_name: str,
        model_name: str,
        seed: int
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.task_name = task_name
        self.model_name = model_name
        self.seed = seed
        self.datetimestr = datetime.datetime.now().strftime('%Y_%b_%d_%H%M%S')

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0

        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([0, 0, 0], dtype=np.float64)
        self.best_test_f1_m = np.array([0, 0, 0], dtype=np.float64)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch()
            self.test()
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1, self.best_test_f1),
            f'./save/results/single_{self.task_name}_{self.datetimestr}_.pt'
        )

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Training'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                loss += _loss.item()
                y_pred = logits.argmax(dim=1).cpu().numpy()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        f1 = metrics.f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        if f1 > self.best_train_f1:
            self.best_train_f1 = f1

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.save_model()

    def train_m_hijack(self):
        print("start time:{}".format(datetime.datetime.now()))
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch_mtl_hijack()
            self.test_m()
            print(f'Best test results A: {self.best_test_f1_m[0]:.4f}')
            print(f'Best test results B: {self.best_test_f1_m[1]:.4f}')
            print(f'Best test results C: {self.best_test_f1_m[2]:.4f}')
            print('=' * 20)
        print("end time:{}".format(datetime.datetime.now()))
        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1_m, self.best_test_f1_m),
            f'E:/quz/hashtag_hijack/save/results/{self.datetimestr}.pt'
        )

    def train_one_epoch_mtl_hijack(self):
        self.model.train()
        dataloader = self.dataloaders['train']

        loss = 0
        iters_per_epoch = 0
        label_A=None
        label_B=None
        label_C=None
        y_pred_A=None
        y_pred_B=None
        y_pred_C=None
        #initial sample bias
        loss_pre_a=0.1
        loss_pre_b=0.4
        loss_pre_c=0.3
        w1=0.1
        w2=0.4
        w3=0.3
        for inputs, lens, mask, label_A, label_B, label_C in tqdm(dataloader, desc='Training M'):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
                all_logits = self.model(inputs, lens, mask)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1][:, 0:2].argmax(dim=1)
                y_pred_C = all_logits[2][:, 0:3].argmax(dim=1)

                # label_A_l.append(label_A)
                # label_B_l.append(label_B)
                # label_C_l.append(label_C)

                # y_pred_A_l.append(y_pred_A)
                # y_pred_B_l.append(y_pred_B)
                # y_pred_C_l.append(y_pred_C)                

                ##dynamic loss 
                x1=self.criterion(all_logits[0], label_A)
                x2=self.criterion(all_logits[1], label_B)
                x3=self.criterion(all_logits[2], label_C)
                w1_=self.loss_weights[0]*x1/loss_pre_a
                w2_=self.loss_weights[1]*x2/loss_pre_b
                w3_=self.loss_weights[2]*x3/loss_pre_c
                w_=w1_+w2_+w3_
                w1=w1_/w_

                w2=w2_/w_
                w3=w3_/w_
                # loss_pre_a=w1*x1
                # loss_pre_b=w2*x2
                # loss_pre_c=w3*x3
                # _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                # _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                # _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)
                # loss += _loss.item()
                _loss1 = w1 * x1
                _loss2 = w2 * x2
                _loss3 = w3 * x3
                _loss=_loss1+_loss2+_loss3
                #save pre loss
                loss_pre_a=_loss1.detach()
                loss_pre_b=_loss2.detach()
                loss_pre_c=_loss3.detach()
                loss += _loss.item()

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        #all_logits[0]=all_logits[0].detach().numpy()
        A_accuracy = metrics.accuracy_score(label_A, y_pred_A)
        A_f1 = metrics.f1_score(label_A, y_pred_A, average='macro')
        A_precision = metrics.precision_score(label_A, y_pred_A, average='macro')
        A_recall = metrics.recall_score(label_A, y_pred_A, average='macro')

        B_accuracy = metrics.accuracy_score(y_pred_B, y_pred_B)
        B_f1 = metrics.f1_score(y_pred_B, y_pred_B, average='macro')
        B_precision = metrics.precision_score(y_pred_B, y_pred_B, average='macro')
        B_recall = metrics.recall_score(y_pred_B, y_pred_B, average='macro')

        C_accuracy = metrics.accuracy_score(label_C, y_pred_C)
        C_f1 = metrics.f1_score(label_C, y_pred_C, average='macro')
        C_precision = metrics.precision_score(label_C, y_pred_C, average='macro')
        C_recall = metrics.recall_score(label_C, y_pred_C, average='macro')
        print(f'loss = {loss:.4f}')
        # print(f'A_accuracy: {A_accuracy:.4f}',f'A_recall: {A_recall:.4f}', f'A_precision: {A_precision:.4f}')
        # print(f'B_accuracy: {B_accuracy:.4f}', f'B_recall: {B_recall:.4f}', f'B_precision: {B_precision:.4f}')
        # print(f'C_accuracy: {C_accuracy:.4f}', f'C_recall: {C_recall:.4f}', f'C_precision: {C_precision:.4f}')


    def train_m(self):
        print("start time:{}".format(datetime.datetime.now()))
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch_m()
            self.test_m()
            print(f'Best test results A: {self.best_test_f1_m[0]:.4f}')
            print(f'Best test results B: {self.best_test_f1_m[1]:.4f}')
            print(f'Best test results C: {self.best_test_f1_m[2]:.4f}')
            print('=' * 20)
        print("end time:{}".format(datetime.datetime.now()))
        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1_m, self.best_test_f1_m),
            f'./save/results/mtl_{self.datetimestr}_.pt'
        )

    def train_one_epoch_m(self):
        self.model.train()
        dataloader = self.dataloaders['train']

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, label_A, label_B, label_C in tqdm(dataloader, desc='Training M'):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
                all_logits = self.model(inputs, lens, mask)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1][:, 0:2].argmax(dim=1)
                y_pred_C = all_logits[2][:, 0:3].argmax(dim=1)

                Non_null_index_B = label_B != LABEL_DICT['b']['NULL']
                Non_null_label_B = label_B[Non_null_index_B]
                Non_null_pred_B = y_pred_B[Non_null_index_B]

                Non_null_index_C = label_C != LABEL_DICT['c']['NULL']
                Non_null_label_C = label_C[Non_null_index_C]
                Non_null_pred_C = y_pred_C[Non_null_index_C]

                labels_all_A = label_A.cpu().numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.cpu().numpy()))
                labels_all_B = Non_null_label_B.cpu().numpy() if labels_all_B is None else np.concatenate((labels_all_B, Non_null_label_B.cpu().numpy()))
                labels_all_C = Non_null_label_C.cpu().numpy() if labels_all_C is None else np.concatenate((labels_all_C, Non_null_label_C.cpu().numpy()))

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = Non_null_pred_B.cpu().numpy() if y_pred_all_B is None else np.concatenate((y_pred_all_B, Non_null_pred_B.cpu().numpy()))
                y_pred_all_C = Non_null_pred_C.cpu().numpy() if y_pred_all_C is None else np.concatenate((y_pred_all_C, Non_null_pred_C.cpu().numpy()))

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(Non_null_label_B, Non_null_pred_B)
                # f1[2] += self.calc_f1(Non_null_label_C, Non_null_pred_C)

                _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)
                loss += _loss.item()

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        # f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        # f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        # f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        A_accuracy = metrics.accuracy_score(labels_all_A, y_pred_all_A)
        A_f1 = metrics.f1_score(labels_all_A, y_pred_all_A, average='macro')
        A_precision = metrics.precision_score(labels_all_A, y_pred_all_A, average='macro')
        A_recall = metrics.recall_score(labels_all_A, y_pred_all_A, average='macro')

        B_accuracy = metrics.accuracy_score(labels_all_B, y_pred_all_B)
        B_f1 = metrics.f1_score(labels_all_B, y_pred_all_B, average='macro')
        B_precision = metrics.precision_score(labels_all_B, y_pred_all_B, average='macro')
        B_recall = metrics.recall_score(labels_all_B, y_pred_all_B, average='macro')

        C_accuracy = metrics.accuracy_score(labels_all_C, y_pred_all_C)
        C_f1 = metrics.f1_score(labels_all_C, y_pred_all_C, average='macro')
        C_precision = metrics.precision_score(labels_all_C, y_pred_all_C, average='macro')
        C_recall = metrics.recall_score(labels_all_C, y_pred_all_C, average='macro')
        print(f'loss = {loss:.4f}')
        print(f'A_accuracy: {A_accuracy:.4f}')
        print(f'A_recall: {A_recall:.4f}')
        print(f'A_precision: {A_precision:.4f}')

        print(f'loss = {loss:.4f}')
        print(f'B_accuracy: {B_accuracy:.4f}')
        print(f'B_recall: {B_recall:.4f}')
        print(f'B_precision: {B_precision:.4f}')

        print(f'loss = {loss:.4f}')
        print(f'C_accuracy: {C_accuracy:.4f}')
        print(f'C_recall: {C_recall:.4f}')
        print(f'C_precision: {C_precision:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append([A_f1, B_f1, C_f1])

        
        
        
        if A_f1 > self.best_train_f1_m[0]:
            self.best_train_f1_m[0] = A_f1
        if B_f1 > self.best_train_f1_m[1]:
            self.best_train_f1_m[1] = B_f1
        if C_f1 > self.best_train_f1_m[2]:
            self.best_train_f1_m[2] = C_f1
    
    


    def test_m(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        loss = 0
        iters_per_epoch = 0

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None

        for inputs, lens, mask, label_A, label_B, label_C in tqdm(dataloader, desc='Test M'):
            iters_per_epoch += 1

            labels_all_A = label_A.numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.numpy()))
            labels_all_B = label_B.numpy() if labels_all_B is None else np.concatenate((labels_all_B, label_B.numpy()))
            labels_all_C = label_C.numpy() if labels_all_C is None else np.concatenate((labels_all_C, label_C.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            with torch.set_grad_enabled(False):
                all_logits = self.model(inputs, lens, mask)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1].argmax(dim=1).cpu().numpy()
                y_pred_C = all_logits[2].argmax(dim=1).cpu().numpy()

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(label_B, y_pred_B)
                # f1[2] += self.calc_f1(label_C, y_pred_C)

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = y_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, y_pred_B))
                y_pred_all_C = y_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, y_pred_C))

                _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)
                loss += _loss.item()

        loss /= iters_per_epoch
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        print(f'loss = {loss:.4f}')
        # print(f'A: {f1_A:.4f}')
        # print(f'B: {f1_B:.4f}')
        # print(f'C: {f1_C:.4f}')

        A_accuracy = metrics.accuracy_score(labels_all_A, y_pred_all_A)
        A_f1 = metrics.f1_score(labels_all_A, y_pred_all_A, average='macro')
        A_precision = metrics.precision_score(labels_all_A, y_pred_all_A, average='macro')
        A_recall = metrics.recall_score(labels_all_A, y_pred_all_A, average='macro')

        B_accuracy = metrics.accuracy_score(labels_all_B, y_pred_all_B)
        B_f1 = metrics.f1_score(labels_all_B, y_pred_all_B, average='macro')
        B_precision = metrics.precision_score(labels_all_B, y_pred_all_B, average='macro')
        B_recall = metrics.recall_score(labels_all_B, y_pred_all_B, average='macro')

        C_accuracy = metrics.accuracy_score(labels_all_C, y_pred_all_C)
        C_f1 = metrics.f1_score(labels_all_C, y_pred_all_C, average='macro')
        C_precision = metrics.precision_score(labels_all_C, y_pred_all_C, average='macro')
        C_recall = metrics.recall_score(labels_all_C, y_pred_all_C, average='macro')


        print(f'A_accuracy: {A_accuracy:.4f}',f'A_recall: {A_recall:.4f}', f'A_precision: {A_precision:.4f}')
        print(f'B_accuracy: {B_accuracy:.4f}', f'B_recall: {B_recall:.4f}', f'B_precision: {B_precision:.4f}')
        print(f'C_accuracy: {C_accuracy:.4f}', f'C_recall: {C_recall:.4f}', f'C_precision: {C_precision:.4f}')


        self.test_losses.append(loss)
        self.test_f1.append([f1_A, f1_B, f1_C])

        if f1_A > self.best_test_f1_m[0]:
            self.best_test_f1_m[0] = f1_A
            self.save_model()
        if f1_B > self.best_test_f1_m[1]:
            self.best_test_f1_m[1] = f1_B
        if f1_C > self.best_test_f1_m[2]:
            self.best_test_f1_m[2] = f1_C

        # for i in range(len(f1)):
        #     for j in range(len(f1[0])):
        #         if f1[i][j] > self.best_test_f1_m[i][j]:
        #             self.best_test_f1_m[i][j] = f1[i][j]
        #             if i == 0 and j == 0:
        #                 self.save_model()

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def printing(self, loss, f1):
        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1[0]:.4f}')
        # print(f'Micro-F1 = {f1[1]:.4f}')
        # print(f'Weighted-F1 = {f1[2]:.4f}')

    def save_model(self):
        print('Saving model...')
        if self.task_name == 'all':
            filename = f'./save/models/{self.task_name}_{self.model_name}_seed{self.seed}.pt'
        else:
            filename = f'./save/models/{self.task_name}_{self.model_name}_seed{self.seed}.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)
