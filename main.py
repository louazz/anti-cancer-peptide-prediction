import itertools

import numpy as np
import torch
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.NeuralNetwork import NeuralNetwork

import pandas as pd
from model.TorchDataset import DatasetTorch


class AnticancerPeptide:
    def __init__(self):
        self.data = pd.read_csv('/home/louai/Documents/anticancer_peptide/dataset/alternate_AntiCP/descriptors.csv')
        self.scaler = MinMaxScaler()
        # self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf = SVC(kernel='rbf', probability=True)
        self.nn = NeuralNetwork(88, 1)
        self.writer = SummaryWriter()
        # consider using knn with 3 and 4 neighbors
        # self.clf = RandomForestClassifier()
        self.rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=50)
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(88, 44),
            torch.nn.BatchNorm1d(44),
            torch.nn.ELU(),

            torch.nn.Linear(44, 22),
            torch.nn.BatchNorm1d(22),
            torch.nn.ELU(),

            torch.nn.Dropout(0.2),

            torch.nn.Linear(22, 1),
            torch.nn.Sigmoid()
        )

    def train_neural_network(self):
        ds = self.data
        ds = ds.drop(columns=['id'])
        # ds = ds.dropna()

        kf = KFold(n_splits=5, shuffle=True)
        n_epochs = 300
        loss_values = []

        true_val = []
        pred_val = []

        sampler = RandomUnderSampler(sampling_strategy='majority')

        j = 0

        for train_idx, test_idx in kf.split(ds):
            j += 1
            optim = torch.optim.Adam(self.nn_model.parameters(), lr=1e-4)
            loss_fn = torch.nn.BCELoss()

            train_set = ds.iloc[train_idx, :]
            test_set = ds.iloc[test_idx, :]

            X_train = train_set.drop(columns=['class'])
            X_test = test_set.drop(columns=['class'])

            y_train = train_set['class']
            y_test = test_set['class']

            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(y_train.value_counts())

            self.scaler.fit(X_train, y_train)

            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)

            train_data = DatasetTorch(X_train, y_train.to_numpy())
            test_data = DatasetTorch(X_test, y_test.to_numpy())

            train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
            test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

            i = 0
            running_loss = 0.0
            for epoch in range(n_epochs):
                i += 1
                print('Epoch number: {}'.format(i))

                k = 0
                for X, y in train_loader:
                    k += 1
                    optim.zero_grad()

                    pred = self.nn_model(X)
                    loss = loss_fn(pred, y.unsqueeze(1))

                    loss_values.append(loss.item())
                    loss.backward()
                    optim.step()
                    running_loss += loss.item()
                    self.writer.add_scalars('run_split_{}'.format(j),
                                                {'training loss': running_loss},
                                                epoch)

            print('Completed training')

            y_true = []
            y_pred = []

            total = 0.0
            correct = 0.0

            with torch.no_grad():
                for X, y in test_loader:
                    out = self.nn_model(X)
                    pred = np.where(out < 0.5, 0, 1)
                    pred = list(itertools.chain(*pred))
                    y_pred.extend(pred)
                    y_true.extend(y.data.cpu().numpy())
                    total += y.size(0)
                    correct += (pred == y.numpy().sum().item())

            report = classification_report(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            print(report)
            print(cm)

            true_val.extend(y_true)
            pred_val.extend(y_pred)
        print('-' * 10)
        report = classification_report(true_val, pred_val)
        print(report)
        cm = confusion_matrix(true_val, pred_val)
        print(cm)
        fpr, tpr, _ = roc_curve(true_val, pred_val)
        print(tpr)
        print(fpr)
        roc_auc = roc_auc_score(true_val, pred_val)
        print(roc_auc)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display.plot()
        plt.show()

    def cross_val_train(self):
        ds = self.data
        ds = ds.drop(columns=['id'])

        # ds = ds.dropna()

        kf = KFold(n_splits=5, shuffle=True)

        true = np.array([])
        pred = np.array([])
        proba = []

        sampler = RandomUnderSampler(sampling_strategy="majority")

        for train_idx, test_idx in kf.split(ds):
            train_set = ds.iloc[train_idx, :]
            test_set = ds.iloc[test_idx, :]

            X_train = train_set.drop(columns=["class"])
            X_test = test_set.drop(columns=["class"])

            y_train = train_set["class"]
            y_test = test_set["class"]

            # balance only the training set and test on the whole ds
            X_train, y_train = sampler.fit_resample(X_train, y_train)

            print(y_train.value_counts())

            self.scaler.fit(X_train)

            scaled_train = self.scaler.transform(X_train)
            scaled_test = self.scaler.transform(X_test)

            self.rfe.fit(scaled_train, y_train)

            reduced_train = self.rfe.transform(scaled_train)
            reduced_test = self.rfe.transform(scaled_test)

            self.clf.fit(reduced_train, y_train)

            y_pred = self.clf.predict(reduced_test)
            prob = self.clf.predict_proba(reduced_test)

            y_test = y_test.to_numpy()

            pred = np.concatenate((pred, y_pred), axis=0)
            true = np.concatenate((true, y_test), axis=0)
            proba.extend(prob[:, 1])

            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print("The confusion matrix of the RF model is")
            print(cm)
            print(report)

        print("-" * 10)

        cm = confusion_matrix(true, pred)
        report = classification_report(true, pred)

        print("The confusion matrix of the RF model is")
        print(cm)
        print(report)
        fpr, tpr, _ = roc_curve(true, pred)
        print(tpr)
        print(fpr)
        roc_auc = roc_auc_score(true, pred)
        print(roc_auc)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display.plot()
        plt.show()


if __name__ == '__main__':
    acp = AnticancerPeptide()
    acp.cross_val_train()
    acp.writer.flush()
    acp.writer.close()
