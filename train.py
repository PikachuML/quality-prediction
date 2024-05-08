from model import *
from data_norm import *
from testing import *
from torch.utils.data import Dataset, DataLoader
import os
import torch
import time
import random

torch.manual_seed(1)
torch.cuda.manual_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed+1)
np.random.seed(seed+2)


class MyDataset(Dataset):
    def __init__(self, features1, features2, labels):
        self.features1 = features1
        self.features2 = features2
        self.labels = labels

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, idx):
        return self.features1[idx], self.features2[idx], self.labels[idx]


def my_collate(batch):
    data_cotton = [cotton[0] for cotton in batch]
    max_len = max((len(l) for l in data_cotton))

    features1 = torch.Tensor([]).cuda()
    prop = torch.Tensor([]).cuda()
    features2 = torch.Tensor([]).cuda()
    labels = torch.Tensor([]).cuda()
    this_batch_size = len(batch)

    for i in range(this_batch_size):
        l1 = len(data_cotton[i])
        pad = nn.ZeroPad2d(padding=(0, 0, 0, (max_len-l1)))
        y = pad(data_cotton[i])
        y1 = y[:, :-1]
        y2 = y[:, -1]

        features1 = torch.cat((features1, y1.view(1, max_len, y1.shape[1])), 0)
        prop = torch.cat((prop, y2.view(1, max_len)), 0)
        features2 = torch.cat((features2, batch[i][1].view(1, batch[i][1].shape[0])), 0)
        labels = torch.cat((labels, batch[i][2].view(1, batch[i][2].shape[0])), 0)

    return features1, prop, features2, labels


def get_cotton_table_list(cotton_table):
    cotton_group = cotton_table.groupby(by='table_nr')
    i = 1
    X_train = []
    while i < len(cotton_group):
        x_cotton = torch.Tensor(cotton_group.get_group(i).drop(['table_nr', 'cotton_idx'], axis=1).values)
        X_train.append(x_cotton.cuda())
        i += 1
    return X_train


def training():
    X_train = get_cotton_table_list(cotton_table_train)
    X1_train = torch.tensor(YC_data_train.iloc[:, 1:-1].values, dtype=torch.float32).cuda()
    y_train = torch.tensor(YQ_data_train.iloc[:, 1:-1].values, dtype=torch.float32).cuda()

    train_dataset = MyDataset(X_train, X1_train, y_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
                                                           patience=250, factor=0.1)

    eval_train = eval_class(cotton_table_train, YC_data_train, YQ_data_train)
    eval_test = eval_class(cotton_table_test, YC_data_test, YQ_data_test)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, prop, x1, y in train_dataloader:
            outputs = model.forward(x, prop, x1)
            loss = loss_func1(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()

        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss, _, _, _, _, _ = eval_train.testing(model)
                mse_test, score, mae, rmse, mape, prediction = eval_test.testing(model)

            print("Epoch: %d, training loss: %1.5f, testing mse: %1.5f, score: %1.5f" % (epoch, train_loss, mse_test,
                                                                                         score))

        model.train()
        scheduler.step(metrics=mse_test)

    model.eval()
    print(score, mae, rmse, mape)

    return mse_test, prediction


if __name__ == "__main__":
    num_epochs = 1000
    d_model = 128
    heads = 2
    N = 2

    learning_rate = 5e-3
    weight_decay = 1e-4
    batch_size = 4
    dropout = 0.1

    m = 9  # number of cotton features
    d_yc = 4  # number of yarn count
    d_y = 8   # number of yarn quality

    loss_func1 = torch.nn.L1Loss()

    cotton_table_train, YC_data_train, YQ_data_train = data_norm()
    cotton_table_test, YC_data_test, YQ_data_test = data_norm()
    print("num_epochs:", num_epochs,
          "d_model:", d_model,
          "heads:", heads,
          "N:", N,
          "lr:", learning_rate,
          "wd:", weight_decay,
          "bs:", batch_size,
          "dropout:", dropout,
          "loss: MAE"
          )

    # define and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Blendmapping(d_model, d_yc, d_y, N, heads, m, dropout)

    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)

    _, prediction = training()
    np.savetxt("prediction.txt", prediction, delimiter=",")
    for e in range(prediction.shape[1]):
        loss = loss_func1(torch.tensor(YQ_data_test.iloc[:, e+1].values), prediction[:, e])
        score = r2_score(YQ_data_test.iloc[:, e+1].values, prediction[:, e].numpy())
        mae = mean_absolute_error(YQ_data_test.iloc[:, e + 1].values, prediction[:, e].numpy())
        rmse = mean_squared_error(YQ_data_test.iloc[:, e + 1].values, prediction[:, e].numpy(), squared=False)
        mape = mean_absolute_percentage_error(prediction[:, e].numpy(), YQ_data_test.iloc[:, e + 1].values)

        print('label:', e + 1)
        print('loss:', loss, 'score:', score, 'mae', mae, 'rmse', rmse, 'mape', mape)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
