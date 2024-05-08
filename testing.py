import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


class eval_class():
    def __init__(self, cotton_table, YC_data, YQ_data):
        self.sample_num = len(YQ_data)
        cotton_group = cotton_table.groupby(by='table_nr')
        self.X_test_tensors = []
        self.prop_test = []
        self.X1_test_tensors = []
        for j in range(self.sample_num):
            x_test = cotton_group.get_group(j+1).drop(['table_nr', 'cotton_idx'], axis=1).to_numpy()
            x1_test = YC_data.iloc[j, 1:-1]

            X_test_tensors = torch.Tensor(x_test[:, :-1]).cuda()
            prop_test = torch.Tensor(x_test[:, -1]).cuda()
            X1_test_tensors = torch.tensor(x1_test, dtype=torch.float32).cuda()

            self.X_test_tensors.append(X_test_tensors.view(1, X_test_tensors.shape[0], X_test_tensors.shape[1]))
            self.prop_test.append(prop_test.view(1, prop_test.shape[0]))
            self.X1_test_tensors.append(X1_test_tensors.view(1, X1_test_tensors.shape[0]))
            self.y_test = YQ_data.iloc[:, 1:-1].values.astype(np.float32)
        self.loss_function = torch.nn.MSELoss()
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

    def testing(self, model):
        result = torch.Tensor([])
        for j in range(self.sample_num):
            test_predict = model.forward(self.X_test_tensors[j],
                                         self.prop_test[j],
                                         self.X1_test_tensors[j])
            result = torch.cat((result, test_predict.cpu()), 0)
        mse = self.loss_function(self.y_test_tensor, result)
        score = r2_score(self.y_test, result.numpy())
        mae = mean_absolute_error(self.y_test, result.numpy())
        rmse = mean_squared_error(self.y_test, result.numpy(), squared=False)
        mape = mean_absolute_percentage_error(result.numpy(), self.y_test)
        return mse, score, mae, rmse, mape, result
