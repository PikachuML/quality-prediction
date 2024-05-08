import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

N = 9


def data_norm():
    current_path = os.path.dirname(__file__)

    YQ_data = pd.read_csv(current_path + '/data/yarn quality.csv', header=0, encoding='utf-8')
    YC_data = pd.read_csv(current_path + '/data/yarn count_param.csv', header=0, encoding='utf-8')

    data = YC_data.iloc[:, 1:-1].to_numpy(dtype=float)
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    YC_data_norm = pd.concat([YC_data.iloc[:, 0], pd.DataFrame(data_norm), YC_data.iloc[:, -1]], axis=1)

    data2 = YQ_data.iloc[:, 1:-1].to_numpy(dtype=float)
    data2_norm = scaler.fit_transform(data2)
    YQ_data_norm = pd.concat([YQ_data.iloc[:, 0], pd.DataFrame(data2_norm), YQ_data.iloc[:, -1]], axis=1)

    cotton = pd.read_csv(current_path + '/data/cotton.csv', header=0)

    cotton1 = cotton.iloc[:, 2:]
    cotton_norm = scaler.fit_transform(cotton1.iloc[:, :-1])
    p = cotton1.iloc[:, -1] / 100

    cotton_table = pd.concat([cotton.iloc[:, 0:2], pd.DataFrame(cotton_norm,
                                                                columns=['CQ_{}'.format(i) for i in range(1, N+1)]),
                              pd.DataFrame(p, columns=['prop'])], axis=1)

    # grouped = cotton_table.groupby(by="table_nr")
    # grouped2 = dict(list(grouped))
    #
    # cotton_table = pd.DataFrame()
    # for i in range(len(YC_data.iloc[:, -1])):
    #     c = grouped2[int(YC_data.iloc[i, -1])]
    #     c['new_nur'] = i
    #     cotton_table = cotton_table.append(c, ignore_index=True)

    return cotton_table, YC_data_norm, YQ_data_norm
