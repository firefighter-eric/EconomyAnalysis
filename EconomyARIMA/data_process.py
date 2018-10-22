import numpy as np
import pandas as pd
import copy


class DataStructure:
    def __init__(self):
        self.num = None
        self.batch_num = None
        self.index = None
        self.raw = None
        self.batch = None
        self.random = None
        self.input = None


class DataXY:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def func_xy(self, func):
        self.x = func(self.x)
        self.y = func(self.y)


class Data:
    def __init__(self, filename, label_col, tag, time_step):
        # property
        self.filename = filename
        self.label_col = label_col
        self.tag = tag
        self.time_step = time_step
        # data_raw
        self.data = DataStructure()
        self.data.raw = self.read_file(self.filename, self.label_col, self.tag)
        self.data.num = len(self.data.raw)
        self.train = self.init_data_train()
        self.valid = self.init_data_valid()
        self.predict = self.init_data_predict()

    # train
    def init_data_train(self):
        train = DataStructure()
        # raw
        train.num = int(0.8 * self.data.num)
        train.raw = DataXY(self.data.raw[:train.num - 1], self.data.raw[1:train.num])
        # batch
        train.batch_num = len(train.raw.x) - self.time_step + 1
        train.batch = DataXY(self.batch_data(train.raw.x, train.batch_num, self.time_step),
                             self.batch_data(train.raw.y, train.batch_num, self.time_step))
        # random
        train.random = DataXY()
        train.random.x, train.random.y = self.batch_random_xy(train.batch.x, train.batch.y)
        # input
        train.input = DataXY(self.input_data(train.random.x), self.input_data(train.random.y))
        return train

    # valid
    def init_data_valid(self):
        valid = DataStructure()
        # raw
        valid.num = self.data.num - self.train.num
        valid.raw = DataXY(self.data.raw[self.train.num: - 1], self.data.raw[self.train.num + 1:])
        # batch
        valid.batch_num = len(valid.raw.x) - self.time_step + 1
        valid.batch = DataXY(self.batch_data(valid.raw.x, valid.batch_num, self.time_step),
                             self.batch_data(valid.raw.y, valid.batch_num, self.time_step))
        # input
        valid.input = DataXY(self.input_data(valid.batch.x), self.input_data(valid.batch.y))
        return valid

    # prediction
    def init_data_predict(self):
        prediction = DataStructure()
        # raw
        prediction.raw = DataXY(copy.deepcopy(self.data.raw[-self.time_step:]), None)
        # input
        prediction.input = DataXY(self.input_data(prediction.raw.x), None)
        return prediction

    def roll_predict(self, _next):
        self.predict.raw.x = np.hstack((self.predict.raw.x, np.array([_next])))
        self.predict.input.x = self.input_data(self.predict.raw.x[-self.time_step:])

    def read_file(self, filename, label_col, tag):
        df = pd.read_csv(filename, index_col=label_col)
        data = df.loc[:, tag]
        data = np.array(data[data.notnull()])
        return data

    def batch_data(self, data, batch_num, time_step):
        out = np.zeros((batch_num, time_step))
        for i in range(batch_num):
            out[i, :] = data[i: i + time_step]
        return out

    def batch_random_xy(self, data_x, data_y):
        data_shape_x = np.shape(data_x)
        data_shape_y = np.shape(data_y)
        n_shuffle = np.arange(data_shape_x[0])
        np.random.shuffle(n_shuffle)
        out_x = np.zeros(data_shape_x)
        out_y = np.zeros(data_shape_y)
        for i in range(len(n_shuffle)):
            out_x[i, :] = data_x[n_shuffle[i], :]
            out_y[i, :] = data_y[n_shuffle[i], :]
        return out_x, out_y

    def input_data(self, data):
        shape = np.shape(data)
        if len(shape) == 1:
            return data.reshape((1, shape[0], 1))
        else:
            return data.reshape((shape[0], shape[1], 1))


# test
# if __name__ == '__main__':
#     filename = "outdata201803_hz_monthly_1.csv"
#     index_col = 'Dates'
#     tag = 'CPI'
#     time_step = 10
#
#     data = Data(filename, index_col, tag, time_step)
