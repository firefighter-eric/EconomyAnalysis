import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def read_file():
    f = open("outdata201803_hz_annual_1.csv")
    f.readline()
    data = list()
    for line in f:
        line = line.strip().split(',')
        for i in range(len(line)):
            if line[i] == 'NaN':
                line[i] = 0;
            else:
                line[i] = float(line[i])
        data.append(line)
    f.close()
    return np.array(data)


def select_data_range(data, column, start, end):
    data_temp = data[start:end, column]
    data_x = data_temp[:-1]
    data_y = data_temp[1:]
    return data_x, data_y


def batch_data(data, batch_num, time_step):
    out = np.zeros((batch_num, time_step))
    for i in range(batch_num):
        out[i, :] = data[i: i + time_step]
    return out


def batch_random(data_x, data_y):
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


data = read_file()
data_x, data_y = select_data_range(data, 1, 29, 68)
# data properties
data_num = len(data_x)
feature_num = 1
label_num = 1

# model parameter
train_rate = 0.001
train_step = 10000
display_step = 500

input_size = feature_num
output_size = label_num
time_step = 10
batch_num = data_num - time_step + 1
train_batch_num = int(batch_num * 0.8)
valid_batch_num = batch_num - train_batch_num
hidden_num = 5

x_batch = batch_data(data_x, batch_num, time_step)
y_batch = batch_data(data_y, batch_num, time_step)

x_, y_ = batch_random(x_batch, y_batch)

x_train = x_[:train_batch_num, :]
y_train = y_[:train_batch_num, :]
x_valid = x_[train_batch_num:, :]
y_valid = y_[train_batch_num:, :]


def input_data(data):
    shape = np.shape(data)
    return data.reshape((shape[0], shape[1], 1))


x = tf.placeholder(dtype=tf.float32, shape=[None, time_step, input_size], name="input_x")
y = tf.placeholder(dtype=tf.float32, shape=[None, time_step, output_size], name="expected_y")

weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, output_size]))
bias = tf.Variable(tf.zeros(shape=[output_size]))

batch_size = tf.shape(x)[0]
inputs = tf.reshape(x, shape=[batch_size, time_step, input_size])
cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
initial_state = tf.zeros([batch_size, cell.state_size])

outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)

outputs = tf.reshape(outputs, [-1, hidden_num])
logits = tf.nn.relu(tf.matmul(outputs, weights) + bias)
predictions = tf.reshape(logits, [-1, output_size])

label = tf.reshape(y, (-1, output_size))
cost = tf.losses.mean_squared_error(label, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    iteration = 0
    for i in range(train_step):
        _, loss = sess.run([optimizer, cost], feed_dict={x: input_data(x_train), y: input_data(y_train)})
        if iteration % display_step == 0:
            print('Iter:{}, Loss:{}'.format(iteration, loss))
        iteration += 1
    for i in range(time_step - 3, time_step):
        loss, pred = sess.run([cost, predictions], feed_dict={x: input_data(x_valid), y: input_data(y_valid)})
        pred.reshape((valid_batch_num, time_step))
        x_valid[:, i] = pred[:, i - 1]
    print('loss:', loss)
    plt.plot(pred, 'g')
    plt.plot(y_valid.reshape(-1, output_size), 'r')
