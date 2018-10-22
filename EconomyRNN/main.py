import copy
import tensorflow as tf
import matplotlib.pyplot as plt
from data_process import *
from model import *

filename = "outdata201803_hz_monthly_1.csv"
# filename = "sin.csv"
index_col = 'Dates'
tag = 'CPI'


class TrainConfig:
    time_step = 10
    input_size = 1
    output_size = 1
    hidden_num = 16
    layer_num = 3


train_config = TrainConfig()
data = Data(filename, index_col, tag, train_config.time_step)

model = LSTMModel(train_config.time_step,
                  train_config.input_size,
                  train_config.output_size,
                  train_config.hidden_num,
                  train_config.layer_num)

train_rate = 0.001
train_step = 10000
display_step = 200

init = tf.global_variables_initializer()
saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    def train():
        class Loss:
            train = list()
            valid = list()

        loss = Loss()
        for i in range(train_step + 1):
            _ = sess.run([model.optimizer], feed_dict={model.x: data.train.input.x,
                                                       model.y: data.train.input.y})
            if i % display_step == 0:
                # train loss
                result, train_loss = sess.run([merged, model.loss], feed_dict={model.x: data.train.input.x,
                                                                               model.y: data.train.input.y})
                loss.train.append(train_loss)
                # valid loss
                valid_loss = sess.run(model.loss, feed_dict={model.x: data.valid.input.x,
                                                             model.y: data.valid.input.y})
                loss.valid.append(valid_loss)
                print('Iter:{}\tTrain Loss:{}\tValid Loss:{}'.format(i, train_loss, valid_loss))
                writer.add_summary(result, i)
        plt.plot(loss.train[1:], 'r')
        plt.plot(loss.valid[1:], 'g')


    def valid():
        loss, prediction = sess.run([model.loss, model.predictions], feed_dict={model.x: data.valid.input.x,
                                                                                model.y: data.valid.input.y})
        print('Validation Loss:', loss)
        plt.plot(prediction.reshape(-1, 1), 'g')
        plt.plot(data.valid.input.y[:, -1, :].reshape(-1, 1), 'r')


    def predict():
        plt.plot(data.predict.raw.x, 'r')
        for i in range(50):
            prediction = sess.run(model.predictions, feed_dict={model.x: data.predict.input.x})
            data.roll_predict(prediction[-1][0])
        plt.plot(data.predict.raw.x, 'b')


    # logs
    writer = tf.summary.FileWriter('logs/', sess.graph)
    # tensorboard --logdir logs
    # http://0.0.0.0:6006

    sess.run(init)

    # saver.restore(sess, 'my_net20181004/save_net.ckpt')

    plt.figure()
    plt.show()
    plt.subplot(2, 1, 1)
    train()
    # saver.save(sess, 'my_net/save_net.ckpt')
    # saver.restore(sess, 'my_net20181004/save_net.ckpt')

    plt.subplot(2, 2, 3)
    valid()
    plt.subplot(2, 2, 4)
    predict()
    writer.close()
