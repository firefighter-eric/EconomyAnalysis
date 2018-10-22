import tensorflow as tf


class Model:
    def __init__(self, time_step, input_size, output_size, hidden_num):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, time_step, input_size], name="input_x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, time_step, output_size], name="expected_y")

        weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, output_size]))
        bias = tf.Variable(tf.zeros(shape=[output_size]))

        batch_size = tf.shape(self.x)[0]
        inputs = tf.reshape(self.x, shape=[batch_size, time_step, input_size])
        cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
        initial_state = tf.zeros([batch_size, cell.state_size])

        outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)

        outputs = tf.reshape(outputs, [-1, hidden_num])
        logits = tf.nn.relu(tf.matmul(outputs, weights) + bias)
        predictions = tf.reshape(logits, [-1, output_size])

        label = tf.reshape(self.y, (-1, output_size))
        cost = tf.losses.mean_squared_error(label, predictions)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        self.weights = weights
        self.predictions = predictions
        self.cost = cost
        self.optimizer = optimizer
