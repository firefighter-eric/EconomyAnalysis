import tensorflow as tf


class LSTMModel:
    def __init__(self, time_step, input_size, output_size, n_hidden, n_layer):
        with tf.name_scope('inputs'):
            x = tf.placeholder(dtype=tf.float32, shape=[None, time_step, input_size], name="input_x")
            y = tf.placeholder(dtype=tf.float32, shape=[None, time_step, output_size], name="expected_y")

        batch_size = tf.shape(x)[0]

        with tf.name_scope('lstm'):
            def get_a_cell(n_hidden):
                return tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(n_hidden) for _ in range(n_layer)])
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
            with tf.name_scope('output'):
                output = output[:, -1:, :]
                output = tf.reshape(output, [-1, n_hidden])

        with tf.name_scope('relu'):
            with tf.name_scope('weight'):
                weights = tf.Variable(tf.truncated_normal(shape=[n_hidden, output_size]), name='weights')
            with tf.name_scope('bias'):
                bias = tf.Variable(tf.zeros(shape=[output_size]), name='bias')
            logits = tf.nn.relu(tf.matmul(output, weights) + bias, name='logits')
            predictions = tf.reshape(logits, [-1, output_size], name='predictions')

        with tf.name_scope('loss'):
            label = tf.reshape(y[:, -1, :], (-1, output_size), name='label')
            loss = tf.losses.mean_squared_error(label, predictions)
            # tf.summary.histogram('Loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer().minimize(loss)

        self.x = x
        self.y = y
        self.weights = weights
        self.output = output
        self.predictions = predictions
        self.loss = loss
        self.optimizer = optimizer
        tf.summary.histogram('Loss', self.loss)

# if __name__ == '__main__':
#     time_step = 10
#     input_size = 1
#     output_size = 1
#     n_hidden = 32
#     n_layer = 3
#     model = LSTMModel(time_step, input_size, output_size, n_hidden, n_layer)
