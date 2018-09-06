  # Possible values: 1, 2, 3, or 4.

import requests

# from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3
import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from random import shuffle


class Phase(object):
    TEST = 0
    TRAIN = 1


class Pos(object):
    pos = 0

    def __init__(self, pos):
        self.pos = pos

    def increment(self):
        self.pos += 1


class Data(object):

    win_x = 0
    win_y = 0

    num_class = 1
    train_batch_position = Pos(0)
    test_batch_position = Pos(0)

    def __init__(self, path, win_x, win_y):
        self.win_x = win_x
        self.win_y = win_y
        data_pd = pd.read_csv(path)
        close_price = data_pd['diff']  # get close prices in Pandas DataFrames array
        close_price_diffs = close_price#.pct_change()  # calculate price change in percents  p(i)/p(i-1) - 1
        # plt.plot(close_price_diffs)
        # plt.show()

        self.data = close_price_diffs.as_matrix()[1:]  # to numpy, without first value, because it is NaN

        self.train, self.test = train_test_split(self.data, test_size=0.3, shuffle=False)
        self.ind_train = range(self.train.shape[0] - self.win_x - self.win_y)
        self.ind_test = range(self.test.shape[0] - self.win_x - self.win_y)
        # shuffle(self.ind_train)
        # shuffle(self.ind_test)

    def get_batch(self, data, indexes, pos,  size, win_x, win_y):
        if pos.pos >= len(indexes) - size:
            pos.pos = 0
            # shuffle(indexes)
        batch_ind = indexes[pos.pos:pos.pos + size]
        batch_x = list()
        batch_y = list()
        for i in batch_ind:
            batch_x.append(data[i:i + win_x])
            batch_y.append(data[i + win_x: i + win_x + win_y])
            pos.pos += 1
        ret_x = np.array(batch_x)
        ret_y = np.array(batch_y)
        ret_x, ret_y = normalize(ret_x, ret_y)

        # if win_y == 1:
        # ret_y = np.expand_dims(ret_y, -1)
        # ret_y = ret_y > 0
        # ret_x = ret_x > 0
        return ret_x, ret_y


def normalize(X, Y=None):
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        # assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


def generate_x_y_data(isTrain, batch_size, l_x, l_y):
    path = '/home/serg/PycharmProjects/seq2seq/hullma_1h_90d.csv'
    data = Data(path, l_x, l_y)
    if isTrain:
        batch_xs, batch_ys = data.get_batch(data.train, data.ind_train, data.train_batch_position, batch_size, data.win_x,
                                        data.win_y)
    else:
        batch_xs, batch_ys = data.get_batch(data.test, data.ind_test, data.test_batch_position,
                                                  batch_size, data.win_x, data.win_y)
    batch_xs = np.expand_dims(batch_xs, axis=2)
    batch_ys = np.expand_dims(batch_ys, axis=2)
    return batch_xs, batch_ys


def train_batch(batch_size):
    X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length )
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t



def test_batch(batch_size):
    X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]



def train_batch_new(batch_size):
    X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length )
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t

encoder_seq_length = 100
decoder_seq_length = 5


batch_size = 6

sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length)
print("Dimensions of the dataset for 3 X and 3 Y training examples : ")
print(encoder_seq_length)
print(decoder_seq_length)
print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters


output_dim = sample_y.shape[-1]
input_dim = sample_x.shape[-1]  # Output dimension (e.g.: multiple signals at once, tied in time)
hidden_dim = 20  # Count of hidden neurons in the recurrent units.
layers_stacked_count = 2  # Number of stacked recurrent cells, on the neural depth axis.

# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
nb_iters = 1000  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):


    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    encoder_input = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="enc_inp")

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_input, dtype=tf.float32)

    expected_sparse_output = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="expected_sparse_output")

    decoder_input = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="dec_inp")
    decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')

    helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, decoder_lengths)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, encoder_state)

    final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

    logits = final_outputs.rnn_output

    output_loss = tf.reduce_mean(tf.nn.l2_loss(logits - expected_sparse_output))

    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())
    for t in range(nb_iters + 1):

        X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length)

        decoder_input_np = np.concatenate((X[:, -2:-1, :], Y[:, 1:, :]), axis=1)
        feed_dict = {encoder_input: X, expected_sparse_output: Y, decoder_input: decoder_input_np, decoder_lengths: [decoder_seq_length]}
        _, loss_t = sess.run([train_op, loss], feed_dict)
        print("Step {}/{}, train loss: {}".format(t, nb_iters, loss_t))

    print 'sdf'


















#
#
#
#
#     enc_inp = [
#         tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
#         for t in range(encoder_seq_length)
#     ]
#
#     expected_sparse_output = [
#         tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
#         for t in range(decoder_seq_length)
#     ]
#
#     dec_inp = [tf.zeros_like(enc_inp[-decoder_seq_length], dtype=np.float32, name="GO")] + enc_inp[-decoder_seq_length+1:]
#
#     cells = []
#     for i in range(layers_stacked_count):
#         with tf.variable_scope('RNN_{}'.format(i)):
#             cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
#     cell = tf.nn.rnn_cell.MultiRNNCell(cells)
#     dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
#         enc_inp,
#         dec_inp,
#         cell
#     )
#
#     w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
#     b_out = tf.Variable(tf.random_normal([output_dim]))
#
#     output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
#
#     reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]
#
#
# # Training loss and optimizer
#
# with tf.variable_scope('Loss'):
#     # L2 loss
#     output_loss = 0
#     for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
#         # output_loss += tf.sqrt(tf.losses.mean_squared_error(_y, _Y))
#         output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
#
#     # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
#     reg_loss = 0
#     for tf_var in tf.trainable_variables():
#         if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
#             reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
#
#     loss = output_loss + lambda_l2_reg * reg_loss
#
# with tf.variable_scope('Optimizer'):
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     train_op = optimizer.minimize(loss)
#
#
# # Training
# train_losses = []
# test_losses = []
#
# sess.run(tf.global_variables_initializer())
# for t in range(nb_iters + 1):
#     train_loss = train_batch(batch_size)
#     train_losses.append(train_loss)
#
#     if t % 10 == 0:
#         # Tester
#         test_loss = test_batch(batch_size)
#         test_losses.append(test_loss)
#         print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))
#
# print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
#
#
# plt.figure(figsize=(12, 6))
# plt.plot(
#     np.array(range(0, len(test_losses))) / float(len(test_losses) - 1) * (len(train_losses) - 1),
#     np.log(test_losses),
#     label="Test loss"
# )
# plt.plot(
#     np.log(train_losses),
#     label="Train loss"
# )
# plt.title("Training errors over time (on a logarithmic scale)")
# plt.xlabel('Iteration')
# plt.ylabel('log(Loss)')
# plt.legend(loc='best')
# plt.show()
#
#
# # Test
# nb_predictions = 100
# print("Let's visualize {} predictions with our signals:".format(nb_predictions))
#
# X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions, l_x=encoder_seq_length, l_y=decoder_seq_length)
# feed_dict = {enc_inp[t]: X[t] for t in range(encoder_seq_length)}
# outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
#
# for j in range(nb_predictions):
#     plt.figure(figsize=(12, 3))
#
#     for k in range(1):
#         past = X[:, j, k]
#         expected = Y[:, j, k]
#         pred = outputs[:, j, k]
#
#         label1 = "Seen (past) values" if k == 0 else "_nolegend_"
#         label2 = "True future values" if k == 0 else "_nolegend_"
#         label3 = "Predictions" if k == 0 else "_nolegend_"
#         plt.plot(range(len(past)), past, "o--b", label=label1)
#         plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b", label=label2)
#         plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)
#         print(pred)
#     plt.legend(loc='best')
#     plt.title("Predictions v.s. true values")
#     plt.show()
#
# print("Reminder: the signal can contain many dimensions at once.")
# print("In that case, signals have the same color.")
# print("In reality, we could imagine multiple stock market symbols evolving,")
# print("tied in time together and seen at once by the neural network.")
