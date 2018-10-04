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
        return ret_x, ret_y


def normalize(X, Y=None):
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


def generate_x_y_data(isTrain, batch_size, l_x, l_y):
    path = '/home/panchenko/PycharmProjects/seq2seq/hullma_3m_180d.csv'
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


def main():
    # input dimensions
    batch_size = 1
    output_dim = 1   # Output dimension (e.g.: multiple signals at once, tied in time)
    hidden_dim = 20  # Count of hidden neurons in the recurrent units.

    # Optmizer:
    learning_rate = 0.005  # Small lr helps not to diverge during training.
    nb_iters = 4000  # How many times we perform a training step (therefore how many times we show a batch).
    lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting

    # NN size
    encoder_seq_length = 50
    decoder_seq_length = 5

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.variable_scope('Seq2seq'):

        cells = list()
        cells.append(tf.contrib.rnn.GRUCell(hidden_dim))
        cells.append(tf.contrib.rnn.GRUCell(output_dim))
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        ### encoder
        encoder_input = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="enc_inp")
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_input, dtype=tf.float32)
        expected_sparse_output = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="expected_sparse_output")

        ### training decoder
        decoder_input = tf.placeholder(shape=(None, None, 1), dtype=tf.float32, name="dec_inp")
        decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(decoder_input, decoder_lengths, sampling_probability=0.0)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, encoder_state)
        final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
        # output_scale_factor = 1.0
        output = output_scale_factor*final_outputs.rnn_output
        ### Loss
        output_loss = tf.losses.mean_squared_error(output, expected_sparse_output)

        # Inference Decoder
        inference_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(decoder_input, decoder_lengths, sampling_probability=1.0)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell, inference_helper, encoder_state)
        inference_final_outputs, inference_final_state, inference_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(inference_decoder)
        inference_logits = inference_final_outputs.rnn_output*output_scale_factor

        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # train_op = optimizer.minimize(loss)

            global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
            train_op = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=learning_rate, optimizer='Adam', global_step=global_step, clip_gradients=2.5)

        sess.run(tf.global_variables_initializer())

        # Training
        train_losses = []
        test_losses = []
        for t in range(nb_iters + 1):

            X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size, l_x=encoder_seq_length, l_y=decoder_seq_length)

            decoder_input_np = np.concatenate((X[:, -1:, :], Y[:, :-1, :]), axis=1)
            feed_dict = {encoder_input: X, expected_sparse_output: Y, decoder_input: decoder_input_np, decoder_lengths: [decoder_seq_length]}
            _, train_loss, logits_np = sess.run([train_op, loss, encoder_output], feed_dict)
            train_losses.append(train_loss)

            if t % 10 == 0:
                # Tester
                X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size, l_x=encoder_seq_length,
                                         l_y=decoder_seq_length)
                feed_dict = {encoder_input: X, expected_sparse_output: Y, decoder_input: decoder_input_np,
                             decoder_lengths: [decoder_seq_length]}
                _, test_loss, logits_np_test = sess.run([train_op, loss, encoder_output], feed_dict)

                test_losses.append(test_loss)
                print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))

        print 'Done'


    ### Build some diagrams

    plt.figure(figsize=(12, 6))
    plt.plot(
        np.array(range(0, len(test_losses))) / float(len(test_losses) - 1) * (len(train_losses) - 1),
        np.log(test_losses),
        label="Test loss"
    )
    plt.plot(
        np.log(train_losses),
        label="Train loss"
    )
    plt.title("Training errors over time (on a logarithmic scale)")
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    plt.show()


    # Test
    nb_predictions = 50
    print("Let's visualize {} predictions with our signals:".format(nb_predictions))

    # encoder_seq_length += 50
    X, Y = generate_x_y_data(isTrain=False, batch_size=1, l_x=encoder_seq_length, l_y=decoder_seq_length)
    inference_decoder_input_np = np.concatenate((X[:, -1:, :], Y[:, :-1, :]), axis=1)
    feed_dict = {encoder_input: X, decoder_input: inference_decoder_input_np, decoder_lengths: [decoder_seq_length]}
    outputs = sess.run(inference_logits, feed_dict)

    for j in range(nb_predictions):

        X, Y = generate_x_y_data(isTrain=False, batch_size=1, l_x=encoder_seq_length, l_y=decoder_seq_length)
        inference_decoder_input_np = np.concatenate((X[:, -1:, :], Y[:, :-1, :]), axis=1)
        feed_dict = {encoder_input: X, decoder_input: inference_decoder_input_np, decoder_lengths: [decoder_seq_length]}
        outputs = sess.run(inference_logits, feed_dict)



        plt.figure(figsize=(12, 3))

        for k in range(1):
            past = X[0, :, 0]
            expected = Y[0, :, 0]
            pred = outputs[0, :, 0]

            label1 = "Seen (past) values" if k == 0 else "_nolegend_"
            label2 = "True future values" if k == 0 else "_nolegend_"
            label3 = "Predictions" if k == 0 else "_nolegend_"
            plt.plot(range(len(past)), past, "o--b", label=label1)
            plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b", label=label2)
            plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)
            print(pred)
        plt.legend(loc='best')
        plt.title("Predictions v.s. true values")
        plt.show()


if __name__ == '__main__':
    main()