# train model to predict conditional probabilities and p(x) individual phrase probabilities

import os
import random
import sys
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from util.Layer import Layers
from util.Train_probability import Training
from util.DataLoader import DataLoader
Sparse = DataLoader()
Layer = Layers()
from util import Probability
import time

sys.path.append(".")


def run(**args):
    tf.reset_default_graph()
    tf.set_random_seed(20160906)
    random.seed(20160408)
    start_time = time.time()

    # Read Training/Dev/Test data
    data_dir = 'data/' + args['data_dir'] + '/'
    np_matrix, index = Sparse.read_glove_vectors('data/' + args['data_dir'] + '/' + args['vector_file'])

    # pair data
    if args['method'] == 'train':
        train_1_pair, train_2_pair, train_xlabels_pair, train_ylabels_pair, train_xylabels_pair, train_cpr_xylabels_pair, train_lens1_pair, train_lens2_pair, maxlength, train_phrase1, train_phrase2, train_labels = Sparse.gzread_cpr(data_dir + args['train_data'], index)
        dev_1_pair, dev_2_pair, dev_xlabels_pair, dev_ylabels_pair, dev_xylabels_pair, dev_cpr_xylabels_pair, dev_lens1_pair, dev_lens2_pair, _, dev_phrase1, dev_phrase2, dev_labels            = Sparse.gzread_cpr(data_dir + args['dev_data'], index)
        test_1_pair, test_2_pair, test_xlabels_pair, test_ylabels_pair, test_xylabels_pair, test_cpr_xylabels_pair, test_lens1_pair, test_lens2_pair, _, test_phrase1, test_phrase2, test_labels           = Sparse.gzread_cpr(data_dir + args['test_data'], index)
    elif args['method'] == 'test': # predict probabilities on test file (probability format)
        test_1_pair, test_2_pair, test_xlabels_pair, test_ylabels_pair, test_xylabels_pair, test_cpr_xylabels_pair, test_lens1_pair, test_lens2_pair, maxlength, test_phrase1, test_phrase2, test_labels           = Sparse.gzread_cpr(data_dir + args['test_data'], index)

    graph = tf.get_default_graph()

    # Input -> LSTM -> Outstate
    inputs1 = tf.placeholder(tf.int32, [args['batch_size'], None])
    inputs2 = tf.placeholder(tf.int32, [args['batch_size'], None])
    x_labels = tf.placeholder(tf.float32, [args['batch_size']])
    y_labels = tf.placeholder(tf.float32, [args['batch_size']])
    xy_labels = tf.placeholder(tf.float32, [args['batch_size']])
    cpr_xy_labels = tf.placeholder(tf.float32, [args['batch_size']])
    lengths1 = tf.placeholder(tf.int32, [args['batch_size']])
    lengths2 = tf.placeholder(tf.int32, [args['batch_size']])

    # RNN
    first_scope = 'part_to_train_first'
    with tf.variable_scope(first_scope):

        # LSTM
        embeddings = tf.Variable(np_matrix, dtype=tf.float32, trainable=False)

        lstm = tf.nn.rnn_cell.LSTMCell(args['hidden_dim'], state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=args['dropout'])

        # Prediction
        output_layer = Layer.W(args['hidden_dim'], args['output_dim'], 'Output')
        output_bias  = Layer.b(args['output_dim'], 'OutputBias')

        Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1)
        Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2)
        output, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1, dtype=tf.float32)
        tf.get_variable_scope().reuse_variables()
        output, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2, dtype=tf.float32)
        logits1 = tf.matmul(fstate1[0], output_layer) + output_bias
        logits2 = tf.matmul(fstate2[0], output_layer) + output_bias

        intersect_vec = Probability.intersection_point_log(logits1, logits2)
        joint_predicted = Probability.joint_probability_log(logits1, logits2)
        cpr_predicted = Probability.cond_probability_log(logits1, logits2)
        cpr_predicted_reverse = Probability.cond_probability_log(logits2, logits1)
        x_predicted = Probability.probability(logits1)
        y_predicted = Probability.probability(logits2)

        x_loss = tf.nn.softmax_cross_entropy_with_logits(Probability.create_log_distribution(x_predicted, args['batch_size']), Probability.create_distribution(x_labels, args['batch_size']))
        y_loss = tf.nn.softmax_cross_entropy_with_logits(Probability.create_log_distribution(y_predicted, args['batch_size']), Probability.create_distribution(y_labels, args['batch_size']))
        cpr_loss = tf.nn.softmax_cross_entropy_with_logits(Probability.create_log_distribution(cpr_predicted, args['batch_size']), Probability.create_distribution(cpr_xy_labels, args['batch_size']))
        mean_loss = tf.reduce_mean(args['lambda_px'] * (x_loss + y_loss) + args['lambda_cpr'] * cpr_loss)

    ## Learning ##
    optimizer = tf.train.AdamOptimizer(args['learning_rate'])
    varlist = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=first_scope)
    train_op = optimizer.minimize(mean_loss, var_list=varlist)

    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        if args['method'] == 'train':
            sess.run(tf.initialize_all_variables())
            Trainer = Training(sess, train_op, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, joint_predicted, cpr_predicted, cpr_predicted_reverse, inputs1, inputs2, x_labels, y_labels, xy_labels, cpr_xy_labels, lengths1, lengths2, args['batch_size'], maxlength)
            best_dev = float('inf')
            for e in range(args['start_epoch'], args['end_epoch']):
                print("Outer epoch %d" % e)
                kl_div = Trainer.train(train_1_pair, train_2_pair, train_xlabels_pair, train_ylabels_pair, train_xylabels_pair, train_cpr_xylabels_pair, dev_1_pair, dev_2_pair, dev_xlabels_pair, dev_ylabels_pair, dev_xylabels_pair, dev_cpr_xylabels_pair, train_lens1_pair, train_lens2_pair, dev_lens1_pair, dev_lens2_pair)
                if kl_div < best_dev:
                    save_path = saver.save(sess, "tmp/" + args['exp_name']+"_best.ckpt")
                    print("Best model saved in file: %s" % save_path)
                    best_dev = kl_div
                print("--- %s seconds ---" % (time.time() - start_time))
                save_path = saver.save(sess, "tmp/" + args['exp_name'] + "_"+str(e)+".ckpt")
                print("Model saved in file: %s" % save_path)
        else:
            saver.restore(sess, "tmp/" + args['exp_name'] + ".ckpt")
            Trainer = Training(sess, train_op, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, joint_predicted, cpr_predicted, cpr_predicted_reverse, inputs1, inputs2, x_labels, y_labels, xy_labels, cpr_xy_labels, lengths1, lengths2, args['batch_size'], maxlength)
            x_pred, y_pred, xy_pred, cpr_pred, cpr_pred_reverse = Trainer.predict(test_1_pair, test_2_pair, test_lens1_pair, test_lens2_pair)
            out_file = open(data_dir + args['exp_name'] + "_" + args['test_data'].split(".")[0] + "_pred_prob.txt", "w")
            count = 0
            for idx, cpr_prob in enumerate(cpr_pred):
                cpr_prob = np.exp(cpr_prob)
                cpr_prob_rev = np.exp(cpr_pred_reverse[idx])
                x_prob = np.exp(x_pred[idx])
                y_prob = np.exp(y_pred[idx])
                xy_prob = np.exp(xy_pred[idx])
                pmi = np.log(xy_prob / (x_prob * y_prob)) / -np.log(xy_prob)
                count += 1
                if count % 1000 == 0:
                    print(count)
                s1 = [str(a) for a in test_1_pair[idx]]
                s2 = [str(a) for a in test_2_pair[idx]]
                if args['method'] == 'test':
                    p1 = test_phrase1[idx]
                    p2 = test_phrase2[idx]
                    out_file.write("%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s\t%s" % (
                    x_prob, y_prob, xy_prob, pmi, cpr_prob, cpr_prob_rev, " ".join(s1), p1, " ".join(s2), p2))
                    if len(test_labels) == len(cpr_pred):
                        out_file.write("\t%s" % test_labels[idx])
                    out_file.write("\n")
            out_file.close()
            if args['method'] == 'test':
                pred_corr = np.corrcoef(np.exp(cpr_pred), test_cpr_xylabels_pair)[0, 1]
                print("Prediction correlation: %f" % pred_corr)
                kldiv = Probability.kl_divergence_batch(cpr_pred, test_cpr_xylabels_pair)
                print("KL divergence: %f" % kldiv)
            print("--- %s seconds ---" % (time.time() - start_time))