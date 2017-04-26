import tensorflow as tf
import numpy as np
import math
tf.set_random_seed(20160408)


def probability(m):
    m_neg = tf.minimum(m, tf.zeros_like(m))
    prob = tf.reduce_sum(m_neg, 1)
    prob = tf.clip_by_value(prob, math.log(1e-10), -0.0001)
    return prob


def joint_probability_log(m1, m2):
    intersect = intersection_point_log(m1, m2)
    return probability(intersect)


def cond_probability_log(m1, m2):
    # pair
    intersect = intersection_point_log(m1, m2)
    joint_prob_pair = probability(intersect)
    y_prob_pair = probability(m2)
    cond_prob_pair = joint_prob_pair - y_prob_pair  # log probabilities
    cond_prob_pair = tf.clip_by_value(cond_prob_pair, math.log(1e-10), -0.0001)
    return cond_prob_pair


def intersection_point_log(m1, m2):
    intersect = tf.minimum(m1, m2)
    return intersect


# log vector is a vector of negative log probabilities
def create_log_distribution(logits, batch_size):
    log_1 = tf.log(tf.ones([batch_size]))
    log_1_minus = logits + tf.log(tf.exp(log_1 - logits) - tf.ones([batch_size]))
    return tf.concat(1, [tf.expand_dims(logits, 1), tf.expand_dims(log_1_minus, 1)])


# vector is a tensor of (gold) probabilities
def create_distribution(probs, batch_size):
    one_minus = tf.ones([batch_size]) - probs
    return tf.concat(1, [tf.expand_dims(probs, 1), tf.expand_dims(one_minus, 1)])


# assume input is 2 vectors:
# vec1 : predicted cpr values (negative log prob)
# vec2 : gold cpr values
def kl_divergence_batch(pred_vec, gold_vec):
    pred_cpr = np.exp(pred_vec)
    pred_cpr = np.clip(pred_cpr, 0, 1)
    vals = []
    for index, pred_val in enumerate(pred_cpr):
        gold_val = gold_vec[index]
        vals.append(kl_divergence(pred_val, gold_val))
    return np.mean(vals)


def kl_divergence(pred_prob, gold_prob):
    val = 0
    if gold_prob > 0 and pred_prob > 0:
        try:
            val += gold_prob * (math.log(gold_prob / pred_prob))
        except ValueError:
            print(gold_prob, pred_prob)
    if gold_prob < 1 and pred_prob < 1:
        try:
            val += (1-gold_prob) * (math.log((1-gold_prob)/(1-pred_prob)))
        except ValueError:
            print(gold_prob, pred_prob)
    return val