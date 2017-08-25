from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Build model...
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784//FLAGS.num_partition])
  W = tf.Variable(tf.zeros([784//FLAGS.num_partition, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  saver = tf.train.Saver()
  # The MonitoredTrainingSession takes care of session initialization,
  # restoring from a checkpoint, saving to a checkpoint, and closing when done
  # or an error occurs.
  with tf.Session() as sess:
    saver.restore(sess, "/tmp/train_logs/model.ckpt-%d" % FLAGS.eval_steps)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument(
      "--eval_steps",
      type=int,
      default=1001,
      help="Stop step of the training"
  )
  parser.add_argument(
      "--num_partition",
      type=int,
      default=1,
      help="Number of partitions of batch_xs"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

