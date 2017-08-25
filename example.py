from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import functools
import itertools
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.protobuf import config_pb2


FLAGS = None

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index,
                           config=tf.ConfigProto(rpc_options=config_pb2.RPCOptions(ex_grpc_compression=FLAGS.compression_on))
                          )

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      partition_size = 784//FLAGS.num_partition
      mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
      x = tf.placeholder(tf.float32, [None, partition_size*FLAGS.num_batch])
      W = tf.Variable(tf.zeros([784, 10]))
      partition_index = tf.placeholder(tf.int32)

      #W_ = W[partition_index*partition_size:(partition_index+1)*partition_size, :]
      #W_ = tf.Variable(tf.zeros([784, 0]))
#      for j in partition_index:
       # W_ = tf.concat(W_, W[partition_index*partition_size:(partition_index+1)*partition_size, :], 1)
      W_ = tf.gather(W, partition_index)
      b = tf.Variable(tf.zeros([10]))
      y = tf.matmul(x, W_) + b
      y_ = tf.placeholder(tf.float32, [None, 10])
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
      #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

      #loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.train.GradientDescentOptimizer(0.5).minimize(
          loss, global_step=global_step)
      #train_op = tf.train.AdagradOptimizer(0.01).minimize(
      #    loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           #config=tf.ConfigProto(log_device_placement=True),
                                           config=tf.ConfigProto(rpc_options=config_pb2.RPCOptions(ex_grpc_compression=FLAGS.compression_on)),
                                           hooks=hooks) as mon_sess:

      q = itertools.cycle(itertools.combinations(range(FLAGS.num_partition), FLAGS.num_batch))

      #import pdb
      #pdb.set_trace()

      while not mon_sess.should_stop():
      #for _ in range(1000):
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        
        #pass in parameter num_partition, num_batch
        which = next(q)#bin(functools.reduce(int.__or__, (1<<d for d in q[i % len(q)]), 0))[2:].rjust(FLAGS.num_partition, "0")
        #print(which)

        batch_xs, batch_ys = mnist.train.next_batch(100)
#        mask_matrix = np.zeros((100, 784), np.float32)


        '''
        for j in which:
                mask_matrix[:, j*partition_size:(j+1)*partition_size] = batch_xs[:, j*partition_size:(j+1)*partition_size]

        batch_xs = mask_matrix
        '''
        #index = [0]
        #for j in which:
        #        index = np.concatenate((index, list(range(j*partition_size, (j+1)*partition_size))))        
        index = np.concatenate(tuple(list(range(j*partition_size, (j+1)*partition_size)) for j in which))

        #new_x = []
        #for j in which:
                #np.concatenate(new_x, batch_xs[:, j*partition_size:(j+1)*partition_size], axis = 1)

        import pdb
        # pdb.set_trace()

        batch_xs = batch_xs[:, index]

        #batch_xs = np.concatenate((batch_xs[:, :196], np.zeros((100, 588), np.float32)), axis=1)
	

       # import pdb
       # pdb.set_trace()
        mon_sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys, partition_index: index})

      #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #print(mon_sess.run(accuracy, feed_dict={x: mnist.test.images,
      #                                y_: mnist.test.labels}))

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
      "--num_partition",
      type=int,
      default=1,
      help="Number of partitions of batch_xs"
  )
  parser.add_argument(
      "--num_batch",
      type=int,
      default=1,
      help="Number of partitions to keep from batch_xs"
  )
  parser.add_argument(
      "--remove_oldlogs",
      type=int,
      default=0,
      help="remove old logs"
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=1000,
      help="steps to train"
  )
  parser.add_argument(
      "--compression_on",
      type=str2bool,
      default=True,
      help="whether to turn on compression"
  )
  FLAGS, unparsed = parser.parse_known_args()
  if FLAGS.remove_oldlogs==1:
    import shutil
    shutil.rmtree("/tmp/train_logs")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
