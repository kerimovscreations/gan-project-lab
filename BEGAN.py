#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import sys
import logging
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.python.layers.base import Layer, InputSpec

from ops import *
from utils import *

def cluster_layer(inputs, n_clusters, weights=None, alpha=1.0, **kwargs):
    """Create a new clustering layer and apply it to the `inputs` tensor."""
    layer = ClusteringLayer(n_clusters, weights=weights, alpha=alpha, **kwargs)
    return layer.apply(inputs)


class ClusteringLayer(Layer):
    """
    Define a layer to calculate soft targets via Student's t-distribution.

    Input to this layer must be 2D.
    Output is a 2D tensor with shape: (None, k)
    """

    def __init__(self, k, weights=None, alpha=1.0, **kwargs):
        """Save all relevant variables needed to build the layer."""
        super(ClusteringLayer, self).__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.initialize_with_weights = weights

        # Define an InputSpec for our layer; we don't know shape of input yet,
        # but we know it has to be 2D.
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        """
        Construct the (tensor) variables for the layer.
        This is (automatically) called a single time before the first call().
        """

        if input_shape[1].value is None or len(input_shape) != 2:
            raise ValueError('The 2nd (and last) dimension of the inputs to '
                             '`ClusteringLayer` should be defined. Found `None`.')

        # Redefine the InputSpec now that we have shape information
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_shape[1]))

        # Create the tensorflow variable for the trainable params of the layer
        # i.e. the weights for the similarities between embedded points and
        # cluster centroids (as measured by Student's t-distribution)
        self.clusters = self.add_variable(
            name='clusters',
            shape=[self.k, input_shape[1]],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=self.dtype,
            trainable=True)

        # If weights were provided to the constructor, load them
        if self.initialize_with_weights is not None:
            self.clusters = tf.assign(self.clusters, self.initialize_with_weights)
            del self.initialize_with_weights

        # We must assign self.built = True for tensorflow to use the layer
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Compute soft targets q_ij via Sudent's t-distribution.

        Here we compute the numerator of equation (1) for q_ij, then normalize
        by dividing by the total sum over all vectors in the numerator.
        :param inputs:
        """

        # We use axis arg to norm so that the tensor is treated as a batch of vectors.
        num = (1.0 + tf.norm((tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha)
        num **= -((self.alpha + 1.0) / 2.0)
        return num / tf.reduce_sum(num)

    def compute_output_shape(self, input_shape):
        """Show output shape as (?, k)."""
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.k

def target_distribution(q):
    """Compute the target distribution p, based on q."""
    # q in this form is a numpy array, i.e. not symbolic
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T


def load_autoencoder_weights(sess, saver):
    """
    Initialize all variables in the session, then restore the weights of the
    autoencoder.
    """
    AE_LOGDIR = os.path.join(os.path.dirname(__file__), "autoencoder_logdir")

    # Create a saver and session, then init all variables
    sess.run(tf.global_variables_initializer())

    logging.info("Attempting to restore pretrained AE weights")
    # Restore the pretrained weights of the AE
    ckpt = tf.train.get_checkpoint_state(AE_LOGDIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        logging.info("Successfully restored AE weights")
    else:
        logging.error("Unable to restore pretrained AE weights; terminating")
        sys.exit(1)


def encode_samples(samples, input_tensor, encode_fn, sess, saver):
    """
    Given a 4D tensor of samples, encode the tensor with encode_fn after
    loading the autoencoder weights.
    """

    load_autoencoder_weights(sess, saver)

    logging.info("Encoding samples into latent feature space Z")
    Z = sess.run(encode_fn, feed_dict={input_tensor: samples})

    return Z

class BEGAN(object):
    model_name = "BEGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # BEGAN Parameter
            self.gamma = 0.75
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist()

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, sess, saver,  is_training=True, reuse=False):
        # It must be Auto-Encoder style architecture
        # Architecture : (64)4c2s-FC32_BR-FC64*14*14_BR-(1)4dc2s_S
        with tf.variable_scope("discriminator", reuse=reuse):

            net = tf.nn.relu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = tf.reshape(net, [self.batch_size, -1])
            code = tf.nn.relu(bn(linear(net, 32, scope='d_fc6'), is_training=is_training, scope='d_bn6'))
            net = tf.nn.relu(bn(linear(code, 64 * 14 * 14, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            net = tf.reshape(net, [self.batch_size, 14, 14, 64])
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='d_dc5'))

            # recon loss
            # recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / self.batch_size

            # -------------

            placeholderImage = tf.placeholder(tf.float32, shape=(None, 1, *(28, 28)), name="image")

            Z = encode_samples(self.data_X, placeholderImage, code, sess, saver)
            var1 = tf.placeholder(tf.float32, shape=(None, 10), name="real_images")

            kmeans = KMeans(n_clusters=10, n_init=20)
            kmeans.fit_predict(out.eval())
            cluster = cluster_layer(code, 10, weights=kmeans.cluster_centers_)
            gamma = 1
            cross_entropy = -tf.reduce_sum(var1 * tf.log(cluster))
            entropy = -tf.reduce_sum(var1 * tf.log(var1 + 0.00001))
            Lc = cross_entropy - entropy

            Lr = tf.losses.mean_squared_error(x, out)

            recon_error = Lr + gamma * Lc
            # ---------

            return out, recon_error, code

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ BEGAN variable """
        self.k = tf.Variable(0., trainable=False)

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        saver, sess = tf.train.Saver(), tf.Session()

        # output of D for real images
        D_real_img, D_real_err, D_real_code = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        self.d_loss = D_real_err - self.k*D_fake_err

        # get loss for generator
        self.g_loss = D_fake_err

        # convergence metric
        self.M = D_real_err + tf.abs(self.gamma*D_real_err - D_fake_err)

        # operation for updating k
        self.update_k = self.k.assign(
            tf.clip_by_value(self.k + self.lamda*(self.gamma*D_real_err - D_fake_err), 0, 1))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
        d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        M_sum = tf.summary.scalar("M", self.M)
        k_sum = tf.summary.scalar("k", self.k)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.p_sum = tf.summary.merge([M_sum, k_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update k
                _, summary_str, M_value, k_value = self.sess.run([self.update_k, self.p_sum, self.M, self.k], feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, M: %.8f, k: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, M_value, k_value))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
