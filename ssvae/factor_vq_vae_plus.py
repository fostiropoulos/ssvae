import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vq_vae_plus import VQVAEPlus

class FactorVQVAEPlus(VQVAEPlus):

    def __init__(self,image_size,z_dim,K,L,tc_gamma=35,commitment_beta=1,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.tc_gamma=tc_gamma
        super().__init__(image_size,z_dim,K,L,commitment_beta,filters,lr,c, num_convs,num_fc)

    def _discriminator_init(self, inputs, reuse=False):
        n_units = 100
        with tf.variable_scope("discriminator"):
            disc=inputs
            for i in range(6):
                disc = tf.layers.dense(inputs=disc, units=n_units, activation=tf.nn.leaky_relu, name="disc_%d"%i, reuse=reuse)
            logits = tf.layers.dense(inputs=disc, units=2, name="disc_logits", reuse=reuse)
            probabilities = tf.nn.softmax(logits, name="disc_prob")

        return logits, probabilities

    def _loss_init(self,inputs,outputs):
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))


        """
        TC Loss
        """
        real_samples = self.z
        permuted_rows = []
        for i in range(self.L):
            permuted_rows.append(tf.random_shuffle(real_samples[:, i*self.D:(i+1)*self.D]))
        permuted_samples = tf.concat(permuted_rows, axis=-1)

        logits_real, probs_real = self._discriminator_init(real_samples)
        logits_permuted, probs_permuted = self._discriminator_init(permuted_samples, reuse=True)

        self.tc_regulariser = self.tc_gamma * tf.reduce_mean(logits_real[:, 0]  - logits_real[:, 1], axis=0)

        """
        VQ-LOSS
        """
        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        total_loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss #+tc_regulariser
        tf.summary.scalar("total_loss", total_loss)


        self.commitment_loss += self.beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        self.vq_loss += tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
        #self.disc_loss = -tf.add(0.5 * tf.reduce_mean(tf.log(probs_real[:, 0])), 0.5 * tf.reduce_mean(tf.log(probs_permuted[:, 1])), name="disc_loss")

        self.losses=[total_loss,self.reconstr_loss,self.vq_loss,self.commitment_loss]#,tc_regulariser,self.disc_loss]
        self.losses_labels=["Total","Recnstr","VQ","Commitment"]#,"TC", "Discrim"]

        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        return total_loss
    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))

        """
        VQ-LOSS
        """
        self.total_loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta

        self.losses=[self.total_loss,self.reconstr_loss,self.vq_loss,self.commitment_loss]
        self.losses_labels=["Total","Recnstr","VQ","Commitment"]

        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.total_loss)

    def partial_fit(self,X,X_test=None, batch_size=64):
        indices=np.arange(X.shape[0])
        random.shuffle(indices)
        X=X[indices]

        num_batches=X.shape[0]//batch_size
        for i in range(num_batches):
            X_batch=X[i*batch_size:(i+1)*batch_size]
            self.sess.run(self.train,feed_dict=self.get_feed_dict(X_batch))
            self.sess.run(self.train_disc,feed_dict=self.get_feed_dict(X_batch))


        train_out=self.sess.run([loss for loss in self.losses],
                                feed_dict=self.get_feed_dict(X[:batch_size]))

        # if a test is given calculate test loss
        if(X_test is not None):
            test_indices=np.arange(X.shape[0])
            random.shuffle(indices)
            X_test=X_test[indices]
            test_out=self.sess.run([loss for loss in self.losses],
                                   feed_dict=self.get_feed_dict(X_test[:batch_size]))
        else:
            test_out=[.0]*len(self.losses)

        return train_out, test_out


    def _train_init(self):
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        vq = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vq')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.train = self.optimizer.minimize(self.loss,  var_list=enc_vars+dec_vars+vq)
        self.train_disc=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.disc_loss,  var_list=disc_vars)
