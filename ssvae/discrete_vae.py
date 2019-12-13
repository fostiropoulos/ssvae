import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vae import VAE

class DiscreteVAE(VAE):

    def __init__(self,image_size,z_dim,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):

        self.beta=1
        super().__init__(image_size,z_dim,filters,lr,c, num_convs,num_fc)

    def _init_z(self,fc_layer):
        size=1
        fc_layer=tf.layers.Dense(self.z_dim*size,activation=tf.nn.relu)(fc_layer)
        #self.mu=[]
        #self.sigma=[]
        z=[]
        K=10
        D=10

        self.embedding = tf.get_variable('lookup_table',dtype=tf.float32,shape=[K, D],initializer=tf.truncated_normal_initializer(mean=0.1, stddev=1.2))
        self.z_e=fc_layer

        expanded_ze = tf.expand_dims(self.z_e, -2)
        distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
        #tf.summary.histogram('distances', distances)

        # q(z|x) refers to the 2d grid in the middle in figure 1
        self.q_z_x = tf.argmin(distances, axis=-1)
        #tf.summary.histogram('q(z|x)', self.q_z_x)
        #self._print('q(z|x):', self.q_z_x)
        self.e_k = tf.gather(params=self.embedding, indices=self.q_z_x)
        #tf.summary.histogram('e_k', self.e_k)

        self.z_q = self.z_e + tf.stop_gradient(self.e_k - self.z_e)



        #for i in range(self.z_dim):
        #self.mu=tf.reshape(tf.layers.Dense(self.z_dim*size,activation=None)(fc_layer),[-1,self.z_dim,size])
        #self.sigma=tf.reshape(tf.layers.Dense(self.z_dim*size,activation=None)(fc_layer),[-1,self.z_dim,size])

        #z=DiscreteVAE.sample(self.mu,self.sigma)
        #self.sigma.append(sigma)
        #print(z.shape)
        #z=argmax(self.mu)
        #z=(tf.reshape(argmax(z),[-1,self.z_dim*10]))
        #print(z.shape)
        #print(z.shape)
        #z=tf.reshape(softmax(z),[-1,self.z_dim*10])
        #z=argmax(z)
        #print(tf.concat(z,axis=-1).shape)

        return self.z_q #tf.add(tf.reshape(z,[-1,self.z_dim*10]),fc_layer)

    def sample_gumbel(shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax(logits, temperature, hard=False):
        gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)

        if hard:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                             y.dtype)
            y = tf.stop_gradient(y_hard - y) + y

        return y

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps), name="z")
        return z
    def get_mu(self,X):
        return self.sess.run(self.z,feed_dict={self.X:X})
    def _init_loss(self,inputs,outputs):
        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))
        #self.latent_loss= -tf.reduce_sum(tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma))))


        """
        VQ-LOSS
        """
        self.vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
        tf.summary.scalar('vq_loss', self.vq_loss)

        self.commitment_loss = self.beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        tf.summary.scalar('commitment_loss', self.commitment_loss)



        #vq_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.mu), (tf.reshape(self.z,[-1,self.z_dim,10]))))
        #enc_loss = 1 * tf.reduce_mean(tf.squared_difference(self.mu, tf.stop_gradient((tf.reshape(self.z,[-1,self.z_dim,10])))))
        #tf.summary.scalar("latent_loss", self.latent_loss)
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        total_loss=self.reconstr_loss + self.vq_loss + self.commitment_loss#+vq_loss+enc_loss#+ self.latent_loss * self.c
        tf.summary.scalar("total_loss", total_loss)

        self.losses=[self.reconstr_loss,self.vq_loss,self.commitment_loss]
        return total_loss
