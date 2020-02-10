import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vq_vae import VQVAE

class VQVAEPlus(VQVAE):

    def __init__(self,image_size,channels,z_dim,K,L,commitment_beta=1,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.L=L
        super().__init__(image_size,channels,z_dim,K,commitment_beta,filters,lr,c, num_convs,num_fc)

    def _z_init(self,fc_layer):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.])
            z=[]
            for i in range(self.L):

                self.embedding = tf.get_variable('lookup_table_%d'%i,dtype=tf.float32,shape=[self.K, self.D],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
                self.z_e=tf.layers.Dense(self.D,activation=tf.nn.relu)(fc_layer)
                self.z_es.append(self.z_e)
                expanded_ze = tf.expand_dims(self.z_e, -2)
                distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
                self.q_z_x = tf.argmin(distances, axis=-1)
                self.e_k = tf.gather(params=self.embedding, indices=self.q_z_x)
                self.z_q = self.z_e + tf.stop_gradient(self.e_k - self.z_e)
                self.commitment_loss += self.beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
                self.vq_loss += tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
                z.append(self.z_q)
            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
            self.z=tf.reshape(tf.stack(z,axis=-2),[-1,self.L*self.D])


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
