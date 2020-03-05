import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vq_vae import VQVAE

class VQVAEPlus(VQVAE):

    def __init__(self,image_size,channels,D,K,L,commitment_beta=1,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.L=L
        self.z_es=[]
        super().__init__(image_size,channels,D,K,commitment_beta,filters,lr,c, num_convs,num_fc)


    def vq_layer(self,inputs,name="lookup_table"):
        #tf.Variable(initial_value=inputs,name=name+"_inputs")
        embeddings = tf.get_variable(name,dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
        #self.z_e=tf.layers.Dense(self.D,activation=tf.nn.relu)(fc_layer)
        self.z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        self.e_k=quantized
        #e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
        #q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
        #loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                        tf.math.log(avg_probs + 1e-10)))

        #expanded_ze = tf.expand_dims(self.z_e, -2)
        #distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
        #self.q_z_x = tf.argmin(distances, axis=-1)
        #self.e_k = tf.gather(params=self.embedding, indices=self.q_z_x)
        #self.z_q = self.z_e + tf.stop_gradient(self.e_k - self.z_e)
        commitment_loss = self.commitment_beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
        return {"outputs":quantized,"commitment_loss":commitment_loss,"vq_loss":vq_loss}
        

    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.])
            self.z=tf.constant([0.])
            print("Inputs %s"%inputs.shape)
            for i in range(self.L):

                out=self.vq_layer(inputs,name="lookup_table_%d"%i)
                self.z+=out["outputs"]
                self.vq_loss+=out["vq_loss"]
                self.commitment_loss+=out["commitment_loss"]

            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
            #print(self.z.shape)
            #self.z=tf.reshape(self.z,[])#[-1,self.D])
            #print(self.z.shape)

    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """
        #reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        #self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))
        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)

        #reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta

        self.losses=[self.loss,self.reconstr_loss,self.vq_loss,self.commitment_loss]
        self.losses_labels=["Total","Recnstr","VQ","Commitment"]

        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.loss)
