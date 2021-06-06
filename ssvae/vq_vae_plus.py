import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vq_vae import VQVAE

class Mine(tf.keras.Model):
    def __init__(self):
        super(Mine, self).__init__()
        
        H=1280

        # propagate the forward pass
        self.dense=[]
        for l in range(4):
            self.dense.append(tf.keras.layers.Dense(H, activation='relu'))
            self.dense.append(tf.keras.layers.Dropout(0.3))
        self.logs=tf.keras.layers.Dense(1)
        
    def call(self, x):
        out=x
        for l in self.dense:
            out=l(out)
        
        return self.logs(out)

class VQVAEPlus(VQVAE):

    def __init__(self,image_size,channels,D,K,L,commitment_beta=1,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):

        super().__init__(image_size,channels,D,K,L,commitment_beta,filters,lr,c, num_convs,num_fc)


    def _loss_init(self,inputs,outputs):
        
        x_sample=tf.reshape(self.encodings[:,:,:,:1],(-1,2))
        y_sample=tf.reshape(self.encodings[:,:,:,1:],(-1,2))
        x_sample1, x_sample2 = tf.split(x_sample, 2)
        y_sample1, y_sample2 = tf.split(y_sample, 2)
        joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
        marg_sample =  tf.concat([x_sample2, y_sample1], axis=1)

        model = Mine()
        self.joint = model(joint_sample)
        self.marginal=model(marg_sample)
        
        
        mine=-tf.reduce_mean(self.joint) +tf.math.log(tf.reduce_mean(tf.exp(self.marginal)))
        self.mine=mine
        """
        opt=tf.train.AdamOptimizer(learning_rate=0.0001)
        gvs = opt.compute_gradients(-woops)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = opt.apply_gradients(capped_gvs)
        """

        """
        VAE-LOSS
        """
        #reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        #self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))
        reconstr_loss=0.1*mine+tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)

        #reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta

        self.losses=[self.loss,self.reconstr_loss,self.vq_loss,self.commitment_loss,self.mine,self.perplexity]
        self.losses_labels=["Total","Recnstr","VQ","Commitment","MINE","perplexity"]

        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("MINE", self.mine)

        tf.summary.scalar("total_loss", self.loss)
