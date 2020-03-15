import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.vq_vae import VQVAE

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
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


    def vq_layer(self,inputs,name="lookup_table"):
        #tf.Variable(initial_value=inputs,name=name+"_inputs")
        embeddings = tf.get_variable(name,dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
        #self.z_e=tf.layers.Dense(self.D,activation=tf.nn.relu)(fc_layer)
        self.z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, self.D])
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
        self.perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                        tf.math.log(avg_probs + 1e-10)))

        #expanded_ze = tf.expand_dims(self.z_e, -2)
        #distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
        #self.q_z_x = tf.argmin(distances, axis=-1)
        #self.e_k = tf.gather(params=self.embedding, indices=self.q_z_x)
        #self.z_q = self.z_e + tf.stop_gradient(self.e_k - self.z_e)
        commitment_loss = self.commitment_beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
        return {"outputs":quantized,"perplexity":self.perplexity,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encoding_indices}
        

    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.])
            self.z=tf.constant([0.])
            print("Inputs %s"%inputs.shape)
            self.vq_inputs=inputs
            z_s=[]
            encodings=[]
            for i in range(self.L):

                out=self.vq_layer(inputs,name="lookup_table_%d"%i)
                z_s.append(out["outputs"])
                self.vq_loss+=out["vq_loss"]
                self.commitment_loss+=out["commitment_loss"]

                encodings.append(tf.cast(tf.expand_dims(out["encodings"],-1),tf.float32))
            self.encodings=tf.concat(encodings,axis=-1)
            self.z=tf.concat(z_s,axis=-1)
            x_sample=tf.reshape(self.encodings[:,:,:,:1],(-1,2))
            y_sample=tf.reshape(self.encodings[:,:,:,1:],(-1,2))
            x_sample1, x_sample2 = tf.split(x_sample, 2)
            y_sample1, y_sample2 = tf.split(y_sample, 2)
            joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
            marg_sample =  tf.concat([x_sample2, y_sample1], axis=1)

            model = MyModel()
            self.joint = model(joint_sample)
            self.marginal=model(marg_sample)
            
            
            
            #print("z= %s"%self.z.shape)
            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
            #print(self.z.shape)
            #self.z=tf.reshape(self.z,[])#[-1,self.D])
            #print(self.z.shape)

    def _loss_init(self,inputs,outputs):


        #mine=tf.constant([0.])#
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
