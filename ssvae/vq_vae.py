import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from ssvae.ssvae.cnn_vae import cnnVAE

class VQVAE(cnnVAE):

    def __init__(self,image_size,channels, D,K,L,commitment_beta=0.25,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.commitment_beta=commitment_beta
        # dimension of quantization vector
        self.D=D
        # number of qunatization vectors
        self.K=K
        # using same parameters as vq-vae demo
        self.num_hiddens=256
        # using same parameters as vq-vae demo
        self.num_res_hiddens=128
        # number of codebooks
        self.L=L
        # vq-layer output
        self.z=None

        super().__init__(image_size,channels,None,filters,lr,c, num_convs,num_fc)

    def quantize(self, embeddings,encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(embeddings, [1, 0])
        # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
        # supported in V2. Are we missing anything here?
        return tf.nn.embedding_lookup(w, encoding_indices)
    
    def vq_layer(self,inputs,name="lookup_table"):

        embeddings = tf.get_variable(name,dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
        z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, self.D])
        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))
        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        e_k=quantized


        # Straight Through Estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                        tf.math.log(avg_probs + 1e-10)))

        commitment_loss = self.commitment_beta * tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)
        return {"outputs":quantized,"perplexity":perplexity,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encoding_indices}
        

    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.])            
            self.vq_inputs=inputs

            z=[]
            encodings=[]
            self.perplexity=[]
            
            for i in range(self.L):

                out=self.vq_layer(inputs,name="lookup_table_%d"%i)
                
                
                self.vq_loss+=out["vq_loss"]

                self.commitment_loss+=out["commitment_loss"]
                
                self.perplexity.append(out["perplexity"])
                

                z.append(out["outputs"])
                encodings.append(tf.cast(tf.expand_dims(out["encodings"],-1),tf.float32))

            self.encodings=tf.concat(encodings,axis=-1)
            self.z=tf.concat(z,axis=-1)
            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])


    def _encoder_init(self,conv):

        with tf.variable_scope("encoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens

            i=0
            conv=(tf.keras.layers.Conv2D(num_hiddens//2,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)

            for _ in range(self.num_convs):
                i+=1
                conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
                self.conv_layers.append(conv)

            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)

            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            
            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=second_res # resnet v1
            conv=tf.nn.relu(conv)
            conv=(tf.keras.layers.Conv2D(self.D,1,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)
            self.conv_layers.append(conv)

            last_conv=self.conv_layers[-1]
            
        return last_conv

    def _decoder_init(self,conv):
        with tf.variable_scope("decoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens


            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=tf.nn.relu(second_res) # resnet v1
            i=0
            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (conv)
            for _ in range(self.num_convs):
                i+=1
                deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)

            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)

        return tf.reshape(last_layer,[-1,self.image_size,self.image_size,self.channels,256])




    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """

        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """

        self.commitment_loss = self.commitment_beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        self.vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)

        self.loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta

        self.losses=[self.loss,self.reconstr_loss,self.vq_loss,self.commitment_loss]
        self.losses_labels=["Total","Recnstr","VQ","Commitment"]

        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.loss)
