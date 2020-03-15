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
        self.D=D
        self.K=K
        self.num_hiddens=256
        self.num_res_hiddens=128
        self.embedding_dim=D
        self.L=L
        self.z_es=[]

        super().__init__(image_size,channels,None,filters,lr,c, num_convs,num_fc)

    def quantize(self, embeddings,encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(embeddings, [1, 0])
        # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
        # supported in V2. Are we missing anything here?
        return tf.nn.embedding_lookup(w, encoding_indices)
    
    def vq_layer(self,inputs,name="lookup_table"):
        #tf.Variable(initial_value=inputs,name=name+"_inputs")
        embeddings = tf.get_variable(name,dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
        #self.z_e=tf.layers.Dense(self.D,activation=tf.nn.relu)(fc_layer)
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
        commitment_loss = self.commitment_beta * tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)
        return {"outputs":quantized,"perplexity":self.perplexity,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encoding_indices}
        


    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.embeddings = tf.get_variable('lookup_table',dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
            #self.z_e=tf.layers.Dense(self.D,activation=tf.nn.relu)(fc_layer)
            self.z_e=inputs
            flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
            distances = (
                tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
                2 * tf.matmul(flat_inputs, self.embeddings) +
                tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

            encoding_indices = tf.argmax(-distances, 1)
            encodings = tf.one_hot(encoding_indices, self.K)
            encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
            quantized = self.quantize(encoding_indices)
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
            self.z=quantized


    def _encoder_init(self,conv):
        # conv layers
        with tf.variable_scope("encoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
            # 64 (2,2)x(4,4)
            # 128  (2,2)x(4,4)
            # 128 (1,1)x(3,3)
            # x2
            # 32 (1,1)x(3,3)
            # 128 (1,1)x(1,1)
            # skip connection to top of loop
            # [64,128,128]
            i=0
            conv=(tf.keras.layers.Conv2D(num_hiddens//2,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            # my changes
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)
            #i+=1
            #conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            #self.conv_layers.append(conv)
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
            conv=(tf.keras.layers.Conv2D(embedding_dim,1,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)
            self.conv_layers.append(conv)

            last_conv=self.conv_layers[-1]

            #conv_shape=int(np.prod(last_conv.shape[1:]))
            #conv_img_shape=int(np.prod(last_conv.shape[1:-1]))
            #print(last_conv.shape[1:-1])
            #last_conv=tf.transpose(last_conv, [0,3,1,2])
            #flatten=tf.reshape(last_conv,(-1,embedding_dim,conv_img_shape))
            #flatten=tf.reshape(last_conv,(-1,conv_shape))
            #print(flatten.shape)
            #dense layers
            #fc=flatten
            #for i in range(2):
            #    fc=tf.layers.Dense(1024,activation=tf.nn.relu,name="enc_dense_%d"%i)(fc)
            #    self.fc.append(fc)

            print(last_conv.shape)
        return last_conv

    def _decoder_init(self,conv):
        with tf.variable_scope("decoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
            print(conv.shape)
            #last_conv=self.conv_layers[-1]
            #conv_shape=int(np.prod(last_conv.shape[2:]))
            #conv_shape=int(np.prod(last_conv.shape[1:]))
            #conv_img_shape=int(np.prod(last_conv.shape[1:-1]))

            #fc=tf.layers.Dense(1024,activation=tf.nn.relu,name="dec_dense")(fc)
            #fc=tf.layers.Dense(conv_shape,activation=tf.nn.relu,name="dec_dense")(fc)
            
            # convert to a 3d tensor from 2d dense
            #conv_reshape=(-1,last_conv.shape[],deconv_size,self.filters)

            #conv_reshape=(-1,int(last_conv.shape[1]),int(last_conv.shape[2]),int(last_conv.shape[3]))
            #conv_reshape=(-1,int(last_conv.shape[1]),int(last_conv.shape[2]),embedding_dim)

            #print(conv_reshape)
            #fc=tf.transpose(fc, [0,2,1])
            #print(fc.shape)

            #reshaped=tf.reshape(fc,conv_reshape)

            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=tf.nn.relu(second_res) # resnet v1
            i=0

            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (conv)
            i+=1

            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)
            i+=1
            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)
            i+=1
            #deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)
            #   i+=1
            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)
            i+=1
            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)
            print(last_layer.shape)
        return tf.reshape(last_layer,[-1,self.image_size,self.image_size,self.channels,256])



    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """
        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)

        #reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
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
