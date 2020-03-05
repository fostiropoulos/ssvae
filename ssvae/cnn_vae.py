import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#from collections import OrderedDict
#from tensorflow.keras.utils import to_categorical
from vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
import random
from PIL import Image
from tqdm import tqdm
from ssvae.ssvae.vae import VAE


class cnnVAE(VAE):

    def __init__(self,image_size,channels,z_dim,filters=None,lr=0.0002,c=0.2, num_convs=None,num_fc=None):
        try:
            self.num_hiddens
            self.num_res_hiddens
            self.embedding_dim
        except:
            self.num_hiddens=256
            self.num_res_hiddens=32
            self.embedding_dim=256
        super().__init__(image_size,channels,z_dim,filters,lr,c, num_convs,num_fc)


    def _loss_init(self,inputs,outputs):

        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)
        self.latent_loss= -tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma)))

        self.loss=self.reconstr_loss+ self.latent_loss * self.c
        self.losses=[self.loss,self.reconstr_loss,self.latent_loss]
        self.losses_labels=["Total","Recnstr","Latent"]

        #tf.summary.scalar("total_loss", self.loss)
        #tf.summary.scalar("latent_loss", self.latent_loss)
        #tf.summary.scalar("reconstr_loss", self.reconstr_loss)

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

            conv_shape=int(np.prod(last_conv.shape[1:]))
            conv_img_shape=int(np.prod(last_conv.shape[1:-1]))
            #print(last_conv.shape[1:-1])
            #last_conv=tf.transpose(last_conv, [0,3,1,2])
            #flatten=tf.reshape(last_conv,(-1,embedding_dim,conv_img_shape))
            flatten=tf.reshape(last_conv,(-1,conv_shape))
            print(flatten.shape)
            #dense layers
            fc=flatten
            #for i in range(2):
            #    fc=tf.layers.Dense(1024,activation=tf.nn.relu,name="enc_dense_%d"%i)(fc)
            #    self.fc.append(fc)

            
        return fc

    def _decoder_init(self,fc):
        with tf.variable_scope("decoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
            last_conv=self.conv_layers[-1]
            #conv_shape=int(np.prod(last_conv.shape[2:]))
            conv_shape=int(np.prod(last_conv.shape[1:]))
            #conv_img_shape=int(np.prod(last_conv.shape[1:-1]))

            #fc=tf.layers.Dense(1024,activation=tf.nn.relu,name="dec_dense")(fc)
            fc=tf.layers.Dense(conv_shape,activation=tf.nn.relu,name="dec_dense")(fc)
            
            # convert to a 3d tensor from 2d dense
            #conv_reshape=(-1,last_conv.shape[],deconv_size,self.filters)

            conv_reshape=(-1,int(last_conv.shape[1]),int(last_conv.shape[2]),int(last_conv.shape[3]))
            #conv_reshape=(-1,int(last_conv.shape[1]),int(last_conv.shape[2]),embedding_dim)

            #print(conv_reshape)
            #fc=tf.transpose(fc, [0,2,1])
            #print(fc.shape)

            reshaped=tf.reshape(fc,conv_reshape)

            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="to_vq"))(reshaped)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=tf.nn.relu(second_res) # resnet v1
            i=0
            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (conv)
            i+=1
            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)
            
        return tf.reshape(last_layer,[-1,self.image_size,self.image_size,self.channels,256])

    def _z_init(self,fc_layer):
        self.mu=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        self.sigma=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        self.z=VAE.sample(self.mu,self.sigma)

    def build_model(self):

        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.channels], name="x_input")

        last_fc=self._encoder_init(self.X)

        self._z_init(last_fc)

        last_layer=self._decoder_init(self.z)
        self.last_layer=last_layer
        #display layer ONLY
        self.display_layer=tf.cast(tf.math.argmax(tf.nn.softmax(last_layer,name="output"),axis=-1),tf.int32)#tf.clip_by_value(last_layer+0.5,0,1)#
        self.inputs=tf.cast((self.X+0.5)*255,tf.int32)
        hstack=tf.cast(tf.concat(([self.display_layer,self.inputs]),axis=1),tf.float32)

        tf.summary.image("reconstruction",hstack)

        # flatten the inputs
        self.inputs=tf.reshape(self.inputs,(-1,self.channels*self.image_size**2), name="inputs")

        # flatten the outputs
        self.outputs=tf.reshape(last_layer,(-1,self.channels*self.image_size**2,256),name="outputs")
        self._loss_init(self.inputs,self.outputs)
        self._train_init()

    def _train_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train = [self.optimizer.minimize(self.loss)]

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps), name="z")
        return z

    def reconstruct(self,rec_imgs,plot=False):
        results=self.sess.run(self.display_layer,feed_dict={self.X:rec_imgs})
        if plot:
            plot_reconstruction(rec_imgs+0.5,results)
        return results


    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict
    def read_batch(self,paths):
        imgs=[]
        for img in paths:
            _img=Image.open(img).convert(self.mode)
            imgs.append(np.array(_img.resize((self.image_size,self.image_size))))
        
        imgs=np.array(imgs)/255-0.5
        return imgs

    def partial_fit(self,X,X_test=None, batch_size=64):
        #indices=np.arange(X.shape[0])
        #random.shuffle(indices)
        #X=X[indices]
        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_images=self.read_batch(X_batch)
                loss,_=self.sess.run([self.loss]+[self.train],feed_dict=self.get_feed_dict(X_images))
                t.set_description("Loss %.2f"%loss)
        X_images=self.read_batch(X[:batch_size])
        train_out=self.sess.run([loss for loss in self.losses],
                                feed_dict=self.get_feed_dict(X_images))

        # if a test is given calculate test loss
        if(X_test is not None):
            #test_indices=np.arange(X.shape[0])
            #random.shuffle(test_indices)
            np.random.shuffle(X_test)
            #X_test=X_test[test_indices]
            X_images=self.read_batch(X_test[:batch_size])
            test_out=self.sess.run([loss for loss in self.losses],
                                   feed_dict=self.get_feed_dict(X_images))
        else:
            test_out=[.0]*len(self.losses)

        return train_out, test_out


    def fit(self,X,X_test=None,epochs=20,batch_size=64, plot=True, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]
        rec_monitor=[]


        x_indices=list(range(X.shape[0]))
        random.shuffle(x_indices)
        #train_batch_sum,_=shuffle_X_y(X,None)
        train_batch_sum=self.read_batch(X[x_indices[:10]])
        writer_test=None
        if not X_test is None:
            x_test_indices=list(range(X_test.shape[0]))
            random.shuffle(x_test_indices)
            #test_batch_sum,_=shuffle_X_y(X_test,None)
            test_batch_sum=self.read_batch(X_test[x_test_indices[:10]])
            writer_test = tf.summary.FileWriter(log_dir+"/test",self.sess.graph) if log_dir else None
        writer_train = tf.summary.FileWriter(log_dir+"/train",self.sess.graph) if log_dir else None

        for epoch in range(epochs):
            train_out,test_out=self.partial_fit(X,X_test, batch_size)

            train_monitor.append([("epoch",epoch)]+list(zip(self.losses_labels,train_out)))
            test_monitor.append([("epoch",epoch)]+list(zip(self.losses_labels,test_out)))
            rec_monitor.append(self.reconstruct(train_batch_sum,plot=plot))

            if writer_train!=None:
                summary=self.sess.run(self.summary_op, feed_dict={self.X:train_batch_sum})
                writer_train.add_summary(summary, epoch)
                writer_train.flush()
                if writer_test!=None:
                    summary=self.sess.run(self.summary_op, feed_dict={self.X:test_batch_sum})
                    writer_test.add_summary(summary, epoch)
                    writer_test.flush()

            if(verbose):
                # print the last one only
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            #ignore epochs by indexing 1:
            plot_loss(np.array(train_monitor)[:,1:],"train")
            plot_loss(np.array(test_monitor)[:,1:],"test")
        return train_monitor,test_monitor,rec_monitor
