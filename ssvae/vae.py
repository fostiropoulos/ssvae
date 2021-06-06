import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#from collections import OrderedDict
#from tensorflow.keras.utils import to_categorical
from vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
import random
from PIL import Image
from tqdm import tqdm

class VAE:

    def __init__(self,image_size,channels,z_dim,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.image_size=image_size
        self.channels=channels
        self.filters=32
        self.z_dim=z_dim
        self.num_convs=num_convs
        self.num_fc=num_fc
        self.lr=lr
        self.c=c
        self.mode="RGB" if self.channels==3 else "L"

        # TF variables
        self.X=None
        self.mu=None
        self.sigma=None
        self.losses=[]
        self.fc=[]
        self.conv_layers=[]
        self.dec_fc=[]
        self.deconv_layers=[]
        self.display_layer=None
        self.inputs=None
        self.outputs=None
        self.summary_op=None

        self.build_model()
        self.saver = tf.train.Saver()
        self.start_session()

    def start_session(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()

    def save(self,file):
        self.saver.save(self.sess, file)

    def load(self,file):
        self.saver.restore(self.sess, file)

    def _z_init(self,fc_layer):
        self.mu=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        self.sigma=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        self.z=VAE.sample(self.mu,self.sigma)

    def predict(self,X):
        return self.sess.run(self.display_layer,feed_dict={self.X:X}).reshape(-1,self.image_size,self.image_size,self.channels)

    def _loss_init(self,inputs,outputs):

        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))
        self.latent_loss= -tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma)))

        self.loss=self.reconstr_loss + self.latent_loss * self.c
        self.losses=[self.loss,self.reconstr_loss,self.latent_loss]
        self.losses_labels=["Total","Recnstr","Latent"]

        tf.summary.scalar("total_loss", self.loss)
        tf.summary.scalar("latent_loss", self.latent_loss)
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)

    def _encoder_init(self,conv):
        # conv layers
        with tf.variable_scope("encoder"):

            for i in range(self.num_convs):
                print(conv.shape)

                conv=(tf.keras.layers.Conv2D(self.filters*(i+1),4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
                self.conv_layers.append(conv)
            print(conv.shape)

            #flatten
            last_conv=self.conv_layers[-1]
            conv_shape=int(np.prod(last_conv.shape[1:]))
            flatten=tf.reshape(last_conv,(-1,conv_shape))
            print(flatten.shape)
            #dense layers
            fc=flatten
            for i in range(self.num_fc):
                fc=tf.layers.Dense(512,activation=tf.nn.relu,name="enc_dense_%d"%i)(fc)
                print(fc.shape)
                self.fc.append(fc)

            last_fc=self.fc[-1]
        return last_fc

    def _decoder_init(self,fc):
        with tf.variable_scope("decoder"):

            for i in range(self.num_convs-1):
                fc=tf.layers.Dense(123,activation=tf.nn.relu,name="dec_dense_%d"%i)(fc)
                print(fc.shape)
                self.fc.append(fc)

            last_conv=self.conv_layers[-1]
            conv_shape=int(np.prod(last_conv.shape[1:]))
            deconv_size=max(1,self.image_size//(2**self.num_convs))

            fc=tf.layers.Dense(deconv_size**2*self.filters,activation=tf.nn.relu,name="dec_dense_%d"%(i+1))(fc)
            print(fc.shape)

            self.fc.append(fc)

            # convert to a 3d tensor from 2d dense
            conv_reshape=(-1,deconv_size,deconv_size,self.filters)
            reshaped=tf.reshape(fc,conv_reshape)

            # deconvolutions to original shape
            deconv=reshaped
            num_deconvs=int((np.log2(self.image_size/deconv_size)))

            print(deconv.shape)
            for i in range(num_deconvs-1):
                deconv = tf.keras.layers.Conv2DTranspose( self.filters*(num_deconvs-i), 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%i) (deconv)
                #deconv=tf.add(skips[i+1],deconv)
                print(deconv.shape)
                self.deconv_layers.append(deconv)
            #print(deconv.shape)
            deconv = tf.keras.layers.Conv2DTranspose( self.filters, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (deconv)

            print(deconv.shape)
            self.deconv_layers.append(deconv)

            last_layer_kernel=int((deconv.shape[-2]-self.image_size)+1)
            last_layer =tf.keras.layers.Conv2D(self.channels,last_layer_kernel, strides=(1, 1), padding="valid",activation=None,name="dec_deconv_%d"%(i+2))(deconv)
            #last_layer =tf.keras.layers.Conv2DTranspose(self.channels,4, strides=(2, 2), padding="same",activation=None,name="dec_deconv_%d"%(i+1))(deconv)
            print(last_layer.shape)
        return last_layer

    def build_model(self):

        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.channels], name="x_input")

        last_fc=self._encoder_init(self.X)

        self._z_init(last_fc)

        last_layer=self._decoder_init(self.z)

        #display layer ONLY
        self.display_layer=tf.nn.sigmoid(last_layer,name="output")
        hstack=tf.concat(([self.display_layer,self.X]),axis=1)

        tf.summary.image("reconstruction",hstack)

        # flatten the inputs
        self.inputs=tf.reshape(self.X,(-1,self.channels*self.image_size**2), name="inputs")

        # flatten the outputs
        self.outputs=tf.reshape(last_layer,(-1,self.channels*self.image_size**2),name="outputs")
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
            plot_reconstruction(rec_imgs,results)
        return results


    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict


    def read_batch(self,paths):
        imgs=[]
        for img in paths:
            _img=Image.open(img).convert(self.mode)
            imgs.append(np.array(_img.resize((self.image_size,self.image_size))))
        imgs=np.array(imgs)/255
        return imgs
        
    
    def partial_fit(self,X,X_test=None, batch_size=64):
        indices=np.arange(X.shape[0])

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size

        train_out=[]
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_images=self.read_batch(X_batch)
                loss,_=self.sess.run([self.loss]+[self.train],feed_dict=self.get_feed_dict(X_images))

                losses=self.sess.run([loss for loss in self.losses],
                                        feed_dict=self.get_feed_dict(X_images))
                train_out+=losses

                t.set_description("Loss %s"%losses)
        X_images=self.read_batch(X[:batch_size])


        if(X_test is not None):
            test_indices=np.arange(X.shape[0])
            random.shuffle(indices)
            X_test=X_test[indices]
            X_images=self.read_batch(X_test[:batch_size],mode)
            test_out=self.sess.run([loss for loss in self.losses],
                                   feed_dict=self.get_feed_dict(X_images))
        else:
            test_out=[.0]*len(self.losses)

        return train_out, test_out



    def z_to_X(self,z):
        return self.sess.run(self.display_layer,feed_dict={self.z:z}).reshape(-1,self.image_size,self.image_size)

    def X_to_z(self,X):
        return self.sess.run(self.z,feed_dict={self.X:X})

    def fit(self,X,X_test=None,epochs=20,batch_size=64, plot=True, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]
        rec_monitor=[]

        x_indices=list(range(X.shape[0]))
        random.shuffle(x_indices)
        #train_batch_sum,_=shuffle_X_y(X,None)
        train_batch_sum=self.read_batch(X[x_indices[:10]])
        writer_test=None
        if X_test:
            x_test_indices=list(range(X_test.shape[0]))
            random.shuffle(x_test_indices)
            #test_batch_sum,_=shuffle_X_y(X_test,None)
            test_batch_sum=X_test[x_test_indices[:10]]
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
