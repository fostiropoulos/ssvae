import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
import random

class VAE:

    def __init__(self,image_size,z_dim,filters=32,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.image_size=image_size
        self.filters=32
        self.z_dim=z_dim
        self.num_convs=num_convs
        self.num_fc=num_fc
        self.lr=0.002
        self.c=1
        self.log_step=0

        # TF variables
        self.X=None
        self.mu=None
        self.sigma=None
        self.losses=[]
        self.fc=[]
        self.conv_layers=[]
        self.dec_fc=[]
        self.deconv_layers=[]
        self.output_layer=None
        self.inputs=None
        self.outputs=None
        self.summary_op=None
        self.build_model()
        self.start_session()

    def start_session(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()

    def save(self,file):
        saver = tf.train.Saver()
        saver.save(self.sess, file)

    def load(self,file):
        self.build_model()
        self.start_session()
        saver = tf.train.import_meta_graph('%s.meta'%file)
        saver.restore(self.sess, file)

    def _init_z(self,fc_layer):
        self.mu=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        self.sigma=tf.layers.Dense(self.z_dim,activation=None)(fc_layer)
        return VAE.sample(self.mu,self.sigma)

    def _init_loss(self,inputs,outputs):

        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))
        self.latent_loss= -tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma)))

        tf.summary.scalar("latent_loss", self.latent_loss)
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        total_loss=self.reconstr_loss + self.latent_loss * self.c
        tf.summary.scalar("total_loss", total_loss)

        self.losses=[self.reconstr_loss,self.latent_loss,total_loss]

        return total_loss

    def build_model(self):

        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,1], name="x_input")

        # conv layers
        conv=self.X
        for i in range(self.num_convs):
            conv=(tf.keras.layers.Conv2D(self.filters,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            self.conv_layers.append(conv)

        #flatten
        last_conv=self.conv_layers[-1]
        conv_shape=int(np.prod(last_conv.shape[1:]))
        flatten=tf.reshape(last_conv,(-1,conv_shape))

        #dense layers
        fc=flatten
        for i in range(self.num_fc):
            fc=tf.layers.Dense(256,activation=tf.nn.relu,name="enc_dense_%d"%i)(fc)
            self.fc.append(fc)
        last_fc=self.fc[-1]
        self.z=self._init_z(last_fc)

        fc=self.z
        for i in range(self.num_convs-1):
            fc=tf.layers.Dense(256,activation=tf.nn.relu,name="dec_dense_%d"%i)(fc)
            self.fc.append(fc)

        fc=tf.layers.Dense(conv_shape,activation=tf.nn.relu,name="dec_dense_%d"%(i+1))(fc)
        self.fc.append(fc)

        # convert to a 3d tensor from 2d dense
        conv_reshape=[-1]+list(conv.shape[1:])
        reshaped=tf.reshape(fc,conv_reshape)

        # deconvolutions to original shape
        deconv=reshaped
        for i in range(3):
            deconv = tf.keras.layers.Conv2DTranspose( self.filters, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%i) (deconv)
            self.deconv_layers.append(deconv)
        last_layer = tf.keras.layers.Conv2DTranspose(1, 4, strides=(2, 2), padding="same",activation=None,name="dec_deconv_%d"%(i+1))(deconv)

        #display layer ONLY
        self.output_layer=tf.nn.sigmoid(last_layer,name="output")
        hstack=tf.concat(([self.output_layer,self.X]),axis=1)

        tf.summary.image("reconstruction",hstack)

        # flatten the inputs
        self.inputs=tf.reshape(self.X,(-1,self.image_size**2), name="inputs")

        # flatten the outputs
        self.outputs=tf.reshape(last_layer,(-1,self.image_size**2),name="outputs")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.loss=self._init_loss(self.inputs,self.outputs)
        self.train = self.optimizer.minimize(self.loss)

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps), name="z")
        return z

    def reconstruct(self,rec_imgs,plot=False):
        results=self.sess.run(self.output_layer,feed_dict={self.X:rec_imgs})
        if plot:
            plot_reconstruction(rec_imgs,results)
        return results


    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict

    def partial_fit(self,X,X_test=None, batch_size=64):
        indices=np.arange(X.shape[0])
        random.shuffle(indices)
        X=X[indices]

        num_batches=X.shape[0]//batch_size
        for i in range(num_batches):
            X_batch=X[i*batch_size:(i+1)*batch_size]
            self.sess.run(self.train,feed_dict=self.get_feed_dict(X_batch))

        random.shuffle(indices)
        X=X[indices]

        train_out=self.sess.run([loss for loss in self.losses],
                                feed_dict=self.get_feed_dict(X[:batch_size]))
        # if a test is given calculate test loss
        if(X_test is not None):
            test_indices=np.arange(X.shape[0])
            random.shuffle(indices)
            X_test=X[indices]
            test_out=self.sess.run([loss for loss in self.losses],
                                   feed_dict=self.get_feed_dict(X_test[:batch_size]))
        else:
            test_out=[.0,.0,.0,.0]

        return train_out, test_out



    def z_to_X(self,z):
        return self.sess.run(self.output_layer,feed_dict={self.z:z}).reshape(-1,self.feature_size,self.feature_size)

    def X_to_z(self,X):
        return self.sess.run(self.z,feed_dict={self.X:X})

    def fit(self,X,X_test=None,epochs=20,batch_size=64, plot=True, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]
        out_names=["total","reconstruction","latent"]
        rec_monitor=[]


        train_batch,_=shuffle_X_y(X,None)
        train_batch=train_batch[:10]
        writer_test=None
        if X_test:
            test_batch,_=shuffle_X_y(X_test,None)
            test_batch=test_batch[:10]

            writer_test = tf.summary.FileWriter(log_dir+"/test",self.sess.graph) if log_dir else None
        writer_train = tf.summary.FileWriter(log_dir+"/train",self.sess.graph) if log_dir else None

        for epoch in range(epochs):
            train_out,test_out=self.partial_fit(X,X_test, batch_size)

            train_monitor.append([("epoch",epoch)]+list(zip(out_names,train_out)))
            test_monitor.append([("epoch",epoch)]+list(zip(out_names,test_out)))
            rec_monitor.append(self.reconstruct(train_batch,plot=plot))

            if writer_train!=None:
                print(epoch)
                summary=self.sess.run(self.summary_op, feed_dict={self.X:train_batch})
                writer_train.add_summary(summary, epoch)
                if writer_test!=None:
                    summary=self.sess.run(self.summary_op, feed_dict={self.X:test_batch})
                    writer_test.add_summary(summary, epoch)
                #hstack=np.squeeze(np.hstack([rec_monitor[-1],rec_imgs]))
                #pad=np.pad(hstack,((0,),(2,),(1,)),constant_values=(np.max(hstack),))
                #out=np.hstack(pad)

                #image=np.expand_dims(np.expand_dims(out,axis=-1),axis=0)
                #image=np.expand_dims(hstack,axis=-1)
                #merged_summary=tf.summary.merge_all()
                #summary=self.sess.run(self.image_summaries,feed_dict={self.image_summaries_placeholder:image})
                #self.writer.add_summary(summary, self.log_step)
            if(verbose):
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            #ignore epochs
            plot_loss(np.array(train_monitor)[:,1:],"train")
            plot_loss(np.array(test_monitor)[:,1:],"test")
        return train_monitor,test_monitor,rec_monitor
