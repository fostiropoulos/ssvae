import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from ssvae.ssvae.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vae import VAE

class DiscreteVAE(VAE):

    def __init__(self,feature_size, attr_list,attr_sizes,filters=32,lra=0.002,l_c=1,s_c=100):
        super().__init__(feature_size,attr_list,attr_sizes,filters,lr,l_c,s_c)


    def save_model(self,file):
        saver = tf.train.Saver()
        saver.save(self.sess, file)
    """
    def load_model(self,file):
        tf.reset_default_graph()
        self.sess=tf.Session()
        saver = tf.train.import_meta_graph('%s.meta'%file)
        saver.restore(self.sess, file)
    """

    def create_model(self,feature_size,model_type,attributes, filters=32,lr=0.002,l_c=1,s_c=100):
        z_dim=len(self.attributes)
        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.float32,[None,feature_size,feature_size,1])
        #bool, if true calculates the supervised latent loss
        self.is_supervised=tf.placeholder(tf.bool)
        # y for latent supervision loss
        Y={}



        conv=(tf.keras.layers.Conv2D(filters,4,strides=(2,2),padding="same", activation=tf.nn.relu))(self.X)
        conv=(tf.keras.layers.Conv2D(filters,4,strides=(2,2),padding="same", activation=tf.nn.relu))(conv)
        conv=(tf.keras.layers.Conv2D(filters,4,strides=(2,2),padding="same", activation=tf.nn.relu))(conv)
        conv=(tf.keras.layers.Conv2D(filters,4,strides=(2,2),padding="same",activation=tf.nn.relu))(conv)
        conv_shape=int(np.prod(conv.shape[1:]))

        flatten=tf.reshape(conv,(-1,conv_shape))
        fc_1=tf.layers.Dense(256,activation=tf.nn.relu)(flatten)
        fc_2=tf.layers.Dense(256,activation=tf.nn.relu)(fc_1)



        # l2 supervision
        if model_type=="l2":
            self.Y=tf.placeholder(tf.float32,[None,z_dim])
            self.mu=tf.layers.Dense(z_dim,activation=None)(fc_2)
            self.sigma=tf.layers.Dense(z_dim,activation=None)(fc_2)
            self.z=SupervisedVAE.sample(self.mu,self.sigma)
        else:

            self.mu={}
            self.sigma={}
            self.Y={}
            z_out={}
            for attr in self.attributes:
                self.Y[attr]=tf.placeholder(tf.int8,[None,self.attributes[attr]])
                self.mu[attr]=tf.layers.Dense(self.attributes[attr],activation=None)(fc_2)
                self.sigma[attr]=tf.layers.Dense(self.attributes[attr],activation=None)(fc_2)
                z_sampled=SupervisedVAE.sample(self.mu[attr],self.sigma[attr])

                if model_type=="sampled":
                    #z_sampled
                    z_out[attr]=z_sampled
                elif model_type=="softmax":
                    #z_softmax
                    z_out[attr]=tf.nn.softmax(z_sampled)
                elif model_type=="argmax":
                    #z_argmax
                    z_out[attr]=tf.cast(tf.math.argmax(tf.nn.softmax(z_sampled),-1), tf.float32)
                elif model_type=="argmax_norm":
                    #z_argmax_norm
                    z_out[attr]=tf.cast(tf.math.argmax(tf.nn.softmax(z_sampled),-1), tf.float32)/tf.cast(z_sampled.shape[-1],tf.float32)
                else:
                    raise Exception("Invalid model_type: %s"%model_type)
            if model_type=="sampled" or model_type=="softmax":
                self.z=tf.concat([z_out[_attr] for _attr in self.attributes],-1,name=model_type)

            else:
                self.z=tf.stack([z_out[_attr] for _attr in self.attributes],-1,name=model_type)

        print("Shape of Z: %s"%self.z.shape)

        fc_3=tf.layers.Dense(256,activation=tf.nn.relu)(self.z)

        # experiment with skip connections
        #fc_4=tf.math.add(fc_2,fc_3)

        fc_4=tf.layers.Dense(conv_shape,activation=tf.nn.relu)(fc_3)

        # convert to a 3d tensor from 2d dense
        conv_reshape=[-1]+list(conv.shape[1:])
        fc_4=tf.reshape(fc_4,conv_reshape)

        # deconvolutions to original shape
        deconv = tf.keras.layers.Conv2DTranspose( filters, 4, strides=(2, 2), padding="same", activation=tf.nn.relu) (fc_4)
        deconv = tf.keras.layers.Conv2DTranspose( filters, 4, strides=(2, 2), padding="same",activation=tf.nn.relu) (deconv)
        deconv = tf.keras.layers.Conv2DTranspose( filters, 4, strides=(2, 2), padding="same", activation=tf.nn.relu) (deconv)
        last_layer = tf.keras.layers.Conv2DTranspose(1, 4, strides=(2, 2), padding="same",activation=None)(deconv)

        #display layer ONLY
        self.output_layer=tf.nn.sigmoid(last_layer)


        # flatten the inputs
        inputs=tf.reshape(self.X,(-1,feature_size*feature_size))

        # flatten the outputs
        outputs=tf.reshape(last_layer,(-1,feature_size*feature_size))

        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(reconstr_loss,1))


        latent_loss=0
        supervised_loss=0

        if model_type=="l2":
            supervised_loss= tf.cond(self.is_supervised,lambda: tf.reduce_mean(tf.nn.l2_loss(self.Y-self.mu)),lambda:0.)
            latent_loss = tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma)))
        else:
            for a in attributes:
                latent_loss += tf.reduce_mean(0.5 * (1 + self.sigma[a] - self.mu[a]**2 - tf.exp(self.sigma[a])))
                supervised_loss +=  tf.cond(self.is_supervised,lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.mu[a],labels=self.Y[a])),lambda:0.)

        self.latent_loss=latent_loss
        self.supervised_loss=supervised_loss

        self.loss = self.reconstr_loss - self.latent_loss * l_c + self.supervised_loss*s_c
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = self.optimizer.minimize(self.loss)

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps))
        return z

    def reconstruct(self,rec_imgs,plot=False):
        results=self.sess.run(self.output_layer,feed_dict={self.X:rec_imgs})
        if plot:
            plot_reconstruction(rec_imgs,results)
        return results


    def get_feed_dict(self,X_batch,Y_batch, is_supervised):
        feed_dict={self.X:X_batch,self.is_supervised:is_supervised}

        for j,a in enumerate(self.attributes):
            feed_dict[self.Y[a]]=np.zeros((X_batch.shape[0],self.attributes[a])) if not is_supervised else to_categorical(Y_batch[:,j],num_classes=self.attributes[a])
        return feed_dict


    def partial_fit(self,X,y,X_test=None,y_test=None, batch_size=64, is_supervised=True):
        if is_supervised and y is None:
            raise Exception("You must provide a y if you are supervising the latents")

        X,y=shuffle_X_y(X,y)

        num_batches=X.shape[0]//batch_size
        for i in range(num_batches):
            X_batch=X[i*batch_size:(i+1)*batch_size]
            Y_batch=y[i*batch_size:(i+1)*batch_size]
            self.sess.run(self.train,feed_dict=self.get_feed_dict(X_batch,Y_batch,is_supervised))

        X,y=shuffle_X_y(X,y)

        train_out=self.sess.run([self.loss,self.reconstr_loss,self.latent_loss,self.supervised_loss],
                                feed_dict=self.get_feed_dict(X[:batch_size],y[:batch_size],is_supervised))

        # if a test is given calculate test loss
        if(X_test is not None):
            X_test,y_test=shuffle_X_y(X_test,y_test)
            test_supervised=False
            if y_test is not None:
                test_supervised=True
            test_out=self.sess.run([self.loss,self.reconstr_loss,self.latent_loss,self.supervised_loss],
                                   feed_dict=self.get_feed_dict(X_test[:batch_size],y_test[:batch_size],test_supervised))
        else:
            test_out=[.0,.0,.0,.0]

        return train_out, test_out

    def index_to_z_softmax(self,z):
        return np.concatenate([to_categorical(_z,num_classes=list(self.attributes.values())[i]) for i,_z in enumerate(z.transpose())],axis=-1)
    def z_to_X(self,z):


        if self.model_type=="sampled" or self.model_type=="softmax":
            z=self.index_to_z_softmax(z)

        return self.sess.run(self.output_layer,feed_dict={self.z:z}).reshape(-1,self.feature_size,self.feature_size)

    def X_to_z(self,X):
        if(self.model_type=="softmax"):
            z=self.sess.run(self.z,feed_dict={self.X:X})
            attr_values=np.array(list(self.attributes.values()))
            return np.stack([np.argmax(z[:,np.sum(attr_values[:i]):np.sum(attr_values[:i])+val],-1) for i,val in enumerate(attr_values)],axis=-1)

    def X_to_z_softmax(self,X):
        if(self.model_type=="softmax"):
            z=self.sess.run(self.z,feed_dict={self.X:X})
            return z
    def fit(self,X,y,X_test=None,y_test=None,rec_imgs=None,epochs=20,batch_size=64,is_supervised=True, plot=True, verbose=True):
        train_monitor=[]
        test_monitor=[]
        out_names=["total","recnstr","ltnt","sprvised"]
        rec_monitor=[]
        if(rec_imgs==None):
            rec_imgs,_=shuffle_X_y(X,y)
            rec_imgs=rec_imgs[:10]

        for epoch in range(epochs):
            train_out,test_out=self.partial_fit(X,y,X_test,y_test, batch_size, is_supervised)
            train_monitor.append([("epoch",epoch)]+list(zip(out_names,train_out)))
            test_monitor.append([("epoch",epoch)]+list(zip(out_names,test_out)))

            rec_monitor.append(self.reconstruct(rec_imgs,plot=plot))

            if(verbose):
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            #ignore epochs
            plot_loss(np.array(train_monitor)[:,1:],"train")
            plot_loss(np.array(test_monitor)[:,1:],"test")
        return train_monitor,test_monitor,rec_monitor
