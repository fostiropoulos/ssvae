import  numpy as np
from ssvae.vae import SupervisedVAE
import matplotlib.pyplot as plt
data=np.load("../dsprites.npz")
ids=np.load("../ids.npz")
all_imgs=np.expand_dims(data["imgs"],-1)
all_latent_classes=data["latents_classes"]
rec_ids=ids["test_reconstruct"]
rec_imgs=all_imgs[rec_ids]
sup_ids=ids["supervised_train"]
sup_imgs=all_imgs[sup_ids]
train_indices=ids["train"]

feature_size=all_imgs.shape[-2]
names=["shape","scale","orientation","pos_x","pos_y"]
sizes=[3,6,40,32,32]
my_model=SupervisedVAE(feature_size,model_type="softmax",attr_list=names,attr_sizes=sizes)

train_imgs=all_imgs[train_indices]
latent_classes=all_latent_classes[train_indices,1:]
my_model.fit(train_imgs,latent_classes,train_imgs,latent_classes,epochs=20)
my_model.reconstruct(train_imgs[:10],plot=True)
plt.imshow(my_model.z_to_X([2,0,1,0,0]))
print(my_model.X_to_z(train_imgs[0]))
my_model.save("my_model")
