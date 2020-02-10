
## Supervised VAE
The goal of this project is to train a VAE by supervising over the Latent Dimension to create interpretable latent variables that correspond to attributes over the data. We achieve that by disentangling all the dimensions. There are many ways to supervise a VAE and disentangle the latent dimensions. There are methods that consider generic information overlap metrics such as KL-Divergence but fail to create interpretable dimensions. What was attempted in this project was a supervision loss functions over different Z spaces. We add our prior knowledge about the attributes of the data to construct a latent space that we can manipulate.

The total loss is calculated as follows: $Loss=\text{reconstruction loss} + \text{latent loss} \times c_{latent} + \text{supervised loss} \times c_{supervised}$, where $c$ is a hyper-parameter, the coefficient for each respective loss.

The model supports the following Z spaces:
* Z Sampled - Z is the concatenated vector of $z_{attribute}$.   $z_{attribute}$ is generated from the reparametrization trick from 2 distinct $\mu_{attribute}$ and $\sigma_{attribute}$ for each known attribute
* Z Softmax - Z is concatenated outcome of the softmax for each $z_{attribute}$
* Z Argmax - Z is the concatenated scalar that is the outcome of the `tf.argmax` over each $z_{attribute}$
* Z Argmax Norm - Z is the concatenated scalar that is the scaled value from Z Argmax calculated as $\frac{z_{attribute}}{max(y_{attribute})}$
* Z - L2 is the direct outcome from a single $\mu$ and $\sigma$ for which each index represents a distinct attribute (real value) with
model_type : `["sampled","softmax","argmax","argmax_norm","l2"]`

`SupervisedVAE(feature_size,model_type=model_type,attr_list=names,attr_sizes=sizes)`

For the Categorical Z spaces the loss is calculated as the categorical cross entropy on $\mu_{attribute}$ with the one-hot vector from the ground truth `y` from the dataset.

For the L2 Z Space, the loss is calculated as the L2 Distance between $\mu$ and $y$

### Results
* Z Softmax is the most preferable because:
	* Can interpolate with integers e.g. `[0,1,2,3,4]`
	* Is trainable
* Z Argmax variants are not preferable because:
	* Encoder and Decoder train as 2 seperate networks, with latent loss applying only on the encoder and reconstruction loss only on the decoder. Such models can only train with supervised loss since it propagates information between the two networks.
* L2 - is similar to Z Softmax with the downside of real values and worse interpolations (Not distinct)
* Z Sampled - unable to create meaningful interpolations.

### Example

```python
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

train_imgs=all_imgs[sup_ids]
latent_classes=all_latent_classes[sup_ids,1:]
my_model.fit(train_imgs,latent_classes,train_imgs,latent_classes,epochs=20)
my_model.reconstruct(train_imgs[:10],plot=True)
plt.imshow(my_model.z_to_X([2,0,1,0,0]))
print(my_model.X_to_z(train_imgs[0]))
my_model.save("my_model")

```
