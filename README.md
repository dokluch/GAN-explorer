
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dokluch/GAN-explorer/blob/main/GAN-explorer.ipynb)

# GAN-explorer

Simple Google Colab tool to animate [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) pickle files. 

![Timeline UI](/imgs/gan-explorer-ui.png)

Comes with three pre-trained 256x256 models:

 - Metfaces + 2500 artwork portraits, 10000 kimgs
 - Art landscapes, 3500 pictures, 10000 kimgs
 - Flickr mountain photos (thanks Jeff Heaton for [pyimgdata](https://github.com/jeffheaton/pyimgdata) tool)

Feel free to use these models as you wish

# Examples of animations

![Timeline UI](/imgs/faces.gif)
![Timeline UI](/imgs/mountains.gif)

# Usage

At this point, GAN Explorer tool support interpolation only between random seeds in the latent space for speed of discovery.

The interpolation itself is naively normalized to the Euclidean distance between two latent vectors which led to a more satisfying results than a simple linear interpolation. Though still far from perfect.

Just generate random seeds, add them to the timeline and render sequence afterwards. Click on "**<<<**" and "**>>>**" buttons to move between previously generated seeds.
The rendered animation is going to be placed in the **/content/renders** folder.

More latent space exploration tools as well as support for other GANs are coming soon.