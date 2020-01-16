# GoogleML-Calligraphy

* [Overview](#overview)
* [Calligraphy](#Calligraphy)
* [Network](#Network)
* [About](#About)
* [Contributing](#contributing)

## overview
书法是我国艺术文化瑰宝
![alt network](assets/demo1.jpg)
## Calligraphy

## Network Structure
### Original Model
![alt network](assets/network.png)

The network structure is based off pix2pix with the addition of category embedding and two other losses, category loss and constant loss, from [AC-GAN](https://arxiv.org/abs/1610.09585) and [DTN](https://arxiv.org/abs/1611.02200) respectively.

### Updated Model with Label Shuffling

![alt network](assets/network_v2.png)

## About

我太难了