# FIXR

Code for Faces are Domains (https://ieeexplore.ieee.org/document/10191542)

### Dataset

To train the model, put the RAVDESS training and testing dataset in data folder in following format:

```
data/
├── test
   ├── 1
      ├── angry
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
      ├── disgust
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
      :
      :
      :
      ├── surprise
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
    :
    :
    ├── 20
├── train
   ├── 1
      ├── angry
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
      ├── disgust
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
      :
      :
      :
      ├── surprise
         ├── img_1.png
         ├── img_2.png
         :
         ├── img_n.png
    :
    :
    ├── 20
```

The Entire Code is based on Mammoth Framework (https://github.com/aimagelab/mammoth). All the copyright goes to Mammoth Framework.
