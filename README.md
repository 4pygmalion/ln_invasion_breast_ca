### Lymph Node Invasion classification with Multiple Instance Learning(MIL)


### Install
```
$ conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
$ python3 -m pip install opencv-python numpy Pillow pandas tensorflow matplotlib scikit-learn
```

### How to use
1. split patch
```
$ python3 split_patch.py -i data/train_imgs -o data/train_imgs_patch -s 64 -w 128 -c 0.7
$ python3 split_patch.py -i data/test_imgs -o data/test_imgs_patch -s 64 -w 128
```

``` 
// output
├─data
│  ├─test_imgs
│  ├─train_imgs
│  ├─train_imgs_patch
│  │  ├─BC_01_0001
│  │  ├─BC_01_0002
│  │  ├─BC_01_0003
│  │  ├─BC_01_0004
│  │  ├─BC_01_0005
│  │  ├─BC_01_0006
│  │  ├─BC_01_0007
│  │  ├─BC_01_0008
│  │  ├─BC_01_0009
│  │  ├─BC_01_0010
....

```

2. model training
```
$ python3 train_model.py -i data/train_imgs_patch
```


### Inspired by 
- https://github.com/utayao/Atten_Deep_MIL
