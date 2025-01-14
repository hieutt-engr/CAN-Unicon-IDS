# UniCon: Universum-inspired Supervised Contrastive Learning

The intuition of this paper is simple: why don't we assign Mixup samples to a generalized negative class? Just as humans
may perceive, if an animal is half dog and half cat, it is actually of neither species.
<p align="center">
  <img src="pic/intuition.png" width="700">
</p>
With this intuition, our framework is like this:
<p align="center">
  <img src="pic/framework.png" width="700">
</p>


Here we present the implementation of UniCon(ours)[1], SupCon[2], SimClr[3], Xent and SupMix(a hybrid of SupCon and
Un-Mix[4] proposed by us) on Pytorch. We use CIFAR-100 as an example dataset.
[1]: Universum-inspired Supervised Contrastive Learning [paper](https://arxiv.org/abs/2204.10695)  
[2]: Supervised Contrastive Learning [paper](https://arxiv.org/abs/2004.11362)  
[3]: A Simple Framework for Contrastive Learning of Visual Representations [paper](https://arxiv.org/abs/2002.05709)  
[4]: Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation
Learning [paper](https://arxiv.org/abs/2003.05438)


## Comparison

Results on CIFAR-100:
|Method|Architecture|Batch size|Accuracy(%)|
|---|---|---|---|
|Cross Entropy|ResNet-50|1024|74.6|
|SimCLR|ResNet-50|1024|68.4|
|SupCon(baseline)|ResNet-50|1024|76.5|
|UniCon(ours)|ResNet-18|256|79.2|
|UniCon(ours)|ResNet-50|256|81.7|


## Running

We refer you to the [official pytorch implementation of SupCon](https://github.com/HobbitLong/SupContrast) for 
the implementations of SupCon and SimCLR. If you want to run SupCon and SimCLR with our code, you can simply change 
`python main_supcon.py` to `python main_unicon.py` and maintain other parts. We will give instructions on SupCon, 
SupMix and CE classifier. Our samples are running on CIFAR-100 and you can run on CIFAR-10, TinyImageNet and ImageNet
(you need to download the datasets yourself for the latter two) with `--dataset cifar10/tinyimagenet/imagenet`.  

**(1) UniCon**

```
python main_unicon.py --batch_size 256
  --learning_rate 0.05 --temp 0.1
  --cosine --warm
```

You can change Mixup parameter with `--lamda 0.5`. Or you can use CutMix with `--mix cutmix`.    

**(2) SupMix**  
```
python main_supmix.py --batch_size 256
  --learning_rate 0.05 --temp 0.1
  --cosine --warm --beta --lamda 0.5
```
Here `--lamda` is used in distribution Beta(lambda, lambda). You can also set lambda to a constant by reducing
`--beta`.  

**(3) Cross Entropy**

```
python main_ce.py --batch_size 1024
  --learning_rate 0.8
  --cosine
```
You can use `--augment`, `--mixup --alpha 0.5`, `--cutmix --alpha 0.5 --cutmix_prob 0.5` to enable augmentations, 
Mixup and CutMix, respectively. Please note that Mixup and CutMix cannot be applied together. If so, only Mixup is 
used.  

**(4) Linear Evaluation**

```
python main_linear.py --batch_size 512
  --learning_rate 5
  --ckpt /path/to/model.pth
```

**(5) Preprocessing CAN dataset**
```
python3 preprocessing.py --window_size=32 --strided=32 --indir=./data/Car-Hacking --outdir=./data/Car-Hacking> data_preprocessing_can.txt
```

**(6) Preprocessing ROAD Fabrication dataset**

```
CUDA_VISIBLE_DEVICES=2 python3 preprocessing_road.py --window_size=32 --strided=8 --attack_type=road_fab --indir=./data/road/fab_dataset --outdir=./data/road/preprocessed/fab_multi_v2/TFRecord > data_preprocessing_roadfab_new.txt
```

**(7) Preprocessing ROAD Masquerade dataset**

```
CUDA_VISIBLE_DEVICES=0 python3 preprocessing_road.py --window_size=32 --strided=8 --attack_type=road_mas --indir=./data/road/mas_dataset --outdir=./data/road/preprocessed/mas_multi/TFRecord > data_preprocessing_roadfab.txt
```


**(8) Preprocessing CAN_ML dataset**
```
python3 preprocessing_can_ml.py --window_size=32 --strided=16 > data_preprocessing_can_ml_2011.txt


```

**(9) Train/Val Split**

```
python3 train_test_split_all.py --data_path ./data/road/preprocessed/fab_multi_v2  --window_size 32 --strided 8 --rid 2
python3 train_test_split_all.py --data_path ./data/can-ml/preprocessed/2017-subaru-forester/all_features_v2  --window_size 32 --strided 16 --rid 2


python3 train_test_split_all.py --data_path ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features --window_size 32 --strided 16 --rid 2
```

**(10) Train Unicon - ROAD**

```
 python3 main_unicon.py --batch_size 128 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s15/2 --epochs 100 --test_freq 90  > ./save/cmd_save/100_epoch_ROAD.log &
```

```


 python transfer.py     --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_v2_cosine_warm    --data_folder ./data/can-ml/2016-chevrolet-silverado/preprocessed/all_features/TFRecord_w32_s16/2 --version v3 --save_path 2016-chevrolet-silverado  --n_classes 10 > ./save/cmd_save/transfer_k_shot_20.log
 ```

## Reference

```
@article{han2022universum,
  title={Universum-inspired Supervised Contrastive Learning},
  author={Han, Aiyang and Chen, Songcan},
  journal={arXiv preprint arXiv:2204.10695},
  year={2022}
}
```
