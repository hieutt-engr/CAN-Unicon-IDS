# UniCon: Universum-inspired Supervised Contrastive Learning


## Running

**(1) UniCon EfficientNet**

```
python main_unicon_efficient_net.py --batch_size 64
  --learning_rate 0.05 --temp 0.7
  --cosine --warm
```
**(2) UniCon Resnet**

```
python main_unicon.py --batch_size 64
  --learning_rate 0.05 --temp 0.7
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


**(8) Preprocessing CAN_ML dataset**
```
python3 preprocessing_can_ml.py --window_size=64 --strided=32 > data_preprocessing_can_ml.txt
```

**(9) Train/Val Split**
```
python3 train_test_split_all.py --data_path ./data/can-ml/2017-subaru-forester/preprocessed --window_size 64 --strided 32 --rid 2
```

## Reference

