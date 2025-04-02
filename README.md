# UniCon: Universum-inspired Supervised Contrastive Learning


## Train

**(1) UniCon EfficientNet**
```
python main_unicon_efficient_net.py --batch_size 64
  --learning_rate 0.05 --temp 0.7
  --cosine --warm
```

**(2) UniCon Resnet-50**
```
python main_unicon.py --batch_size 64
  --learning_rate 0.05 --temp 0.7
  --cosine --warm
```

**(3) Cross Entropy Baseline**
```
python main_baseline.py --batch_size 64
  --model [rec_cnn, lstm_cnn]
  --learning_rate 0.05
  --cosine --warm
```

**(4) Cross Entropy Resnet-50**
```
python main_ce.py --batch_size 128
  --learning_rate 0.8
  --cosine
```

## Transfer learning
**(1) Transfer CAN-ML E-UniCon 20 epochs**
```
python transfer.py     
  --trained_model_path ./save/CAN-ML_models
  --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed
  --version v1
  --n_classes 10
```

**(1) Transfer CAN-ML UniCon Resnet 20 epochs**
```
python transfer_resnet.py     
  --trained_model_path ./save/CAN-ML_models
  --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/size_64_10/TFRecord_w64_s32/2
  --version v1
  --n_classes 10

python transfer_resnet.py --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_mixup_lambda_0.5_trial_can_ml_uni_resnet_cosine_warm --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/size_64_10/TFRecord_w64_s32/2 --version v1 --n_classes 10 > ./save/cmd_save/transfer_unicon_resnet_2011.log

```

**(2) Transfer CAN-ML CE Resnet 20 epochs**
```
  python transfer_ce.py 
    --trained_model_path ./save/CAN-ML_models 
    --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed
    --version v1_CE_resnet 
    --n_classes 10
```

**(3) Transfer CAN-ML LSTM-CNN 20 epochs**
```
python transfer_baseline.py  
  --model lstm_cnn 
  --trained_model_path ./save/CAN-ML_models
  --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed
  --version v1_lstm_cnn
  --n_classes 10
```

**(4) Transfer CAN-ML Rec-cnn 20 epochs**
```
python transfer_baseline.py  
  --model rec_cnn 
  --trained_model_path ./save/CAN-ML_models
  --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed
  --version v1_rec_cnn
  --n_classes 10
```

## Preprocessing and Train, Test split
**(1) Preprocessing CAN_ML dataset**
```
python3 preprocessing_can_ml.py 
  --window_size=64 
  --strided=32
```

**(2) Train/Val Split**
```
python3 train_test_split_all.py 
  --data_path ./data/can-ml/2017-subaru-forester/preprocessed 
  --window_size 64 
  --strided 32 
  --rid 2
```

## Reference

