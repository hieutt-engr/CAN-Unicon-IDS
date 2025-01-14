**(1) Last run**
```
 -------ADAM---------
  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.001 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_adamw_v2 --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer AdamW --save_freq 2 > ./save/cmd_save/100_epoch_ROAD_AdamW_v2.log
  
 -------SGD---------
  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_sgd --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer SGD --save_freq 1 > ./save/cmd_save/100_epoch_ROAD_SGD.log
```

**(2) CAN 100 epochs - Using CAN ID + CAN DATA > (Binary)**
```
  python3 main_unicon_v2.py --batch_size 64 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 5 --dataset CAN --trial can_v2 --data_folder ./data/Car-Hacking/all_features/v2/TFRecord_w32_
  s32/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN.log &
```

**(9) CAN-ML 200 epochs**
```

  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.0002 --optimizer AdamW --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_200 --data_folder ./data/can-ml/preprocessed/2017-subaru-forester/all_features_v2/TFRecord_w32_s16/2 --epochs 200 > ./save/cmd_save/200_epoch_CAN_ML.log &

  Continue:
  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml --data_folder ./data/can-ml/preprocessed/2017-subaru-forester/all_features/TFRecord_w32_s16/2 --epochs 200 --resume ckpt_epoch_150.pth > ./save/cmd_save/200_epoch_CAN_ML.log &

  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.0005 --optimizer AdamW --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_200 --data_folder ./data/can-ml/preprocessed/2017-subaru-forester/all_features_v2/TFRecord_w32_s16/2 --epochs 200 --resume ckpt_epoch_33.pth > ./save/cmd_save/200_epoch_CAN_ML_resume.log &


```
**(10) CAN-ML 100 epochs 128**
```
  python main_unicon_lstm.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_con_lstm --data_folder ./data/can-ml/2017-subaru-forester/preprocessed/20%_data/TFRecord_w32_s16/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_ConLSTM.log &

  ==== =====
    python main_unicon_efficient_net.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_con_enet --data_folder ./data/can-ml/2017-subaru-forester/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 200 > ./save/cmd_save/200_epoch_CAN_ML_ENET_B4.log &

  ==== =====
    python main_unicon_efficient_net.py --batch_size 64 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_con_enet_b4_64 --data_folder ./data/can-ml/2017-subaru-forester/preprocessed/size_64_10/TFRecord_w64_s32/2 --epochs 200 > ./save/cmd_save/200_epoch_CAN_ML_ENET_B4_64.log &
  ==== =====
    python main_unicon_efficient_net_v2.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_con_enet_b6 --data_folder ./data/can-ml/2017-subaru-forester/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 200 > ./save/cmd_save/200_epoch_CAN_ML_ENET_B6.log &
```


**(10) Transfer CAN-ML Unicon Resnet 20 epochs**
```
  python transfer.py     --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_v2_cosine_warm    --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v4  --n_classes 10 > ./save/cmd_save/transfer_k_shot_25_center_loss.log

  ==== New ====
  python transfer_v2.py     --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_v2_cosine_warm    --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v5  --n_classes 10 > ./save/cmd_save/transfer_fs.log
```

  python transfer.py     --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_v2_cosine_warm    --data_folder ./data/can-ml/2016-chevrolet-silverado/preprocessed/all_features/TFRecord_w32_s16/2 --version v1 --save_path 2016-chevrolet-silverado  --n_classes 10 > ./save/cmd_save/transfer_k_shot_2017.log

**(10) Transfer CAN-ML Unicon Enet 20 epochs**
```
  python transfer.py     --trained_model_path ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_efficient-net_lr_0.05_decay_0.0001_bsz_64_temp_0.07_mixup_lambda_0.5_trial_can_ml_con_enet_b4_64_cosine_warm    --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/size_64_10/TFRecord_w64_s32/2 --version v2  --n_classes 10 > ./save/cmd_save/transfer_enet_2011_v2.log

```

**(10) Transfer CAN-ML CE Resnet 20 epochs**
```
  python transfer_ce.py --trained_model_path ./save/CAN-ML_models/CE/CE_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_trial_can_ml_cosine_augment_warm --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v2_CE --n_classes 10 > ./save/cmd_save/transfer_k_shot_v2_CE.log
```

**(10) Transfer CAN-ML Rec-cnn 20 epochs**
```
  python transfer_random.py  --model rec_cnn   --trained_model_path ./save/CAN-ML_models/BaseLine/UniCon_CAN-ML_rec_cnn_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_rec_cnn_cosine_warm   --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v1_Rec_cnn --n_classes 10 > ./save/cmd_save/transfer_k_shot_v1_Rec_cnn.log
 ```

**(10) Transfer CAN-ML LSTM-CNN 20 epochs**
```
  python transfer_random.py  
        --model lstm_cnn   
        --trained_model_path ./save/CAN-ML_models/BaseLine/UniCon_CAN-ML_rec_cnn_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_rec_cnn_cosine_warm   
        --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v1_lstm_cnn 
        --n_classes 10 > ./save/cmd_save/transfer_k_shot_v1_lstm_cnn.log
 ```

**(10) Transfer CAN-ML LSTM-CNN 20 epochs**
```
  python transfer_random.py  --model vit   --trained_model_path ./save/CAN-ML_models/BaseLine/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can-vit-v2_cosine_warm   --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features/TFRecord_w32_s16/2 --version v1_vit --n_classes 10 > ./save/cmd_save/transfer_k_shot_v1_vit.log
```


python3 preprocessing_can_ml_enet.py --window_size=64 --strided=32
/home/hieutt/UniCon/data/can-ml/2011-chevrolet-impala/preprocessed/all_features_30/TFRecord_w64_s32
python3 train_test_split_all.py --data_path ./data/can-ml/2011-chevrolet-impala/preprocessed/all_features_30 --window_size 64 --strided 32 --rid 2
  /home/hieutt/UniCon/data/can-ml/2011-chevrolet-impala/preprocessed/size_64_10/TFRecord_w64_s32/2
  python main_ce.py --batch_size 64 --learning_rate 0.05 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_ce_64_2011 --data_folder ./data/can-ml/2011-chevrolet-impala/preprocessed/size_64_10/TFRecord_w64_s32/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_CE_64_2011.log &