#!/bin/bash

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
export CUDA_VISIBLE_DEVICES=0

chmod +x walmartamazon/run_pretraining_simclr_clean_roberta.sh
./walmartamazon/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
chmod +x walmartamazon/run_pretraining_barlow_clean_roberta.sh
./walmartamazon/run_pretraining_barlow_clean_roberta.sh 64 5e-5 20

#chmod +x walmartamazon/run_pretraining_clean_roberta.sh
#./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 20
#./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 200

#chmod +x walmartamazon/run_finetune_siamese_frozen_roberta.sh
#./walmartamazon/run_finetune_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 20
#./walmartamazon/run_finetune_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 200

#chmod +x walmartamazon/run_finetune_cross_encoder_roberta.sh
#./walmartamazon/run_finetune_cross_encoder_roberta.sh 64 5e-5 50