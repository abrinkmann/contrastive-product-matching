#!/bin/bash

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
export CUDA_VISIBLE_DEVICES=6

#chmod +x wdcproducts80cc20rnd000un/run_pretraining_simclr_clean_roberta.sh
#./wdcproducts80cc20rnd000un/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
#chmod +x wdcproducts80cc20rnd000un/run_pretraining_barlow_clean_roberta.sh
#./wdcproducts80cc20rnd000un/run_pretraining_barlow_clean_roberta.sh 64 5e-5 20

chmod +x wdcproducts80cc20rnd000un/run_pretraining_roberta.sh
./wdcproducts80cc20rnd000un/run_pretraining_roberta.sh 1024 5e-5 0.07 20
./wdcproducts80cc20rnd000un/run_pretraining_roberta.sh 1024 5e-5 0.07 200

chmod +x wdcproducts80cc20rnd000un/run_finetune_general_siamese_frozen_roberta.sh
./wdcproducts80cc20rnd000un/run_finetune_general_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 20
./wdcproducts80cc20rnd000un/run_finetune_general_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 200

#chmod +x wdcproducts80cc20rnd000un/run_finetune_cross_encoder_roberta.sh
#./wdcproducts80cc20rnd000un/run_finetune_cross_encoder_roberta.sh 64 5e-5 50