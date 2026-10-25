# setting1：
CUDA_VISIBLE_DEVICES=0 python continual_train_MM25_end_new_aug_final_paper.py \
--data-dir /home/xukunlun/DATA/PRID/ \
--logs-dir logs/setting1 \
--setting 1 \
--learn_weight 0.75 \
--AF_weight 0.5 \
--prototypes_per_class 5 \
--n_samples 1 \
--lambda_1 0.8 \
--lambda_2 0.8 
# setting2：
CUDA_VISIBLE_DEVICES=0 python continual_train_MM25_end_new_aug_final_paper.py \
--data-dir /home/xukunlun/DATA/PRID/ \
--logs-dir logs/setting2 \
--setting 2 \
--learn_weight 0.75 \
--AF_weight 0.5 \
--prototypes_per_class 5 \
--n_samples 1 \
--lambda_1 0.8 \
--lambda_2 0.8 

