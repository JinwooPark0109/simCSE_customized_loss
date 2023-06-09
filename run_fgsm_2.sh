#!/bin/bash

root="/home/jwpark/SimCSE/"
loss=original_loss
name=$(echo ${loss}| sed "s/_loss//")
echo ${name}

for eta in 3e-5
do
for number in 5838 16822 19294 17173
do
echo -e "\n------------------------name ${name} seed ${number} -----------------------\n"
python ${root}/train.py \
    --model_name_or_path bert-base-uncased \
    --train_file ${root}/data/wiki1m_for_simcse.txt \
    --output_dir ${root}/result/fgsm_eta_${eta}_${number} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --cl_loss ${loss} \
    --seed ${number} \
    --get_log \
    --use_fgsm \
    --noise_std 1e-3\
    --fgsm_eta ${eta}
done
done