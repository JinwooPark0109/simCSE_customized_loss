#!/bin/bash

root="/home/jwpark/SimCSE/"
loss=original_loss
name=$(echo ${loss}| sed "s/_loss//")
echo ${name}

for number in 19984
do
echo -e "\n------------------------name ${name} seed ${number} -----------------------\n"
python ${root}/train.py \
    --model_name_or_path bert-base-uncased \
    --train_file ${root}/data/wiki1m_for_simcse.txt \
    --output_dir ${root}/result/mul_bern_${number} \
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
    --mul_bern

done