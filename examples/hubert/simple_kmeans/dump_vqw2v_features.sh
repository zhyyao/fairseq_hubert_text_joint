#!/usr/bin/env bash
model_dir=/modelblob/users/v-chengw/librispeech_model/fairseq/pretrain/fb_released/wav2vec_small.pt

nshard=$1

train_set=train_960
tsv_dir=/modelblob/users/v-chengw/data/librispeech/manifest/resource
label_dir=${tsv_dir}/wav2vec2_kmeans_label

test_sets="dev_clean dev_other"
mkdir -p ${label_dir}


# Step 1 dump mfcc features

for rank in $(seq 0 $((nshard-1))); do
	CUDA_VISIBLE_DEVICES=$((${rank}%8)) python examples/hubert/simple_kmeans/dump_wav2vec_label.py ${tsv_dir} ${train_set} ${model_dir} ${nshard} ${rank} ${label_dir} &
done
wait

for subset in ${test_sets}; do
	for rank in $(seq 0 $((nshard-1))); do
		CUDA_VISIBLE_DEVICES=$((${rank}%8)) python  examples/hubert/simple_kmeans/dump_wav2vec_label.py ${tsv_dir} ${subset} ${model_dir} ${nshard} ${rank} ${label_dir} &
	done
	wait
done


for rank in $(seq 0 $(($nshard-1))); do
        cat ${label_dir}/${train_set}_${rank}_${nshard}.vq1
done > ${label_dir}/${train_set}.vq1


for rank in $(seq 0 $(($nshard-1))); do
        cat ${label_dir}/${train_set}_${rank}_${nshard}.vq2
done > ${label_dir}/${train_set}.vq2

for subset in ${test_sets}; do
	for rank in $(seq 0 $(($nshard-1))); do
        	cat ${label_dir}/${subset}_${rank}_${nshard}.vq1
	done > ${label_dir}/${subset}.vq1

	for rank in $(seq 0 $(($nshard-1))); do
        	cat ${label_dir}/${subset}_${rank}_${nshard}.vq2
	done > ${label_dir}/${subset}.vq2
done

