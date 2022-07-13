#!/usr/bin/env bash
model_dir=/modelblob/users/v-chengw/librispeech_model/fairseq/pretrain/hubert_base_relative_libri960_iter1_new
nshard=$1
layer=$2
ncluster=$3

stage=2
stop_stage=2
train_set=train_960
tsv_dir=/modelblob/users/v-chengw/data/librispeech/manifest/resource
feat_dir=/home/t-chewang/hubert_feature_layer${layer}
label_dir=${model_dir}/km_label_${layer}_0.5

test_sets="dev_clean dev_other"


# Step 1 dump mfcc features
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	for rank in $(seq 0 $((nshard-1))); do
		CUDA_VISIBLE_DEVICES=$((${rank}%4)) python examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${train_set} ${model_dir}/checkpoint_best.pt ${layer}  ${nshard} ${rank} ${feat_dir} &

	done
	wait

	for subset in ${test_sets}; do
		for rank in $(seq 0 $((nshard-1))); do
			CUDA_VISIBLE_DEVICES=$((${rank}%4)) python  examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${subset} ${model_dir}/checkpoint_best.pt ${layer} ${nshard} ${rank} ${feat_dir} &
		done
		wait
	done
fi

# Step 2 kmeans clustering
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
	mkdir -p ${label_dir}
	python  examples/hubert/simple_kmeans/learn_kmeans.py ${feat_dir} ${train_set} ${nshard} ${label_dir}/k${ncluster}.model ${ncluster} --percent 0.1
fi


# Step 3 dump km labels
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	mkdir -p ${label_dir}
	for rank in $(seq 0 $((nshard-1))); do
		CUDA_VISIBLE_DEVICES=$((${rank}%4)) python  examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} $train_set ${model_dir}/k${ncluster}.model ${nshard} ${rank} ${label_dir}/k${ncluster}/ &
	done
	wait

	for subset in ${test_sets}; do
		for rank in $(seq 0 $((nshard-1))); do
			CUDA_VISIBLE_DEVICES=$((${rank}%4)) python  examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} $subset ${model_dir}/k${ncluster}.model ${nshard} ${rank} ${label_dir}/k${ncluster}/ &
		done
		wait
	done

	for rank in $(seq 0 $((nshard-1))); do
		cat ${label_dir}/k${ncluster}/${train_set}_${rank}_${nshard}.km
	done > ${label_dir}/k${ncluster}/${train_set}.km

	for subset in ${test_sets}; do
		for rank in $(seq 0 $((nshard-1))); do
			cat ${label_dir}/k${ncluster}/${subset}_${rank}_${nshard}.km
		done > ${label_dir}/k${ncluster}/${subset}.km
	done
fi




