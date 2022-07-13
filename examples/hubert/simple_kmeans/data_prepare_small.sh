#!/usr/bin/env bash
model_dir=/modelblob/users/v-chengw/librispeech_model/fairseq/pretrain/fb_released/hubert_base_ls960.pt

nshard=$1
layer=9
ncluster=500

train_set=small_chunks
tsv_dir=/modelblob/users/v-chengw/data/librispeech/manifest/resource
feat_dir=/home/t-chewang/hubert_feature_layer${layer}
label_dir=${tsv_dir}/km_label_hubert_iter2_layer9/



# Step 1 dump mfcc features
for rank in $(seq 0 7); do
	python examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${train_set} ${model_dir} ${layer}  ${nshard} ${rank} ${feat_dir} 

done
mkdir -p ${label_dir}
for rank in $(seq 0 7); do
	python  examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} $train_set ${label_dir}/k${ncluster}.model ${nshard} ${rank} ${label_dir}/k${ncluster}/ 
done
wait



