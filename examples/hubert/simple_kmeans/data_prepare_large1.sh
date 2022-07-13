#!/usr/bin/env bash
model_dir=/modelblob/users/v-chengw/librispeech_model/fairseq/pretrain/fb_released/hubert_base_ls960.pt

nshard=$1
layer=9
ncluster=500

train_set=chunks_transcribed
tsv_dir=/datablob/users/v-chengw/data/GigaSpeech
feat_dir=/mnt/blobfuse/hubert_feature_layer${layer}
label_dir=${tsv_dir}/km_label_hubert_iter2_layer9/

mkdir -p ${label_dir}



# Step 1 dump mfcc features
function run(){
	cgpu=$1
	rank=$2
	CUDA_VISIBLE_DEVICES=$cgpu python examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${train_set} ${model_dir} ${layer}  ${nshard} ${rank} ${feat_dir} 
	CUDA_VISIBLE_DEVICES=$cgpu python  examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} $train_set ${label_dir}/k${ncluster}.model ${nshard} ${rank} ${label_dir}/k${ncluster}/ 
	rm ${feat_dir}/${train_set}_${rank}_${nshard}.*

}

#for i in $(seq 0 15); do
#	for j in $(seq 0 7); do
#		run $j $(($i*8+$j)) &
#	done
#	wait
#done

#run 0 51
run 1 0


