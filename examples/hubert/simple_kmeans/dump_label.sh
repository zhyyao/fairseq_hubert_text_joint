
nshard=$1
ncluster=$2
stage=0
train_set=train_960
tsv_dir=/modelblob/users/v-chengw/data/librispeech/manifest/resource
model_dir=/modelblob/users/v-chengw/librispeech_model/fairseq/pretrain/hubert_base_relative_libri960_iter1_new/
feat_dir=/home/t-chewang/hubert_feature
label_dir=/modelblob/users/v-chengw/data/librispeech/manifest/resource

test_sets="dev_clean dev_other"


# Step 1 dump mfcc features
if [ ${stage} -le 0 ]; then
	for rank in $(seq 0 $((nshard-1))); do
		CUDA_VISIBLE_DEVICES=$((${rank}%8)) python dump_km_label.py ${feat_dir} $train_set ${model_dir}/k${ncluster}.model ${nshard} ${rank} ${model_dir}/km_label/k${ncluster}/ &
	done
	wait

	for subset in ${test_sets}; do
		for rank in $(seq 0 $((nshard-1))); do
			CUDA_VISIBLE_DEVICES=$((${rank}%8)) python dump_km_label.py ${feat_dir} $subset ${model_dir}/k${ncluster}.model ${nshard} ${rank} ${model_dir}/km_label/k${ncluster}/ &
		done
		wait
	done
fi




