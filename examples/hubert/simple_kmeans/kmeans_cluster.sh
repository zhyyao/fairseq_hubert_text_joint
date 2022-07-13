for rank in $(seq 0 7);
do
	python dump_km_label.py /modelblob/users/v-chengw/data/librispeech/feat/ dev_clean /modelblob/users/v-chengw/data/librispeech/feat/libri960h_mfcc_kmeans100.model 8 ${rank} /modelblob/users/v-chengw/data/librispeech/kmeans_label/ &
done

