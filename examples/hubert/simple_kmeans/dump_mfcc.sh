for ((i=0;i<8;i+=1));
do
	python dump_mfcc_feature.py /modelblob/users/v-chengw/data/librispeech/manifest/resource/ dev_clean 8 $i /modelblob/users/v-chengw/data/librispeech/feat/ &
done

