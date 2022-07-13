km_path=/datablob/users/v-chengw/data/GigaSpeech/km_label_hubert_iter2_layer9/k500/
nshard=128
subset=chunks_transcribed
for rank in $(seq 77 127); do
	cat $km_path/${subset}_${rank}_${nshard}.km
done > $km_path/${subset}2.km

