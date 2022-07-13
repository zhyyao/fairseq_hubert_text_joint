model_path=/datablob/users/v-zhuoyao/model/finetune/phf_char_renew_5_40w
gen_subset="dev_other"
result_path=${model_path}/decode_ctc/${gen_subset}

mkdir -p ${result_path}
export PYTHONENCODING=UTF-8


python examples/speech_recognition/infer.py /datablob/users/v-zhuoyao/data/Librispeech/manifest/resource/ \
	--task audio_pretraining --nbest 1 \
	--path ${model_path}/checkpoint_best.pt --gen-subset ${gen_subset} --results-path ${result_path} \
	--w2l-decoder viterbi --word-score -1 --sil-weight 0 --criterion ctc --batch-size 1 \
	--dict-path /datablob/users/v-zhuoyao/data/Librispeech/manifest/resource/dict.ltr.txt \
	--post-process letter --quiet


