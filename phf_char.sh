model_path=/datablob/users/v-zhuoyao/model/phf_char_renew_5
data_path=/datablob/users/v-zhuoyao/data/Librispeech/manifest/resource
text_path=/datablob/users/v-zhuoyao/data/Librispeech/text
speech_subset=multi_task
valid_subset=dev_clean
text_subset=text
code_dir=sp_fairseq
labels='["km","ltr","phf"]'


mkdir -p ${model_path}
python train.py \
--distributed-world-size 1 \
--distributed-port 0 --distributed-backend nccl --nprocs-per-node 1 \
/datablob/users/v-zhuoyao/data/Librispeech/manifest/resource \
--save-dir ${model_path} --num-workers 6 --fp16 \
--task joint_hubert_mlm_pretrain \
--criterion joint_hubert_mlm \
--arch hubert_cross_attn_mtl \
--train-subset ${speech_subset} \
--multi-task-subset '["train_960", "train_clean_100"]' \
--multi-task-labels 'km | km phf ' \
--add-ltr-layer \
--add-ctc-after-num-updates 50000 \
--text-data ${text_path}/preprocess/train \
--phone-text-data ${text_path}/preprocess_phone_rep/train \
--valid-subset valid --log-keys '["time_speech","time_text","length_speech", "length_text","time_text_loss", "batch_speech", "batch_text"]' \
--label-dir /datablob/users/v-zhuoyao/data/Librispeech/label/hubert_iter1_layer6_km_label/train_960/k500 \
--labels ${labels} --swap-embedding-ratio 0.3 \
--sample-rate 16000 --max-sample-size 250000 --min-sample-size 32000 --max-tokens 1400000 --update-freq 2 \
--extractor-mode default --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' \
--final-dim 256 --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 \
--attention-dropout 0.1 --dropout 0.1 --feature-grad-mult 0.1  --activation-dropout 0.0 \
--pred-masked-weight 1.0 --pred-nomask-weight 0.0 --loss-weights [10,]  --label-rate 50 --mask-prob 0.8 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
--warmup-updates 64000 --total-num-update 835000 --lr 0.0005 --weight-decay 0.01  --max-update 835000 \
--skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --find-unused-parameters --log-interval 100 \
--log-format json --clip-norm 10.0 --relative-position-embedding --num-buckets 320 --max-distance 800 --validate-interval 10 --validate-interval-updates 10000  \
--seed 1337 --required-batch-size-multiple 8 --save-interval-updates 25000 --keep-interval-updates 16 --no-epoch-checkpoints \
--text-encoder-mask-prob 0.4 --text-encoder-mask-length 15 --shared-encoder-layer 6 --text-encoder-layers 6 --encoder-layers 6  \
--speech-cross-attn-layers '[-1,-1,-1,-1,-1,-1]' --text-cross-attn-layers '[-1,-1,-1,-1,-1,-1]' \
--batch-ratio "1:1:0.003125" --sample-ratio "1:1:0.0150989889736469"


