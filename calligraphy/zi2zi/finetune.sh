if [ $1 == 'sample' ]; then
	PYTHONPATH=. python font2img.py --fonts_dir data/finetune_fonts --sample_dir data/paired_images_finetune --save_dir experiments_finetune/data

elif [ $1 == 'ft' ]; then
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python train.py --experiment_dir=experiments_finetune \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=10 \
                --sample_steps=5 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --freeze_encoder_decoder=1 \
                --optimizer=sgd \
                --fine_tune=0 \
                --flip_labels=1 \

fi
