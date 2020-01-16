##########################
## PreProcess
##########################

# Sample draw the fonts and save to paired_images, about 10-20 mins
PYTHONPATH=. python font2img.py


##########################
## Train and Infer
##########################

# Train the model
PYTHONPATH=. python train.py --experiment_dir=experiments \
                --experiment_id=0 \
                --batch_size=64 \
                --lr=0.001 \
                --epoch=40 \
                --sample_steps=50 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15

# Infer
PYTHONPATH=. python infer.py --model_dir=experiments/checkpoint/experiment_0_batch_32 \
                --batch_size=32 \
                --source_obj=experiments/data/val.obj \
                --embedding_ids=0 \
                --save_dir=save_dir/

##########################
## Finetune
##########################

# Generate paired images for finetune
PYTHONPATH=. python font2img_finetune.py


# Train/Finetune the model
PYTHONPATH=. python train.py --experiment_dir=experiments_finetune \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=10 \
                --sample_steps=1 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --freeze_encoder_decoder=1 \
                --optimizer=sgd \
                --fine_tune=67 \
                --flip_labels=1

PYTHONPATH=. python infer.py --model_dir=experiments_finetune/checkpoint/experiment_0 \
                --batch_size=32 \
                --source_obj=experiments_finetune/data/val.obj \
                --embedding_id=67 \
                --save_dir=save_dir/



PYTHONPATH=. python infer_by_text.py --model_dir=experiments_finetune/checkpoint/experiment_0 \
                --batch_size=32 \
                --embedding_id=67 \
                --save_dir=save_dir/