CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python infer_by_text.py --model_dir=experiments/checkpoint/experiment_0 \
                                                                         --batch_size=32 \
                                                                         --embedding_id=67 \
                                                                         --save_dir=save_dir/


