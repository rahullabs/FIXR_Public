

python utils/main.py \
                        --model fixr \
                        --notes NOTES \
                        --dataset ravdess\
                        --lr 0.01 \
                        --n_epochs 10 \
                        --batch_size 32 \
                        --buffer_size 10 \
                        --minibatch_size 32\
                        --alpha 1\
                        --csv_log \
                        --domain_id 1,2
