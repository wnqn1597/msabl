for (( i=0; i<=4; i++ ))
do
  python main.py \
    --task mnist-add \
    --epochs 10 \
    --first_level_epochs 2 \
    --label_smoothing 0.0 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --use_level \
    --loop_list 1 4 3 2 2 2 2 2 2 5 \
    --nbpt_list 1 1 1 1 1 1 1 1 1 2 \
    --seed $i

  python main.py \
    --task fashion-add \
    --epochs 10 \
    --first_level_epochs 2 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --use_level \
    --loop_list 1 5 4 4 3 3 3 3 3 5 \
    --nbpt_list 1 1 1 1 1 1 2 2 2 4 \
    --seed $i

  python main.py \
    --task cifar-add \
    --epochs 10 \
    --first_level_epochs 2 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_workers 4 \
    --use_level \
    --loop_list 1 5 5 4 4 3 3 3 3 7 \
    --nbpt_list 1 1 1 2 2 4 4 4 4 8 \
    --seed $i
done