for (( i=0; i<=4; i++ ))
do
  python main.py \
    --task mnist-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --num_batch_per_train 1 \
    --seed $i

  python main.py \
    --task fashion-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 2 \
    --seed $i

  python main.py \
    --task cifar-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --lr 2e-4 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 8 \
    --num_workers 4 \
    --seed $i

  python main.py \
    --task mnist-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --num_batch_per_train 1 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task fashion-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 2 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task cifar-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --lr 2e-4 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 8 \
    --num_workers 4 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task mnist-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --num_batch_per_train 1 \
    --use_a3bl \
    --seed $i

  python main.py \
    --task fashion-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 2 \
    --use_a3bl \
    --seed $i

  python main.py \
    --task cifar-add \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --lr 2e-4 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 8 \
    --num_workers 4 \
    --use_a3bl \
    --seed $i
done