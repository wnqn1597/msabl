for (( i=0; i<=4; i++ ))
do
  python main.py \
    --task mnist-max \
    --pretrain_epochs 20 \
    --loops 10 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_batch_per_train 1 \
    --seed $i

  python main.py \
    --task fashion-max \
    --pretrain_epochs 20 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 1 \
    --seed $i

  python main.py \
    --task cifar-max \
    --pretrain_epochs 20 \
    --lr 5e-4 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 4 \
    --num_workers 4 \
    --seed $i

  python main.py \
    --task mnist-max \
    --pretrain_epochs 20 \
    --loops 10 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_batch_per_train 1 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task fashion-max \
    --pretrain_epochs 20 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 1 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task cifar-max \
    --pretrain_epochs 20 \
    --lr 5e-4 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.2 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 4 \
    --num_workers 4 \
    --use_ablsim \
    --seed $i

  python main.py \
    --task mnist-max \
    --pretrain_epochs 20 \
    --loops 10 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_batch_per_train 1 \
    --use_a3bl \
    --seed $i

  python main.py \
    --task fashion-max \
    --pretrain_epochs 20 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 1 \
    --use_a3bl \
    --seed $i

  python main.py \
    --task cifar-max \
    --pretrain_epochs 20 \
    --lr 5e-4 \
    --loops 20 \
    --epochs 10 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_batch_per_train 4 \
    --num_workers 4 \
    --use_a3bl \
    --seed $i
done
