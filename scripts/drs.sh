# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 2000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 4000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 6000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 8000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 10000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 20000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py dcgan 40000 &

# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 2000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 4000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 6000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 8000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 10000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 20000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 40000 &

# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 100 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 1000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 2000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 3000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27 python drs-impgan.py wgan 4000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 5000 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 2 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 4 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 8 &
srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 16 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 32 &
# srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 64 &
srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 128 &
srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 256 &
srun -p gpuc --gres=gpu:1 --exclude=guppy27,guppy8 python drs-impgan.py wgan 40000 --dim 512 &
