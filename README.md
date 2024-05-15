```shell
srun -J "interactive" --cpus-per-task=4 --mem=8G --gres=gpu:2 -w cs-venus-06 --pty zsh  # request 2 GPUs on cs-venus-06

git clone https://github.com/eamonn-zh/ddp_test.git  # pull the code

pip install torch torchvision torchaudio  # install the latest pytorch

python -m torch.distributed.launch --nproc_per_node=2  test.py  # run the code using 2 GPUs
```
