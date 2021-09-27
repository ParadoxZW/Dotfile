vim ~/.vimrc
cd ~/distributed-vqa-bert/
cp det3-bert-008.py det3-bert-010.py 
vim det3-bert-010.py 
python det3-bert-010.py 
vim det3-bert-010.py 
python det3-bert-010.py 
vim det3-bert-010.py 
vim det3-bert-009.py 
python det3-bert-010.py 
cd /data-ssd/shaozw/dataset/vqa/raw/
vim ./
python trans.py 
cd -
python det3-bert-010.py 
cd /data/shaozw/checkpoint/vqa-bert/
ls
rm *ch1.pkl
rm *ch2.pkl
rm *ch3.pkl
rm *ch4.pkl
rm *ch5.pkl
rm *ch6.pkl
ls
cd -
vim ./
ls /data1/
ls /
ls ~
cd /data/shaozw/
ls
cd vqa_backup/
ls
ls ckpts/
nvidia-smi
python run.py --RUN=test --MODEL=scan_4 --GPU=3 --DATASET=vqa --SEED=888 --VERSION=scan_bert --CKPT_V=scan_1218
vim ./
ls ckpts/
python run.py --RUN=test --MODEL=scan_4 --GPU=3 --DATASET=vqa --SEED=888 --VERSION=scan_bert --CKPT_V=scan_albert_dev --CKPT_E=13
vim ./
python run.py --RUN=test --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_bert --CKPT_V=scan_albert_dev --CKPT_E=13
cd ..
cd ~
nvidia-smi
cd /data/shaozw/code_from_cyh/
vim ./
cd ~/distributed-vqa-bert/
vim ./
cd utils_vqa/
cp load_data1.py load_data2.py
vim load_data2.py
cd ..
ls
cp det3-bert-011.py mcan-grid-001.py 
vim mcan-grid-001.py 
vim ./
ls /data/cuiyh/features/idgrid_x152-600_1000/
vim ./
python mcan-grid-001.py 
vim ./
python mcan-grid-001.py 
vim ./
python mcan-grid-001.py 
vim ./utils_vqa/load_data2.py 
python mcan-grid-001.py 
vim ./utils_vqa/load_data2.py 
python mcan-grid-001.py 
vim ./utils_vqa/load_data2.py 
vim mcan-grid-001.py 
python mcan-grid-001.py 
vim ./utils_vqa/load_data2.py 
vim mcan-grid-001.py 
vim ./utils_vqa/load_data2.py 
vim mcan-grid-001.py 
python mcan-grid-001.py 
vim mcan-grid-001.py 
python mcan-grid-001.py 
cd ~
cp /data/shaozw/checkpoint/vqa-bert/cnn-bert-004_epoch2.pkl ./
conda activate pyt0rch
python run.py --RUN=train --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_xalbert1
vim ./
ls ckpts/
ls ckpts/ckpt_scan_xalbert
ls ckpts/ckpt_scan_xalbert1
rm -r ckpts/ckpt_scan_xalbert1
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_xalbert2
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_xalbert2
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_xalbert2
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=0 --DATASET=vqa --SEED=888 --VERSION=scan_xalbert2
ls ckpts/
rm -r ckpts/ckpt_scan_xalbert2
rm -r ckpts/ckpt_scan_xalbert
cd ~/distributed-vqa-bert/
cp det3-bert-008.py det3-bert-011.py
vim det3-bert-011.py
cd /data-ssd/shaozw/dataset/vqa/raw/
vim ./
python trans.py 
cd -
vim ./
python det3-bert-011.py 
vim det3-bert-011.py 
python det3-bert-011.py 
vim det3-bert-011.py 
python det3-bert-011.py 
vim det3-bert-011.py 
python det3-bert-011.py 
vim det3-bert-011.py 
python det3-bert-011.py 
vim det3-bert-011.py 
python det3-bert-011.py 
cd /data-ssd/shaozw/dataset/
ls
cd vqa/
ls
cd raw/
ls
vim ./
python trans.py 
cd ~/distributed-vqa-bert/
vim ./
python det3-bert-011.py 
cd /data/shaozw/ensemble/
ls
vim ensemble.py 
ls
cp ensemble.py ensemble1.py 
vim ensemble1.py 
vim ~/.vimrc 
vim ensemble1.py 
mv ensemble1.py ~/distributed-vqa-bert/utils_vqa/
cd ~/distributed-vqa-bert/
vim ensemble1.py 
vim ./
vim /home/cuiyh/Work/Code/Python/VQA_Challenge19/
vim ./
vim /home/cuiyh/Work/Code/Python/VQA_Challenge19/
vim ./
cd /data/shaozw/
ls
mv result_run_scan_1218_epoch13.pkl ensemble/
cd ensemble/
ls
ln -s /data/cuiyh/ensemble/VQA_Challenge19/ensemble_run_100_epoch13.pkl ./ensemble_run_100_epoch13.pkl
ln -s /data/cuiyh/ensemble/VQA_Challenge19/ensemble_run_101_epoch13.pkl ./ensemble_run_101_epoch13.pkl
ln -s /data/cuiyh/ensemble/VQA_Challenge19/ensemble_run_102_epoch13.pkl ./ensemble_run_102_epoch13.pkl
cd ~/distributed-vqa-bert/
vim ./
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
ls
mv 67.json ~/
vim bert-arch2-001.py
cd /data/shaozw/ensemble/
ln -s /data/cuiyh/ensemble/VQA_UAN/ensemble_run_700_epoch13.pkl ./ensemble_run_700_epoch13.pkl
ln -s /data/cuiyh/ensemble/VQA_UAN/ensemble_run_701_epoch13.pkl ./ensemble_run_701_epoch13.pkl
ln -s /data/cuiyh/ensemble/VQA_UAN/ensemble_run_702_epoch13.pkl ./ensemble_run_702_epoch13.pkl
python utils_vqa/ensemble1.py 
cd -
python utils_vqa/ensemble1.py 
vim ./
python utils_vqa/ensemble1.py 
mv 67.json ~/7e.json
vim ./
nvidia-smi
vim ./
python bert-arch2-001.py 
vim ./
python bert-arch2-001.py 
vim ./
python bert-arch2-001.py 
vim bert-arch2-001.py 
python bert-arch2-001.py 
vim bert-arch2-001.py 
python bert-arch2-001.py 
nvidia-smi
ps aux|grep 24873
nvidia-smi
vim ./
cd ../
zip -r dvb2.zip distributed-vqa-bert/
nvidia-smi
cd distributed-vqa-bert/
cp bert-arch3-001.py bert-arch4-001.py 
vim bert-arch4-001.py 
python bert-arch4-001.py 
cd ~/openvqa/
cp configs/vqa/scan_3.yml configs/vqa/scan_4.yml 
vim configs/vqa/scan_4.yml 
python run.py --RUN=train --MODEL=scan_4 --GPU=0 --DATASET=vqa --SEED=11 --VERSION=scan_arch4
python run.py --RUN=train --MODEL=scan_4 --GPU=0 --DATASET=vqa --SPLIT='train+val+vg' --SEED=11 --VERSION=scan_arch4
nvidia-smi
python run.py --RUN=test --MODEL=scan_4 --GPU=2 --DATASET=vqa --SPLIT='train+val+vg' --SEED=11 --VERSION=scan_arch4 --CKPT_V=scan_arch4 --CKPT_E=13
cp results/pred/result_run_scan_arch4_epoch13.pkl /data/shaozw/ensemble/vqa-bert/arch4_13.pkl 
conda activate pyt0rch
cd /data/shaozw/
ls
ls vqa_backup/
cp -r vqa_backup/ vqa_nccl
cd vqa_nccl
nvidia-smi
vim ./
ls /data-ssd/
ls /data-ssd/vqa/
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
clear
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
conda install pytorch=1.0.1 -y
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
conda list
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/
vim ~/miniconda3/envs/pyt0rch/lib/python3.6/site-packages/torch/distributed/
conda install pytorch=1.5.0 -y
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cudatoolkit-10.1.243-h6bb024c_0.tar.bz2
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cudnn-7.6.0-cuda10.1_0.tar.bz2
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/pytorch-1.5.0-py3.6_cuda10.1.243_cudnn7.6.3_0.tar.bz2
mv pytorch-1.5.0-py3.6_cuda10.1.243_cudnn7.6.3_0.tar.bz2 ~/miniconda3/pkgs/
mv cudatoolkit-10.1.243-h6bb024c_0.tar.bz2 ~/miniconda3/pkgs/
mv cudnn-7.6.0-cuda10.1_0.tar.bz2 ~/miniconda3/pkgs/
vim ~/miniconda3/pkgs/
conda install pytorch=1.5.0 -y
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
kill -s 9 26543
kill -s 9 22014
nvidia-smi
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
kill -s 9 6673
kill -s 9 27468
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
kill -s 9 7277
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
nvidia-smi
vim ./
ls /data/
ls /data/cuiyh/
ls /data-ssd/cuiyh/
ls /home/cuiyh/
ls /home/cuiyh/Work/
ls /home/cuiyh/Work/Code/
vim /home/cuiyh/cama@192.168.157.159 
ls /home/cuiyh/
vim /home/cuiyh/cama@192.168.157.159 
ls /home/cuiyh/Work/Code/Python/
cd /home/cuiyh/Work/Code/Python/
ls
vim ./
cd ../
ls
du -h
cp -r Python/ /data/shaozw/
vim ./
cd /data/shaozw/
ls
mv Python/ code_from_cyh
du -h code_from_cyh/
cd code_from_cyh/
du -h
ls VQA_UAN/result/
cd VQA_UAN/result/
du -h
df -h
ls -lh
rm ./*
ls
cd -
du -h
ls VQA_base/model_save/
rm VQA_base/model_save/*
ls darts-master/data/
ls darts-master/data/cifar-10-batches-py/
ls darts-master/data/imagenet/
ls darts-master/data/penn/
rm -r darts-master/data/*
du -h
ls Rnn/
ls Rnn/model/
rm Rnn/model/*
rm Rnn/stanford-segmenter-2018-02-27.zip 
ls VQA_UAN/log/
rm -r VQA_UAN/log/*
ls VQA_UAN/log/
cd ../
cd -
ls VQA_UAN/utils
du -h
ls mcan_release/
ls mcan_release/ckpts/
rm -r mcan_release/ckpts/*
du -h
ls ObjectDetection/
ls ObjectDetection/Model_save/
rm -r ObjectDetection/Model_save/*
du -h
ls VQA_Challenge19/
ls VQA_Challenge19/result_test/
rm -r VQA_Challenge19/result_test/*
ls VQA_Challenge19/result_ensemble/
rm -r VQA_Challenge19/result_ensemble/*
ls pt.darts-master/
ls pt.darts-master/data/
rm -r pt.darts-master/data/*
du -h
ls VQA_base/
ls VQA_base/model
rm VQA_base/model/*
ls ObjectDetection/Model/
rm ObjectDetection/Model/*
du -h
rm ObjectDetection/VIDEO/
ls ObjectDetection/VIDEO/
rm -r ObjectDetection/VIDEO/*
ls ObjectDetection/
ls ObjectDetection/Model_withNet/
rm ObjectDetection/Model_withNet/*
du -h
ls VQA_FTB/
ls VQA_FTB/result_ensemble/
ls VQA_FTB/result_test/
rm -r VQA_FTB/result_test/*
ls VQA_FTB/img_feature/
ls VQA_FTB/log/
rm VQA_FTB/log/*
ls VQA_FTB/
ls VQA_FTB/result_test/
du -h
ls VQA_bottom_up/
ls VQA_bottom_up/model
rm -r VQA_bottom_up/model
ls VQA_bottom_up/model_save/
ls SemanticSegmentation_AICAR/
ls SemanticSegmentation_AICAR/Model
rm -r SemanticSegmentation_AICAR/Model/*
ls SemanticSegmentation_AICAR/ModelSave/
rm -r SemanticSegmentation_AICAR/ModelSave/
du -h
ls visualground/
rm visualground/visualground.zip 
du -h
ls faster-rcnn/
cd faster-rcnn/
du -h
cd utils/
du -h
vim ./
ls
cd tools/
cd ../
du -h
cd tools/
du -h
cd CUDA
du -h
cd build/
du -h
cd ../../../
cd ../
vim ./
cd nas-1/
ls
vim nccl-vqa-pxy14-0001.py 
cd ../
cd -
vim ./
cd ../
zip nas-1 ./nas-1/
zip -r nas-1.zip ./nas-1/
mv nas-1.zip ~
cd /home/cuiyh/
ls
cd Work/
cd Code/Python/nas-1/
ls -lrt
cd /data/cuiyh/checkpoint/
ls
cd /data/cuiyh/checkpoint/nas-1/
ls
cd arch/
ls
vim nccl-vqa-pxy14-0056-search.json 
clear
cd ~
ls /data-ssd/cuiyh/features/
ls /data-ssd/cuiyh/
ls /data-ssd/cuiyh/features/
ls /data-ssd/shaozw/
ls /data-ssd/shaozw/dataset/
ls /data-ssd/shaozw/dataset/vqa/
ls /data-ssd/shaozw/dataset/vqa/feats/
ls /data/shaozw/
mkdir /data/shaozw/checkpoint
mkdir /data/shaozw/ensemble
ls /data/cuiyh/dataset/mscoco/image2014/
ls /data/cuiyh/dataset/mscoco/image2014/train2014/
ls /data/shaozw/checkpoint
mkdir /data/shaozw/checkpoint/result_test
mkdir /data/shaozw/checkpoint/result_ens
mkdir /data/shaozw/checkpoint/tmp
rm -r /data/shaozw/checkpoint/*
ls
unzip distributed-vqa-bert.zip 
ls
cd distributed-vqa-bert/
ls
ls /data/shaozw/vqa_backup/
cp -r /data/shaozw/vqa_backup/transformers/ ./
python det3-bert-000.py 
vim det3-bert-000.py 
python det3-bert-000.py 
vim det3-bert-000.py 
python det3-bert-000.py 
vim det3-bert-000.py 
python det3-bert-000.py 
vim det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
ls /data-ssd/shaozw/dataset/vqa/raw/
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
nvidia-smi
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
ls
mkdir log
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py 
vim ./
python det3-bert-000.py | tee log.txt
vim ./
python det3-bert-000.py | tee log.txt
vim ./
python det3-bert-000.py | tee log.txt
vim ./
python det3-bert-000.py | tee log.txt
vim ./
python det3-bert-000.py | tee log.txt
vim ./
python det3-bert-000.py | tee log.txt
python det3-bert-000.py
nohup python det3-bert-000.py 2>&1 &
ls
cat nohup.out 
nvidia-smi
nohup python det3-bert-000.py &
nvidia-smi
cat nohup.out 
nohup python det3-bert-000.py &
cat nohup.out 
nvidia-smi
cat nohup.out 
nvidia-smi
vim nohup.out 
nvidia-smi
kill -s 9 29554
kill -s 9 29555
ps aux|python
ps aux|grep python
vim ./
vim .vimrc
vim ./
vim ~
mv .vimrc ~
vim ./
vim ~/.vimrc
vim ./
vim ~/.vimrc
vim ./
vim ~/.vimrc
vim ./
vim ~/.vimrc
vim ./
vim ~/.vimrc
vim ./
vim ~/.vimrc
vim ./det3-bert-000.py 
vim ~/.vimrc
python
vim ./det3-bert-000.py 
python det3-bert-000.py
vim ./det3-bert-000.py 
python det3-bert-000.py
vim ./det3-bert-000.py 
python det3-bert-000.py
nvidia-smi
vim ./det3-bert-000.py 
python det3-bert-000.py
vim ./det3-bert-000.py 
python det3-bert-000.py
nvidia-smi
vim ./det3-bert-000.py 
python det3-bert-000.py
vim ./det3-bert-000.py 
python det3-bert-000.py
ls /data/shaozw/checkpoint/
vim ./
mkdir /data/shaozw/checkpoint/vqa-bert
python det3-bert-000.py
ls /data/shaozw/checkpoint/vqa-bert/
vim ./
python det3-bert-000.py
vim ./
python det3-bert-000.py
vim ./
python det3-bert-000.py
cp det3-bert-000.py det3-bert-001.py 
vim det3-bert-001.py 
vim ~/.vimrc 
cd ~/.vim/
git clone git://github.com/altercation/solarized.git
mv ~/colors/ ./
cd -
vim det3-bert-001.py 
vim ~/.vimrc 
vim det3-bert-001.py 
vim ~/.vimrc 
vim det3-bert-001.py 
vim ~/.vimrc 
vim det3-bert-001.py 
python det3-bert-001.py
vim ./
python det3-bert-001.py
vim ./
python det3-bert-001.py
cd /data/shaozw/vqa_nccl/
ls
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
python -m torch.distributed.launch --nproc_per_node=4 run.py --RUN=train --MODEL=scan_1 --GPU=0,1,2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=0,1 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
vim ./
python -m torch.distributed.launch --nproc_per_node=1 run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
python -m torch.distributed.launch --nproc_per_node=2 run.py --RUN=train --MODEL=scan_1 --GPU=1,2 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
python -m torch.distributed.launch --nproc_per_node=4 run.py --RUN=train --MODEL=scan_1 --GPU=0,1,2,3 --DATASET=vqa --SEED=888 --VERSION=scan_nccl
cd -
ls
cp det3-bert-001.py det3-bert-002.py 
ls /data/shaozw/checkpoint/vqa-bert/
ls /data/shaozw/checkpoint/vqa-bert/ -lh
vim ./
python det3-bert-002.py 
vim ./
python det3-bert-002.py 
nvidia-smi
kill -s 9 19362
kill -s 9 19363
kill -s 9 19364
nvidia-smi
python det3-bert-002.py 
cp det3-bert-002.py det3-bert-003.py 
vim det3-bert-003.py 
python det3-bert-003.py 
ls
cp det3-bert-003.py det3-bert-003_1.py
vim det3-bert-003_1.py
python det3-bert-003_1.py
vim ./
python det3-bert-004.py
vim det3-bert-004.py det3-bert-004_1.py
cp det3-bert-004.py det3-bert-004_1.py
vim det3-bert-004_1.py
vim ./
python det3-bert-004_1.py
cp det3-bert-004.py det3-bert-005.py
vim det3-bert-005.py
python det3-bert-005.py
cp det3-bert-004.py det3-bert-006.py
vim det3-bert-006.py
python det3-bert-006.py
vim det3-bert-006.py
python det3-bert-006.py
vim det3-bert-006.py
python det3-bert-006.py
nvidia-smi
python det3-bert-006.py
nvidia-smi
vim det3-bert-006.py
python det3-bert-006.py
vim det3-bert-006.py
python det3-bert-006.py
vim det3-bert-006.py
python det3-bert-006.py
rm det3-bert-006.py
cp det3-bert-004.py det3-bert-006.py
vim det3-bert-006.py
vim ./
vim det3-bert-006.py
python det3-bert-006.py
vim ./
cp det3-bert-006.py det3-bert-006_dev.py 
vim det3-bert-006_dev.py 
python det3-bert-006_dev.py 
vim det3-bert-006_dev.py 
python det3-bert-006_dev.py 
vim det3-bert-006_dev.py 
python det3-bert-006_dev.py 
ls
vim ./
ls /data/shaozw/ensemble/
ls /data/shaozw/checkpoint/
ls /data/shaozw/checkpoint/vqa-bert/
ls /data/shaozw/checkpoint/vqa-bert/result_ens/
ls /data/shaozw/checkpoint/vqa-bert/result_test/
vim ./ 
python det3-bert-006_dev.py 
cp det3-bert-008.py det3-bert-009.py 
vim det3-bert-009.py 
python det3-bert-009.py 
nvidia-smi
vim det3-bert-009.py 
python det3-bert-009.py 
nvidia-smi
vim det3-bert-009.py 
python det3-bert-009.py 
vim det3-bert-009.py 
python det3-bert-009.py
ls /data/shaozw/checkpoint/
ls /data/shaozw/checkpoint/vqa-bert/result_test/
cp /data/shaozw/checkpoint/vqa-bert/result_test/result_det3-bert-009_epoch12.json ~/9_12.json
cd /data/cuiyh/
ls
vim ./
cd ensemble/VQA_Challenge19/
ls
ls -lh
nvidia-smi
vim /home/cuiyh/Work/Code/Python/VQA_Challenge19/
vim /home/cuiyh/Work/Code/Python/
nvidia-smi
ps aux|grep 8502
vim /home/cuiyh/Work/Code/Python/
cd /data/shaozw/
vim ./
ps aux|grep 8502
vim ./
cd ensemble/
ls
mv result_run_scan_1218_epoch13.pkl ensemble_run_bs_epoch13.pkl 
vim ./
cd /data/cuiyh/ensemble/
vim ./
cd ~/distributed-vqa-bert/
cp bert-arch2-001.py bert-arch3-001.py 
vim ./
python bert-arch3-001.py 
vim bert-arch3-001.py 
python bert-arch3-001.py 
vim bert-arch3-001.py 
python bert-arch3-001.py 
vim bert-arch3-001.py 
python bert-arch3-001.py 
ls /data/shaozw/ensemble/vqa-bert/
ls ..
ls /data/shaozw/ensemble/
mv /data/shaozw/ensemble/ensemble_run_bs_epoch13.pkl /data/shaozw/ensemble/vqa-bert/ensemble_bert-arch1-001_epoch13.pkl 
vim ./
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
ls /data/shaozw/ensemble/vqa-bert/ -lh
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
vim utils_vqa/ensemble1.py 
python utils_vqa/ensemble1.py 
ls
mkdir ~/archs/
mv 67.json ~/archs/4en.json
ls /data/shaozw/checkpoint/vqa-bert/result_test/
mv /data/shaozw/checkpoint/vqa-bert/result_test/result_bert-arch2-001_epoch13.json ~/archs/2.json
mv /data/shaozw/checkpoint/vqa-bert/result_test/result_bert-arch3-001_epoch13.json ~/archs/3.json
mv /data/shaozw/checkpoint/vqa-bert/result_test/result_bert-arch4-001_epoch13.json ~/archs/4.json
vim ./
ls
cp bert-arch2-001.py bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
vim ./
python bert-arch1-001.py 
nvidia-smi
vim ./
nvidia-smi
python bert-arch1-001.py 
vim ./
python bert-arch1-001.py 
vim ./
python bert-arch1-001.py 
nvidia-smi
vim ./
nvidia-smi
python bert-arch1-001.py 
vim ./
python bert-arch1-001.py 
vim bert-arch1-001.py 
nvidia-smi
ps aux|grep 12622
python bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
vim bert-arch1-001.py 
python bert-arch1-001.py 
ls /data/shaozw/checkpoint/vqa-bert/result_test/
mv /data/shaozw/checkpoint/vqa-bert/result_test/result_bert-arch1-001_epoch13.json  ~/a1.json
vim bert-arch1-001.py 
python bert-arch1-001.py 
cp bert-arch1-001_1.py bert-arch1-001_2.py 
vim bert-arch1-001_2.py 
nvidia-smi
vim bert-arch1-001_2.py 
python bert-arch1-001_2.py 
nvidia-smi
vim bert-arch1-001_1.py 
python bert-arch1-001_2.py 
python bert-arch1-001_1.py 
vim bert-arch1-001_1.py 
python bert-arch1-001_1.py 
vim bert-arch1-001_1.py 
python bert-arch1-001_1.py 
vim bert-arch1-001_1.py 
python bert-arch1-001_1.py 
cd ..
ls
rm -r tmp
rm -r *.json
unzip tmp.zip 
ls
cd tmp
cd ../
mv tmp openvqa
cd openvqa/
ls
vim ./
ls /data-ssd/shaozw/dataset/
ls /data/shaozw/
ls /data/shaozw/checkpoint/
mkdir /data/shaozw/checkpoint/openvqa
ln -s /data/shaozw/checkpoint/openvqa/ ckpts
ls
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=1 --DATASET=vqa --SEED=11 --VERSION=scan_test
nvidia-smi
python run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=11 --VERSION=scan_test
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=11 --VERSION=scan_test
vim ./
ls /data/shaozw/checkpoint/vqa-bert/ -lh
rm /data/shaozw/checkpoint/vqa-bert/*
ls /data/shaozw/checkpoint/openvqa/ -lh
ls /data/shaozw/checkpoint/openvqa/ckpt_scan_test/ -lh
vim ./
python run.py --RUN=train --MODEL=scan_1 --GPU=2 --DATASET=vqa --SEED=11 --VERSION=scan_test
cd /home/cuiyh/Work/Code/Python/
ls
ls -lh
cd VQA_Challenge19/
ls
cd img_feature/
ls
ls -lj
ls -lh
cd ../../
cd GQA_graph6/
ls
ls utils_vqa/
cd ../../../
../
cd ../
ls
cd /data/cuiyh/
ls
ls features/
ls features/idgrid_x152-448_448/
ls tools/
whereis grid
cd /home/cuiyh/Work/Code/Python/
cd /home/cuiyh/
vim ./
tail .bash_history 
cd ~/
ls
mkdir tmp
cp -r distributed-vqa-bert/ tmp
cp -r openvqa/ tmp
cd tmp
vim ./
ls
ls distributed-vqa-bert/
rm -r distributed-vqa-bert/transformers/
rm -r distributed-vqa-bert/log
rm -r openvqa/transformers/
rm openvqa/ckpts
rm -r ./*/*/__pycache__
rm -r ./*/*/*/__pycache__
rm -r ./*/*/*/*/__pycache__
vim ./
cd ../
zip -r tmp.zip tmp
ls miniconda3/pkgs/
cp miniconda3/pkgs/pytorch-1.5.0-py3.6_cuda10.1.243_cudnn7.6.3_0.tar.bz2 ./
python
conda list
cd frp_0.27.0_linux_amd64/
vim ./
nohup ./frpc -c ./frpc.ini &
ps aux |grep frpc
kill -s 9 11787
vim ./
nohup ./frpc -c ./frpc.ini &
ps aux |grep frpc
kill -s 9 13629
vim ./
nohup ./frpc -c ./frpc.ini &
exit
screen -ls
screen -r nas-bert
screen -r openvqa-nas
screen -r nas-bert
screen -r openvqa-nas
screen -r nas-bert
screen -r read_code
screen -S read_code
screen -S a-nas-bert
screen -r openvqa-nas
screen -r a-nas-bert
screen -r openvqa-nas
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
vim ./
screen -r a-nas-bert
screen -r nas-bert
nvidia-smi
screen -S nas-bert
screen -S openvqa-nas
screen -S nas-bert
screen -r nas-bert
screen -r openvqa-nas
screen -r nas-bert
screen -r a-nas-bert
screen -r openvqa-nas
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r openvqa-nas
screen -r nas-bert
screen -r a-nas-bert
screen -S download
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
vim ./
screen -r download
screen -r a-nas-bert
screen -r nas-bert
screen -r download
ls /data/shaozw/datasets/lxmert/data/npzdata/
screen -r download
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r openvqa-nas
screen -r nas-bert
screen -r openvqa-nas
screen -r a-nas-bert
screen -r nas-bert
cd /data/shaozw/checkpoint/
ls
cd vqa-bert/
ls
rm nas-bert-001*.pkl
ls
rm nas-bert-002*.pkl
ls
rm nas-bert-003*.pkl
ls
screen -r nas-bert
screen -r a-nas-bert
vim ./
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
cd labs/vqa_project/
vim ./
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
vim ./
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r nas-bert
nvidia-smi
screen -r b-nas-bert
screen -S b-nas-bert
screen -S c-nas-bert
cd /data/shaozw/checkpoint/
ls
cd vqa-bert/
ls
rm nas-bert-004*.pkl
ls
rm nas-bert-005*.pkl
rm nas-bert-006*.pkl
rm nas-bert-007*.pkl
ls
nvidia-smi
screen -r c-nas-bert
screen -r b-nas-bert
screen -r a-nas-bert
screen -r nas-bert
screen -r a-nas-bert
screen -r c-nas-bert
clear
ls /data/
ls /data/shaozw/
ls /data/cuiyh/
ls /data/cuiyh/dataset/
ls /data/cuiyh/dataset/VQA/
ls /data/gaopb/
ls /data/gaopb/features/
ls /data/gaopb/dataset/
cd labs
ls
git clone https://github.com/ParadoxZW/openvqa.git
cd labs/openvqa/
screen -S opnas
screen -S a-opnas
screen -r a-opnas
screen -r opnas
screen -r a-opnas
nvidia-smi
screen -S b-opnas
screen -r a-opnas
screen -r b-opnas
screen -S c-opnas
screen -r b-opnas
cd labs
git clone https://github.com/ParadoxZW/ANNS.git
cd ANNS
mv ~/sift.zip ./
unzip sift.zip 
screen -S anns
screen -r anns
screen -S bnns
screen -r b-opnas
screen -r a-opnas
screen -r opnas
screen -r c-opnas
screen -r a-opnas
screen -r opnas
screen -r a-opnas
screen -r c-opnas
screen -r b-opnas
screen -r c-opnas
screen -r b-opnas
screen -r opnas
screen -r bnns
ls
rm gist.zip 
screen -r bnns
screen -r opnas
nvidia-smi
screen -r opnas
screen -S download
screen -r bnns
screen -r download
screen -r 21545
mv gist.zip labs
cd labs
unzip gist.zip 
screen -r bnns
screen -S faiss
screen -r faiss
screen -r bnns
screen -r faiss
screen -r bnns
ls
cd labs
ls
cd ANNS/
ls
vim run_graph_search.sh 
screen -r faiss
cd labs/ANNS/
vim demo.py
screen -r faiss
screen -S bfaiss
screen -r bfaiss
screen -r bfaiss
screen -rD bfaiss
cd labs/ANNS/
vim rr.py 
screen -rD bfaiss
screen -rD faiss
cat /proc/cpuinfo 
cat /proc/meminfo | grep MemTotal
g++ version
g++ -version
g++ --version
g++ -v
screen -rD faiss
cd labs/ANNS/
ls
cp opq.txt ~/
ls labs
cp labs/demo.py ./
cd labs
cd openvqa/
vim 
cd results/
vim ./
cd labs/openvqa/
cd results/log/
vim ./
cd labs
ls
rm caption_datasets.zip 
rm gist.tar.gz 
ls
ls gist
rm -r gist
ls
rm -r gist.zip 
rm demo.py 
rm rr.py 
rm opq_good.zip 
rm -r opq_good/
ls
rm update.sh 
rm dataset_flickr30k.json 
rm dataset_flickr8k.json 
clear
ls
ls backup/
mv backup/ lxmert-backup
ls
rm opq16.zip 
ls opq16/
rm -r opq16/
ls
ls openvqa/
cd openvqa/
rm -r ckpts/*
cd ../
mv openvqa/ openvqa_mmnasnet
ls
rm -r old-ANNS/
clear
ls
ls self-critical.pytorch/
ls vqa_project
ls
rm vqa_project.zip 
ls -lh
cd ~
ls
rm pytorch-1.5.0-py3.6_cuda10.1.243_cudnn7.6.3_0.tar.bz2 
rm dvb2.zip 
rm dvb.zip 
rm v0.1.2.zip 
rm v0.1.3.zip 
clear
ls
ls 3archs/
rm 3archs/
rm -r 3archs/
clear
ls
rm 67.json 
rm -r log88
ls
rm Eigen.zip 
ls archs/
rm -r archs/
ls
ls ansdict/
ls labs
rm -r nns_log/
rm backup.tar.gz 
ls board/
rm -r board/
cat demo.py 
ls
rm demo.py 
clear
ls
cat opq.txt 
rm opq.txt 
ls
ls tmp
ls tmp1
clear
ls
ls -lh
ls tmp
rm opq.zip
ls openvqa/
cd openvqa/
ls results/log/
ls -lh
cd ../
clear
ls -lh
ls tmp
cd tmp
ls -lh
ls distributed-vqa-bert/
ls
cd end2end/
ls
ls colors/
rm -r colors/
ls
ls -lh | grep .py
ls -lh | grep \.py
ls -lh | grep *.py
ls -lh | grep .py
ls
vim cnn-bert-008.py 
ls
ls log -lh
vim log/
ls -lh | grep .py
vim cnn-bert-008.py 
vim cnn-bert-001.py 
cd labs
ls
git clone https://github.com/MILVLG/openvqa.git
ls
mv openvqa position
cd position/
ls
screen -S pos
screen -r pos
nvidia-smi
screen -S apos
screen -r pos
screen -r apos
screen -r pos
ls
cd labs/
ls
git clone https://github.com/MILVLG/mmnas.git
cd mmnas/
ls
screen -ls
screen -S mmnas
screen -r mmnas
screen -Dr mmnas
nvidia-smi
screen -S  vg-mmnas
screen -S itm-mmnas
screen -r itm-mmnas
screen -Dr itm-mmnas
screen -rd itm-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r mmnas
screen -r vg-mmnas
vim .bashrc 
vim /home/cuiyh/.bashrc
cp .bashrc .bashrc.backup
vim .bashrc
screen -S newBashrc
screen -r vg-mmnas
conda list
python
mv .bashrc .bashrc.1
mv .bashrc.backup .bashrc
screen -r newBashrc
ls -a
vim .torch/
vim ./
screen -ls
screen -S xp
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -S a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r mmnas
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r itm-mmnas
screen -r a-vg-mmnas 
nvidia-smi
screen -r a-vg-mmnas 
nvidia-smi
screen -r a-vg-mmnas 
screen -r itm-mmnas
cd labs/mmnas/
vim .
screen -r itm-mmnas
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -r a-vg-mmnas 
cd labs/mmnas/logs/ckpts/
ls
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -S data_process
cd labs/mmnas/data/
vim data_filter.py 
screen -r data_process
exit
screen -r itm-mmnas
screen -S a-itm-mmnas
screen -r itm-mmnas
screen -S a-itm-mmnas
screen -r a-itm-mmnas
screen -r mmnas
nvidia-smi
screen -r mmnas
screen -r a-itm-mmnas
screen -r data_process
screen -r mmnas
screen -r data_process
nvidia-smi
screen -r mmnas
nvidia-smi
screen -r data_process
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r data_process
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r mmnas
cd labs/mmnas/
vim ./
screen -r a-itm-mmnas
screen -r mmnas
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r a-itm-mmnas
screen -r a-vg-mmnas 
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r a-itm-mmnas
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-itm-mmnas
screen -r itm-mmnas
screen -r vg-mmnas 
screen -r itm-mmnas
screen -r a-vg-mmnas 
screen -r itm-mmnas
ls /home/cuiyh/Work/Code/Python/
ls /home/cuiyh/Work/Code/Python/nas-1/
ls /home/cuiyh/Work/Code/Python/nas-1/log/
vim /home/cuiyh/Work/Code/Python/nas-1/
ls /data/cuiyh/checkpoint/
ls /data/cuiyh/checkpoint/nas-1/
ls /data/cuiyh/checkpoint/
ls /data/cuiyh/checkpoint/std_ckpt/
screen -ls
screen -r xp
screen -r itm-mmnas
screen -r mmnas
clear
screen -r mmnas
screen -r itm-mmnas
screen -r mmnas
nvidia-smi
screen -r data_process
cd labs/mmnas/
cd data/
ls
ls itm
ls dataset/flickr/
du -h dataset/flickr/
du -h dataset/flickr/dataset.json 
du -h dataset/flickr/dataset_flickr30k.json 
head -10 dataset/flickr/dataset.json 
screen -r mmnas
screen -r data_process
cd labs/mmnas/
cd data/
ls
cd itm/
ls
cd flickr
ls
head -10 test_caps.txt 
head -10 test_tags.txt 
head -10 test_ids.txt 
screen -r mmnas
nvidia-smi
screen -r mmnas
cd labs
cd mmnas/
cd logs/
ls
cd ckpts/
cd ../log/
ls
cp log_train_vqa-search.txt ~/
cp log_train_vqa-search.txt ~/search_log.txt
cd labs/mmnas/
ls
cd /data/cuiyh/
ls
cd release/nas-1/
ls
vim log/
vim log/log_nccl-vqa-pxy14-0019-search.txt 
cd ~/labs/mmnas/
ls
vim search_vqa.py 
screen -r mmnas
cd labs/mmnas/logs/
vim .
screen -r mmnas
cd labs/mmnas/logs/log/
vim .
screen -r mmnas
screen -ls
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r itm-mmnas 
screen -r a-vg-mmnas 
cd /data/cuiyh/release/nas-1/log/
ls
vim log_nccl-vqa-pxy14-0019-search.txt 
screen -r a-vg-mmnas 
screen -r itm-mmnas 
screen -r vg-mmnas 
screen -r mmnas
screen -r vg-mmnas 
screen -r itm-mmnas 
screen -r a-vg-mmnas 
cd labs/mmnas/
vim train_vqa.py 
screen -r a-vg-mmnas 
screen -r itm-mmnas 
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r mmnas
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r itm-mmnas 
cd labs/mmnas/
cd logs/log/
vim ./
screen -r mmnas
cd labs/mmnas/
cd logs/log/
vim log_train_vqa-search.txt 
screen -r mmnas
screen -r itm-mmnas 
screen -r mmnas
screen -r itm-mmnas 
screen -r mmnas
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -r a-vg-mmnas 
screen -r mmnas
screen -r itm-mmnas 
cd labs/mmnas/logs/log/
ls
vim log_train_vqa-search.txt 
screen -r itm-mmnas 
screen -r mmnas
screen -r a-vg-mmnas 
screen -r vg-mmnas 
screen -S frp
cd frp_0.27.0_linux_amd64/
nohup ./frpc -c ./frpc.ini &
clear
ls
screen -r mmnas 
cd ~
ll
cd frp_0.27.0_linux_amd64/
ll
./frpc -c frpc.ini 
nohup ./frpc -c frpc.ini 
tail -f nohup.out 
htop
screen -ls
kill 6904
screen -ls
tail -f nohup.out 
htop
./frpc -c frpc.ini 
screen -ls
cd ..
cat .bash_history | grep frpc
cd frp_0.27.0_linux_amd64/
nohup ./frpc -c ./frpc.ini 
nohup ./frpc -c ./frpc.ini &
ll -rt
cat frpc.ini 
cd ..
ll
cd data/
cd /data
ll
cd shaozw/
ll
cd usr
ll
cd ../data1
ll
cd ..
ll
cd /data-ssd/
ll
cd shaozw/
ll
cd ~/
ll
find frpc
cd frp_0.27.0_linux_amd64/
l
ll
cd ..
ll
cd tools1/
ll
cd ..
ls | grep frp
cd tmp
ll
cd ..
cd tmp1
ll
cd labs
ll
cd ../labs
ll
cd ..
ls
ll
ls
cd headfiles/
ll
cd ..
ll
cd miniconda3/
ll
cd ..
ll
ping www.baidu.com
ping www.hdu.edu.cn
cd ..
ll
cd frp
cd 
ll
cd frp_0.27.0_linux_amd64/
ll
nohup ./frpc -c frpc.ini &
tail -f nohup.out
lslogins
ll
cat frpc.ini 
lslogins
watch -n1 gpustat
nvidia-smi
screen -r mmnas
kill -s9 7632
kill -s 9 7632
kill -s 9 7633
screen -r mmnas
screen -r vg-mmnas
screen -r a-vg-mmnas
screen -r vg-mmnas
screen -r itm-mmnas
screen -r vg-mmnas
screen -r a-vg-mmnas
screen -r vg-mmnas
screen -r a-vg-mmnas
screen -r itm-mmnas
screen -r a-vg-mmnas
screen -r vg-mmnas
screen -r mmnas
screen -r vg-mmnas
screen -r mmnas
ll
cd ~
ll
cd frp_0.27.0_linux_amd64/
ll
./frpc -c ./frpc.ini 
screen -r mmnas
cd labs/mmnas/logs/log/
vim .
clear
cd labs/mmnas/
vim ./
vim ~/.vimrc
cd labs/mmnas/
vim search_vqa.py 
cd labs/mmnas/
vim search_vqa.py 
vim .vimrc
which vim
vim -version
screen mmnas
screen -r mmnas
cd /data/cuiyh/release/nas-1/
conda activate pyt0rch
ls
python "print(-3 // 2)"
python print(-3 // 2)
python -u "print(-3 // 2)"
python -help
python -h
python -c "print(-3 // 2)"
clear
ls
python -c "print(-3 // 2)"
ls /data/cuiyh/
ls /data/cuiyh/features/
cd ..
ls
cd openvqa_mmnasnet/
vim ./
cd ../mmnas/
cd data/
ls
ln -s /data/cuiyh/features/roisfeat_resnet101_10-100/ coco_extract
ls 
ls /data/cuiyh/dataset/
ls /data/cuiyh/dataset/VQA/
ls /data/cuiyh/dataset/mscoco
ls /data/cuiyh/dataset/mscoco/image2014/
ls /data/cuiyh/dataset/data_no_feature
ls /data/cuiyh/dataset/data_no_feature/coco_precomp/
ls /data/cuiyh/dataset/
clear
ls /data/cuiyh/dataset/
ls /data/cuiyh/dataset/VQA/
ls /data/cuiyh/dataset/VQA/v2_Questions_Train_mscoco/
cp -r /data/cuiyh/dataset/VQA/ vqa
cd vqa/
ls
mv */* ./
ls
ls v2_Questions_Val_mscoco/
nvidia-smi
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json'
cd ../..
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json'
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --DATASET='train+val+vg'
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --SPLIT='train+val+vg'
vim train_vqa.py 
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --VERSION='vqa-bbox'
vim train_vqa.py 
vim ./
python3 train_vqa.py --RUN='test' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --SPLIT='train+val+vg' --CKPT_PATH='./logs/ckpts/train_vqa-full_epoch13.pkl' --GPU=3
vim ./
python3 train_vqa.py --RUN='test' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --SPLIT='train+val+vg' --CKPT_PATH='./logs/ckpts/train_vqa-full_epoch13.pkl' --GPU=3
vim ./
cp logs/ckpts/result_test/result_train_vqa-full.json ~/vqa_mmnas.json
ls ~
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=123 --SPLIT='train+val+vg' --GPU=0,1,2,3
vim ./
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=123 --SPLIT='train+val+vg' --GPU=0,1,2,3
python3 train_vqa.py --RUN='test' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --SPLIT='train+val+vg' --CKPT_PATH='./logs/ckpts/train_vqa-full_epoch13.pkl' --GPU=3
cp logs/ckpts/result_test/result_train_vqa-full.json ~/vqa_mmnas.json
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=123 --SPLIT='train+val+vg' --GPU=0
python3 train_vqa.py --RUN='test' --GENO_PATH='./logs/ckpts/arch/train_vqa.json' --SEED=888 --SPLIT='train+val+vg' --CKPT_PATH='./logs/ckpts/train_vqa-full_epoch13.pkl' --GPU=0
cp logs/ckpts/result_test/result_train_vqa-full.json ~/vqa_mmnas.json
python mmnas/search/search_vqa.py 
cp mmnas/search/search_vqa.py 
cp mmnas/search/search_vqa.py  ./
python search_vqa.py 
vim search_vqa.py 
python search_vqa.py 
vim search_vqa.py 
vim ./
python search_vqa.py 
vim ./
python search_vqa.py 
vim ./
python search_vqa.py 
vim /data/cuiyh/release/nas-1/nccl-vqa-pxy14-0019.py 
vim mmnas/model/hygr_vqa.py 
vim /data/cuiyh/release/nas-1/nccl-vqa-pxy14-0019.py 
vim ./
vim ./search_vqa.py 
python search_vqa.py 
vim mmnas/model/hygr_vqa.py 
python search_vqa.py 
ls logs/ckpts/arch/
ls logs/ckpts/arch/train_vqa-search.json 
vim logs/ckpts/arch/train_vqa-search.json 
cd logs/ckpts/arch/
ls
vim train_vqa-search.json 
vim vqa-search-38.json
cd ../../
cd ..
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/vqa-search-38.json' --SEED=123 --GPU=0 --VERSION='arch38'
vim logs/ckpts/arch/vqa-search-38.json 
python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/vqa-sexir
exit
clear
cd /data/cuiyh/release/nas-1/
cd log/
ls
vim ./
screen -r mmnas
nvidia-smi
ps aux|grep 5175
screen -r mmnas
nvidia-smi
kill -s 9 9527
nvidia-smi
screen -r mmnas
nvidia-smi
screen -r mmnas
nvidia-smi
screen -r mmnas
screen -S mmnas
ps aux|grep shaozw
screen -S mmnas
screen -r mmnas
ps aux|grep multiprocessing
kill 2800
kill 2801
kill 2802
kill 4011
kill 4012
kill 4013
kill 5211
kill 19522
kill 19523
kill 19524
kill 32562
kill 32563
kill 32564
ps aux|grep multiprocessing
screen -r mmnas
ps aux|grep multiprocessing
screen -r mmnas
screen -r vg-mmnas
screen -r mmnas
ps aux|grep multiprocessing
kill 7277
kill 9268
ps aux|grep multiprocessing
kill 7277
ps aux|grep multiprocessing
kill 7277
ps aux|grep multiprocessing
kill 7277
kill 7278
kill 7279
ps aux|grep multiprocessing
kill 9269
kill 9270
ps aux|grep multiprocessing
screen -r mmnas
screen -r itm-mmnas
screen -r mmnas
screen -r itm-mmnas
nvidia-smi
screen -r itm-mmnas
screen -r mmnas
screen -r itm-mmnas
cd labs/mmnas/
vim ./
screen -r itm-mmnas
exit
apt-get install zsh
screen -r itm-mmnas
screen -r mmnas
ls .ssh/
vim .ssh/authorized_keys 
ls
clear
ls
ls grid-feats-vqa/
clear
ls
zsh5
./zsh5
cd /usr/lib/
ls
cd x86_64-linux-gnu/
ls
tourch szw_tmp
touch szw_tmp
cd ~
wget -O zsh.tar.xz https://sourceforge.net/projects/zsh/files/latest/download
mkdir zsh && unxz zsh.tar.xz && tar -xvf zsh.tar -C zsh --strip-components 1
ls
cd zsh
./configure --prefix=$HOME
sudo
sudo ls
cd ..
whereis zsh
ls /usr/share/zsh/
ls /usr/share/zsh/vendor-completions/
ls /bin/zsh
ls /usr/bin/zsh
ls
ls tools1
rm -r tools1/
rm zsh5
rm zsh.tar 
rm log_train_vqa-search.txt 
ls ansdict/
ls labs
mv ansdict/ labs/
ls labs
exit
sudo apt-get zsh
sudo apt-get install zsh
ls
chsh -s /usr/bin/zsh
vim .zshrc
vim .bashrc
ls
rm -r .antigen/
rm -rf .antigen/
ls
rm -r zsh
ls headfiles/
exit
