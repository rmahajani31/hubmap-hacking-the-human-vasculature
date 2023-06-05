AWS Setup:
sudo yum update
sudo yum install gcc gcc-c++ make git mesa-libGL tmux -y
export TMPDIR='/var/tmp'

curl https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh -b -p /home/ec2-user/anaconda3
echo 'export PATH="/home/ec2-user/anaconda3/bin:$PATH"' >> ~/.bashrc
bash
conda init
bash

git clone https://github.com/rmahajani31/hubmap-hacking-the-human-vasculature.git

git clone https://Akash-sopho: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@github.com/rmahajani31/hubmap-hacking-the-human-vasculature.git
cd hubmap-hacking-the-human-vasculature

conda env create -f HubMapEnv.yml
conda activate HubMapEnv

pip install gdown chardet torchmetrics
gdown https://drive.google.com/uc?id=1kQPgf0OhGQLdcIUDuCO8gk1DZIOWPyIh
unzip full_training_data.zip

python deepLabV3-model-training.py
mkdir models







curl https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh
bash

sudo apt update
sudo apt install gcc -y
sudo apt install build-essential

conda env create -f HubMapEnv.yml

conda install -c anaconda ipykernel
python -m ipykernel install --user --name=HubMapEnv
conda install jupyter

mkdir projects
cd projects
git clone https://github.com/rmahajani31/hubmap-hacking-the-human-vasculature.git

cp ~/full_training_data.zip ~/projects/hubmap-hacking-the-human-vasculature/
cd hubmap-hacking-the-human-vasculature
unzip full_training_data


sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p /mnt/disks/pre-installed
sudo mount -o discard,defaults /dev/sdb /mnt/disks/pre-installed
sudo chmod a+w /mnt/disks/

------------------------------------------------
aws


export TMPDIR='/var/tmp'
lsblk
sudo file -s /dev/xvdf
If the above command output shows “/dev/xvdf: data“, it means your volume is empty.

Step 6: Format the volume to the ext4 filesystem using the following command.
sudo mkfs -t ext4 /dev/xvdf
Alternatively, you can also use the xfs format. You have to use either ext4 or xfs.

 sudo mkfs -t xfs /dev/xvdf
Step 7: Create a directory of your choice to mount our new ext4 volume. I am using the name “newvolume“. You can name it something meaningful to you.

sudo mkdir /pre-installed
Step 8: Mount the volume to “newvolume” directory using the following command.

sudo mount /dev/xvdf /pre-installed/
Step 9: cd into newvolume directory and check the disk space to validate the volume mount.

cd /pre-installed
df -h .
The above command should show the free space in the newvolume directory.

To unmount the volume, use the unmount command as shown below..
sudo umount /dev/xvdf

sudo yum update
yum install gcc gcc-c++ make 

pip install gdown

gdown --id <put-the-ID>

