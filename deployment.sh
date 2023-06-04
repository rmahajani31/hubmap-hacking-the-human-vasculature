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

