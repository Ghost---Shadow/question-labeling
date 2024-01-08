lsblk
sudo mkdir /mnt/checkpoints
# sudo mkfs.ext4 /dev/nvme0n2
sudo mount /dev/nvme0n2 /mnt/checkpoints
sudo mkdir /mnt/checkpoints/checkpoints
ln -s /mnt/checkpoints/checkpoints ./checkpoints
sudo chown -R $(whoami):$(whoami) /mnt/checkpoints/checkpoints
sudo chmod 766 /mnt/checkpoints/checkpoints
