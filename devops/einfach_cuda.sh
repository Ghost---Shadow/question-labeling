curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output ./devops/install_gpu_driver.py
sudo python3 ./devops/install_gpu_driver.py
sudo python3 ./devops/install_gpu_driver.py verify 
