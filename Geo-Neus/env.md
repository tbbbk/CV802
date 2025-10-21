```
conda create -n geoneus python=3.9
conda activate geoneus  
pip install torch==1.11.0 torchvision
pip install fvcore
pip install iopath
conda install -c bottler nvidiacub  
conda install pytorch3d==0.6.2 -c pytorch3d -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt