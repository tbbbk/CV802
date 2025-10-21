Bro use HPC. Bro cannot see the GUI window. Bro use the MacBook.

# Environment Management
Use [uv](https://docs.astral.sh/uv/getting-started/installation/#pypi) to manage the python environment. Initialize the environment by 
```bash
uv sync
```
Activate the environement by
```bash
source ./.venv/bin/activate
```
# Workflow Guide
Make sure you are working on the `dev` branch.
```bash
git checkout dev
```
Pull the latest update
```bash
git pull
```
Commit your changes to the `dev` branch.
```bash
git add .
git commit -m "Your commit message"
git push
```
Create a Pull Request (PR) on GitHub.
# Basic Method
Please refer to [vgg-t](https://vgg-t.github.io).


# Subproject2 Surface Reconstruction
* Local Part: additional format convertion, point cloud cleaning, data preparation for NeuS/GeoNeuS, and mesh visualization.
* Server Part: GeoNeuS and Textured-NeuS for training and evaluation of textured Mesh respectively.

## Workflow of Subproject2
1. Run `python main.py` with GUI, just like Subproject1, to obtain the sparse point cloud and camera poses. An extra preproccessed folder will be generated under the data folder. It may take longer than before, about 10s before you can see the point cloud.
2. Move the preproccessed folder to a server/laptop with CUDA support.
3. Run GeoNeuS on the server, which will start training. 
```
python exp_runner.py --mode train --conf ./confs/womask.conf --case preprocessed
```
Typically, it takes 40 minutes to run 30k steps on a RTX A6000 GPU to obtain good mesh.
Run 300k steps (about 7 hours) if you want state-of-the-art quality mesh.
4. After training, copy the generated exp folder to the same position of Textured-NeuS repo. Run Textured-NeuS on the server to obtain high resolution textured mesh. 
```
python exp_runner.py --mode validate_mesh --case preprocessed --is_continue
```
This is important, as the mesh generated when training is lack of texture and in low resolution for quick visualization.
5. Move the generated `.ply` file back to original data folder on your MacBook.
6. Visualize the generated mesh by clicking the `Show Constructed Mesh` button on the GUI.
7. Tips: Sometimes NCC Loss will cause training instability. When this happens, you can try to disable NCC Loss by commenting the corresponding lines in `exp_runner.py` in GeoNeuS.

## Additional Environment of Subproject2 Local Part
```
uv pip install trimesh opencv-python
```

## Suggestion for GeoNeuS/Textured-NeuS Environment
You can follow the instruction in [GeoNeuS] to set up the environment with CUDA support.
However, due to old version of some dependencies and CUDA version compatibility issues, you may need to try different versions multiple times to find a working combination for your device.

A series of commands working on my server is:
```
conda create -n geoneus python=3.9
conda activate geoneus  
pip install torch==1.11.0 torchvision
pip install fvcore
pip install iopath
conda install -c bottler nvidiacub  
conda install pytorch3d==0.6.2 -c pytorch3d -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt
```
