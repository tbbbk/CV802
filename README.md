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