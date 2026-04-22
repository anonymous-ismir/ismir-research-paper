**_We will use musicgen enviroment only for the project_**

python 3.10
make conda enviroment:

pip install git+https://github.com/declare-lab/TangoFlux

# for Jupyter kernel registration

pip install ipykernel
python -m ipykernel install --user --name <Env name> --display-name "<Env name to display>"
pip install spacy
python -m spacy download en_core_web_sm

for conda environment:
conda install -c conda-forge ffmpeg
