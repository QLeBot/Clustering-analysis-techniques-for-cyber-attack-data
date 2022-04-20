FROM continuumio/miniconda3:4.10.3p1

RUN conda install -c conda-forge\
    numpy==1.21.2\
    pandas==1.4.1\
    matplotlib==3.5.1\
    scikit-learn==1.0.2 -y