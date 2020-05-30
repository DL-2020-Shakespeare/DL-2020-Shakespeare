# Deep Learning 2020 Project - Shakespeare

### Local install

* Python 3.7 or above enviroment required
* ``python3 -m venv venv``
* ``source venv/bin/activate``
* ``pip install -U pip setuptools wheel``
* ``pip install -r requirements.txt``
* ``python -m spacy download en_core_web_sm``
* Put these to train/:
    * [data.pkl](https://drive.google.com/uc?id=1LjxAbrjAjsQa1ss1Z56KseJafFsMinEL)
    * [preprocessed_docs_tokenized_cased.pkl](https://drive.google.com/uc?id=1cuEn_-oEv6AkD5H9FZ5BRKr2JjwAlqTD)
* Put these to text/:
    * [data.pkl](https://drive.google.com/uc?id=1sWoaq2UevAx85AX8vQuS0QMtd0EcV39t)
    * [preprocessed_docs_tokenized_cased.pkl](https://drive.google.com/uc?id=1fWh0Eyus3R1DozcDNPtivbfASH-89z5g)

### Using Puhti cluster's interactive jupyter server
 
* ``ssh <username>@puhti.csc.fi``
* ``cd /projappl/project_2002961/DL-2020-Shakespeare``
* ``module load python-data gcc/8.3.0 cuda/10.1.168 cudnn/7.6.1.34-10.1``
* ``source venv/bin/activate``
* ``sinteractive -A project_2002961 -c 20 -m 200000 -g 4 start-jupyter-server``
