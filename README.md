# How To Run the Code
## Download dataset
1. [CICAndMal2017](https://www.unb.ca/cic/datasets/andmal2017.html)
```
mkdir -p /scratch/Malware/CICAndMal
wget -c -i data/CICAndMal2017/cic_url.txt -P /scratch/Malware/CICAndMal
```
2. [IoT23](https://www.stratosphereips.org/datasets-iot23)
```
mkdir -p /scratch/Malware/iot23
wget -c -i data/IoT23/iot23_url.txt \
     -P /scratch/Malware/iot23/mal \
     -x -nH --cut-dirs=3
```

## Data Preprocess
CICAndAMl2017 : data/CICAndMal2017/CIC_preprocess.py (saves to /scratch/Malware/CICAndMal/processed_data/)
IoT23 : data/IoT23/store_by_capture.py -> capture_preprocess.py (saves to /scratch/Malware/iot23/data/)

## Usage
- training with default arguments(kmeans exemplar selection)

### XGBoost
- to run xgboost, follow all the setup steps from the original project [TraMEL](https://github.com/atg205/code) and run xgboost/main_xg.py
## Dependencies & Setup

A minimal set of Python packages is required to run the plotting and evaluation scripts. Create a virtual environment and install dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to run the our code

We have an other preprocessing step for the data preprocessing and then the code is run from the *main.py* file
```bash
python src/preprocess.py
python src/main.py
```

