# Monocular Depth Estimation to Predict Topology of the Martian Surface

``` console
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.11 python3.11-venv
cd MarsDEM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

``` console
nohup python scrape.py > download.log 2>&1 &
```

``` console
python splits/generate_splits.py
```

``` console
nohup python train.py --model glpdepth --data_dir PATH --batch_size 8 > train.log 2>&1 &
```

