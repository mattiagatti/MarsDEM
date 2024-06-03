# Monocular Depth Estimation to Predict Digital Elevation Model of the Lunar Surface

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