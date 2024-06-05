# Monocular Depth Estimation to Predict Topology of the Martian Surface

![alt text](https://www.ox.ac.uk/sites/files/oxford/styles/ow_medium_feature/s3/field/field_image_main/Banner%20image%20resized.jpg?itok=Zp2eLA3N)

Monocular depth estimation is the task of estimating the depth value (distance relative to the camera) of each pixel, given a single (monocular) image. Typically, the depth values of an image are obtained using stereoscopic algorithms, which require the use of a stereoscopic camera (a camera with two or more lenses). Digital Terrain Models (DTMs) are essential for planetary exploration and scientific research. They are digital representations of the elevations of a surface, from which it is possible to reconstruct a three-dimensional model of the area. The High Resolution Imaging Science Experiment (HiRISE) is a camera on board the Mars Reconnaissance Orbiter, which has been orbiting and studying Mars since 2006. By using its images of the same area on the ground, taken from different angles, many DTMs of some parts of the surface have been created. However, creating a DTM is complicated and requires sophisticated software and a lot of time, both computational and human. This project will investigate the use of monocular depth estimation models to predict the topology of Mars in a faster and cheaper way.

## Dataset
The dataset was scraped from the [UAHiRISE](https://https://www.uahirise.org/) website. The associated 1m/px 8-bit greyscale raster image (left or right image of the stereo pair) and Float32 raster DTM with the same spatial resolution were downloaded for each observation. The complexity of deep learning models is proportional to input size, and these rasters could reach very large shapes. Therefore, the image was split into smaller tiles, which will be merged after processing. In total, 160k patches were generated. Each patch is a 2-channel 512 x 512 raster, where the first channel is the image and the second channel is a DTM. The DTMs are stored as absolute values. However, the absolute heights must be converted to relative heights for use with a monocular depth estimation model. The dataset size is 500 GB.

## Results
The proposed model achived the following metrics:
| δ<sub>1<sub> | δ<sub>2<sub> | δ<sub>3<sub> | MAE | RMSE |
| ------ | ------ | ------ | ------- | ------- |
| 0.5502 | 0.7561 | 0.8488 | 10.1459 | 21.2730 |

Given that the relative heights of the Martian surface range from 0 to 700, and that these values are well distributed across all DTM pixels (but there are more lower heights), this delta one and mean absolute value show that the model is performing well. Of course, if these images were provided by a multispectral satellite that produces multi-band images, rather than a stereoscopic satellite that only produces greyscale images, the results could be further improved. In addition, the average time for predicting a single patch is in the millisecond range (depending on the accelerator used), making this approach faster than conventional stereoscopic algorithms. In conclusion, for this approach to be a valid solution, the tile merging approach has to be done cleverly, as relative heights have to be converted to absolute heights in a real case scenario. Intuitively, by predicting the heights of two tiles that have a small overlap, the difference in this overlap area allows them to be merged by rescaling their heights.

## Commands

Load the virtual environment and install the requirements:
``` console
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.11 python3.11-venv
cd MarsDEM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scrap mars data from [UAHiRISE](https://https://www.uahirise.org/):
``` console
nohup python scrape.py > download.log 2>&1 &
```

Divide generated tiles in train, validation and test splits:
``` console
python splits/generate_splits.py
```

Train the monocular depth estimation model:
``` console
nohup python train.py --batch_size 2 > train.log 2>&1 &
```

Try the demo:
``` console
python app.py
```

Build a docker image
``` console
docker build -f ./API_App/Dockerfile --tag marsdem .
```

Download and run last uploaded image
``` console
docker pull ghcr.io/mattiagatti/marsdem:latest
docker run -d -p 7860:7860 ghcr.io/mattiagatti/marsdem:latest
```
