# Scar screenshots

Screenshot Cleanup Automation Routine (scar) : Python script that gets rid of all my old screenshots I leave lying around on my desktop and organize them for me in labeled folders

# Installation

To install the script, you can use pip:
```
pip install -r requirements.txt
```

# Usage

To run the script, use the following command:
```
python script.py
```

The script will automatically organize your screenshots into labeled folders. The script uses the OpenAI CLIP model to get vector embeddings of each screenshot. It then uses k-means clustering to classify the images.

## Features

- Automatic screenshot organization
- Vector embeddings calculation using OpenAI CLIP
- K-means clustering for image classification

## Dependencies

- Python
- torch
- clip
- numpy
- scikit-learn
- tqdm
- Pillow