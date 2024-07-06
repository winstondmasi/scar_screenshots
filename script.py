import os
import torch
import clip
from PIL import Image

def main():
    files = access_desktop

    screenshot = [file for file in files if is_screenshot(file)]

    embeddings = [get_vector_embeddings(file) for file in screenshot]

    similarity = compare_embeddings(embeddings)

    clusters = classify_images(similarity)
    
    make_folders_and_place_files(screenshot, clusters)

    setup_cron_job()

def access_desktop():
    '''
    Function that allow files to have access to desktop where the files are located
    '''
    pass

def is_screenshot():
    '''
    Function to determine whether which file in the desktop are screenshots 
    '''
    pass

def get_vector_embeddings():
    '''
    Get Vector embeddings of each screenshot using OPENAI CLIP
    '''
    pass

def compare_embeddings():
    '''
    Function to compare the vectors of the files gotten using cosine similarity
    '''

def classify_images():
    '''
    Classify the images based on the vector embeddings using k-means clusteriing
    '''
    pass

def make_folders_and_place_files(files, clusters):
    '''
    Process clusters by making folders and placinf respective files there
    '''
    pass


def setup_cron_job():
    '''
    Function to run the script every 24 hours (or any other time periods that you want)
    '''
    pass

if __name__ == "__main__":
    main()