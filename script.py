import os
import shutil
import torch
import clip
import numpy as np

from collections import defaultdict
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class script:
    def __init__(self , desktop_location):
        self.desktop_location = desktop_location
        
    def main(self):
        files = self.desktop_location

        screenshot = script.get_all_files(files)

        embeddings = [script.get_vector_embeddings(file) for file in screenshot]

        similarity = script.compare_embeddings(embeddings)

        clusters = script.classify_images(screenshot, similarity)
        
        script.make_folders_and_place_files(screenshot, clusters)

    def is_screenshot(files):
        '''
        Function to determine whether which file in the desktop are screenshots 
        '''
        try:
            with Image.open(files) as img:
                img.verify()
                return True
        except(IOError, SyntaxError):
            return False
        
    def get_all_files(directory):
        '''
        Append all files to list that are screenshots to list
        ''' 
        resulting_filenames = []

        for file in directory:
            filename = os.fsdecode(file)
            if script.is_screenshot(filename) and filename.endswith(('.png', '.jpeg', '.gif')): 
                resulting_filenames.append(filename)

        return resulting_filenames

    def get_vector_embeddings(image_path):
        '''
        Get Vector embeddings of each screenshot using OPENAI CLIP
        and compare the vectors of the files gotten using cosine similarity
        '''

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        images = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)

        # normalize embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # convert to numpy arrays
        return image_features.cpu().np()

    def compare_embeddings(list_of_embeddings):
        '''
        Function to compare the vectors of the files gotten using cosine similarity
        '''
        embeddings_array = np.array(list_of_embeddings)

        similarity = cosine_similarity(embeddings_array)

        return similarity

    def classify_images(screenshot_list, similarity_matrix):
        '''
        Classify the images based on the vector embeddings using k-means clusteriing
        '''
        resulting_dictionary = {}
        num_clusters = 2

        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(similarity_matrix)

        labels = kmeans.labels_

        for image_path, label in zip(screenshot_list, labels):
            print(f"IMAGE {image_path} belongs to the CLUSTER {label}")
            resulting_dictionary[image_path] = label
        return resulting_dictionary



    def make_folders_and_place_files(files, clusters):
        '''
        Process clusters by making folders and placing respective files there
        '''

        # group the images by their labels
        grouped_res = defaultdict(list)

        for image_path, label in clusters.items():
            grouped_res[label].append(image_path)
        #print(f"Grouped images by label is : {grouped_res}")

        # make folder in desktop based on the label name and place files there
        for image_path, label in grouped_res.items():
            newpath = '/Desktop/' + label
            if not os.path.exists(newpath):
                os.mkdir(newpath)
            shutil.move(os.path.join(newpath, os.path.basename(image_path)))
            
        

    def setup_cron_job():
        '''
        Function to run the script every 24 hours (or any other time periods that you want)
        '''
        pass

desktop_pathname = os.path.join(os.path.expanduser("~", "Desktop"))
run = script(desktop_pathname)
run.main() # run once 
# run.setup_cron_job() # run multiple times over a set period