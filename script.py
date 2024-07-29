import os
import shutil
import torch
import clip
import numpy as np
import sys
import json

from tqdm import tqdm
from PIL import Image
from crontab import CronTab
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class Script:

    def __init__(self , desktop_location):
        self.desktop_location = desktop_location
        
    def main(self):
        print("Starting Script")
        print(f"Do you want to run this script once or a cron job (every wednesday at 12:00) 
              \n 1. Once 
              \n 2. Multiple times
              \n 3. Delete cron job
              \n 4. Exit")

        choice = int(input("Enter your choice (1-4): "))

        if choice == 1:
            files = self.desktop_location

            screenshot = Script.get_all_files(files)

            embeddings = [Script.get_vector_embeddings(file) for file in screenshot]

            embeddings_array = np.array(embeddings)

            # write the embeddings to a file for examination
            np.savetxt('embeddings.txt', embeddings_array)
            
            similarity = Script.compare_embeddings(embeddings_array)

            np.savetxt('cosine_similarity.txt', similarity)

            clusters = Script.classify_images(screenshot, similarity)
            
            organization_result = Script.make_folders_and_place_files(self.desktop_location,screenshot, clusters)

            return clusters, organization_result
        elif choice == 2:
            self.setup_cron_job()
        elif choice == 3:
            self.delete_cron_job()
        elif choice == 4:
            print("Exiting Script")
            return None
        else:
            print("Invalid choice. pick between 1 - 4.")

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

        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if Script.is_screenshot(file_path) and file_path.endswith(('.png', '.jpeg', '.gif')): 
                resulting_filenames.append(file_path)

        return resulting_filenames

    def get_vector_embeddings(image_path):
        '''
        Get Vector embeddings of each screenshot using OPENAI CLIP
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        images = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)

        # normalize embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # convert to numpy arrays
        numpy_array = image_features.cpu().numpy().flatten()

        return numpy_array

    def compare_embeddings(emebeddings_array):
        '''
        Function to compare the vectors of the files gotten using cosine similarity
        '''
        similarity = cosine_similarity(emebeddings_array)
        
        return similarity

    def classify_images(screenshot_list, similarity_matrix):
        '''
        Classify the images based on the vector embeddings using k-means clustering
        and write the results to a file.
        '''
        resulting_dictionary = {}
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(similarity_matrix)
        labels = kmeans.labels_

        with open('locations.txt', 'w') as files:
            for image_path, label in zip(screenshot_list, labels):
                cluster_info = f"IMAGE {image_path} belongs to the CLUSTER {label}"
                print(cluster_info)
                files.write(cluster_info + '\n')
                resulting_dictionary[image_path] = int(label)

            # Write the entire dictionary to the file
            files.write('\n--- Full Classification Dictionary ---\n')
            json.dump(resulting_dictionary, files, indent=2)
        return resulting_dictionary
    
    def group_images_by_label(clusters):
        '''
        Group the images by their labels
        '''
        grouped_res = defaultdict(list)

        for image_path, label in clusters.items():
            grouped_res[label].append(image_path)
        print(f"Grouped images by label is : {grouped_res}\n\n")
        print(f"The number of keys is : {len(grouped_res.keys())}\n\n")
        return grouped_res

    def create_folders_and_place_files(desktop_location,folder_name, image_path):
        '''
        Process clusters by making folders and placing respective files there
        '''
        # make folder in desktop based on the label name and place files there
        newpath = os.path.join(desktop_location, folder_name)
        if not os.path.exists(newpath):
            os.mkdir(newpath)
        shutil.move(image_path, os.path.join(newpath, os.path.basename(image_path)))
            
    
    def make_folders_and_place_files(desktop_location,files, clusters):
        '''
        Process clusters by making folders and placing respective files there
        '''
        
        grouped_res = Script.group_images_by_label(clusters)

        files_moved = 0
        for image_paths, label in grouped_res.items():
            folder_name = f"cluster_{label}"
            for image_path in image_paths:
                Script.create_folders_and_place_files(desktop_location,folder_name, image_path)
                files_moved += 1

        summary = {
        "folders_created": len(set(clusters.values())),
        "files_moved": len(files)
        }
        return summary

    def setup_cron_job():
        '''
        Function to run the Script every 24 hours (or any other time periods that you want)
        '''

        cron = CronTab(user=True)
        existing_jobs = cron.find_comment("SCAR SCRIPT: Screenshpt Organization Script")

        if existing_jobs:
            print("Cron job for this script already exists. You can't create a duplicate one.")
            return
        
        # get path to current script and python interpreter
        script_path = os.path.abspath(__file__)
        python_path = sys.executable

        command = f"{python_path} {script_path}"

        job = cron.new(command=command, comment="SCAR SCRIPT: Screenshpt Organization Script")

        # set job to run every wednesday at 12:00
        job.setall("0 12 * * 3")

        cron.write()
        print("Cron job created successfully")
    
    def delete_cron_job():
        '''
        Function to delete the Script's cron job
        '''
        cron = CronTab(user=True)
        job = list(cron.find_comment("SCAR SCRIPT: Screenshpt Organization Script"))
        
        if not job:
            print("Cron job for this script doesn't exist.")
            return
        
        cron.remove(job[0])
        cron.write()
        print("Cron job deleted successfully")

desktop_pathname = os.path.join(os.path.expanduser("~"), "Desktop")
run = Script(desktop_pathname)
run.main() # run once or multiple times