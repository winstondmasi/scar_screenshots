import os
import shutil
import torch
import clip
import numpy as np
import schedule
import json

from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

class Script:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = clip.load("ViT-B/32", device=device)
    preprocess = clip.load("ViT-B/32", device=device)

    # load dataset
    root = os.path.expanduser("~/.cache")
    train = CIFAR100(root, download=True, train=True, transform=preprocess)
    test = CIFAR100(root, download=True, train=True, transform=preprocess)

    def __init__(self , desktop_location):
        self.desktop_location = desktop_location
        
    def main(self):
        files = self.desktop_location

        screenshot = Script.get_all_files(files)

        embeddings = [Script.get_vector_embeddings(file) for file in screenshot]

        embeddings_array = np.array(embeddings)

        # write the embeddings to a file for examination
        np.savetxt('embeddings.txt', embeddings_array)
        
        similarity = Script.compare_embeddings(embeddings_array)

        np.savetxt('cosine_similarity.txt', similarity)

        clusters = Script.classify_images(screenshot, similarity)
        
        # make_files = Script.make_folders_and_place_files(screenshot, clusters)

        return clusters

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

    def get_features_from_dataset(dataset):
        '''
        Function to extract features and labels from Dataset: see lines 24 - 28 (feel free to change dataset at ones discreation)
        '''
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = Script.model.encode_image(images.to(Script.device))
                
                all_features.append(features)
                all_labels.append(labels)
        return torch.cat(all_features).cpu().numpy()
    
    def calc_image_features(map_of_clusters):
        '''
        Calculate the image features of each cluster and perform logistic regression classification
        '''

        train_features, train_labels = Script.get_features_from_dataset(Script.train)
        test_features, test_labels = Script.get_features_from_dataset(Script.test)

        classifier = LogisticRegression(random_state=0).fit(train_features, train_labels)
        predictions = classifier.predict(test_features)

        accuracy = np.mean((test_labels == predictions).astype(np.float) * 100)
        print(f"Accuracy is : {accuracy:.3f}%")

        return predictions
    
    def get_keywords_from_predictions(predictions):
        '''
        Get the keywords from the predictions
        '''
        pass

    def create_folders_and_place_files(folder_name, image_path):
        '''
        Process clusters by making folders and placing respective files there
        '''
        # make folder in desktop based on the label name and place files there
        newpath = '/Desktop/' + folder_name
        if not os.path.exists(newpath):
            os.mkdir(newpath)
        shutil.move(os.path.join(newpath, os.path.basename(image_path)))
            
    
    def make_folders_and_place_files(files, clusters):
        '''
        Process clusters by making folders and placing respective files there
        '''
        
        grouped_res = Script.group_images_by_label(clusters)

        extracted_featues_dict = Script.calc_image_features(grouped_res)
        gotten_keywords = Script.get_keywords_from_predictions(extracted_featues_dict)

        for image_path, label in grouped_res.items():
            folder_name = gotten_keywords.get(label, f"cluster_{label}")
            Script.create_folders_and_place_files(folder_name, image_path)

    def setup_cron_job():
        '''
        Function to run the Script every 24 hours (or any other time periods that you want)
        '''
        schedule.every().wednesday.at("12:00").do(Script.main)

desktop_pathname = os.path.join(os.path.expanduser("~"), "Desktop")
run = Script(desktop_pathname)
run.main() # run once
# run.setup_cron_job() # run multiple times over a set period