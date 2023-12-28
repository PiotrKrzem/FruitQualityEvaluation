import cv2
import os

from xml.etree.ElementTree import parse as parse_annotation
from shutil import rmtree as delete_contents

from src.const import *

OFFSETS = [(0,0)] #[(-5,0),(0,5),(0,0),(0,-5),(5,0)]:

def perform_data_augmentation():
    pass
    # ssize = input("[INFO] Target image size (single integer, empty for default 64): ")
    # if(ssize == ""): ssize = "64"
    # size = int(ssize)

    # if not os.path.exists(OUTPUTS_PATH):
    #     os.mkdir(OUTPUTS_PATH)
    # if not os.path.exists(PREPRC_PATH):
    #     os.mkdir(PREPRC_PATH)

    # if not os.path.exists(OUTPUTS_PATH+ssize) and not os.path.exists(OUTPUTS_PATH+ssize+"-test"):
    #     os.mkdir(OUTPUTS_PATH+ssize)
    #     os.mkdir(OUTPUTS_PATH+ssize+"-test")
    # else:
    #     decision = input("[INFO] This process will delete the previous augmentation results for this image size. Continue? Y/N: ")
    #     if(not decision.lower() == "y"): exit(1)

    #     print("[INFO] Deleting previous preprocessing results...")
    #     if(os.path.exists(OUTPUTS_PATH+ssize)):
    #         delete_contents(OUTPUTS_PATH+ssize)
    #     if(os.path.exists(OUTPUTS_PATH+ssize+"-test")):
    #         delete_contents(OUTPUTS_PATH+ssize+"-test")

    #     os.mkdir(OUTPUTS_PATH+ssize)
    #     os.mkdir(OUTPUTS_PATH+ssize+"-test")

    # fruitgos_directory = os.listdir(IMAGES_PATH)

    # fruitgos_files  = ['']*len(fruits_directory)
    # fruitgos_folders = ['']*len(fruits_directory)

    # paths_file = open(PREPRC_PATH+ssize+".csv", 'w')
    # paths_tests_file = open(PREPRC_PATH+ssize+"-test.csv", 'w')
    # paths_file.write("fruit_path;fruit_quality\n")
    # paths_tests_file.write("fruit_path;fruit_quality\n")

    # count = 0
    # for iterator, fruit_folder in enumerate(fruitgos_directory):
    #     fruit_images = os.listdir(IMAGES_PATH+fruit_folder)
    #     count += len(fruit_images)

    #     fruitgos_files[iterator] = fruit_images
    #     fruitgos_folders[iterator] = fruit_folder
    
    # i = 0
    # for folder, files in zip(fruits_folders, fruits_files):
    #     for fruits_file in files:
    #         if(i%10<2):
    #             save_as_test_image(folder, fruits_file, paths_tests_file, size, str(i))
    #         else:
    #             augment_image(folder, fruits_file, paths_file, size, str(i))
    #         print_animated_loader(i, count)
    #         i += 1
    #         if DEBUG: break
    #     if DEBUG: break
        
    # paths_file.close()
    print("\n[HOORAY] Process completed successfully!")

def print_animated_loader(i, count):
    if (i%8<=1):  c = '\\'
    elif(i%8<=3): c = '|'
    elif(i%8<=5): c = '/'
    else:         c = '-'
    
    print("\r[LOAD] {} Processing and augmenting images... {}/{} {}".format(c,i+1,count,c),end="")

if __name__ == "__main__":
    perform_data_augmentation()