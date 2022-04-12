import cv2
import os
import argparse
import re
import shutil
from tqdm import tqdm
from cityscapesscripts.preparation import createTrainIdLabelImgs
def copy_json(base_path:str):
    base_path=os.path.join(base_path,"gtFine")
    tqdm(shutil.copytree("C:\projects\cnn_project\data\dataBackup\gtFine",base_path,dirs_exist_ok=True))
    print(f"JSON files has been copied to: {base_path}")
def delete_png(base_path:str):
    base_path=os.path.join(base_path,"gtFine")
    splits = ["test/", "train/", "val/"]
    for split in tqdm(splits):
        split_path = os.path.join(base_path, split)
        cities = os.listdir(split_path)
        for city in cities:
            city_path = os.path.join(split_path, city)
            files = os.listdir(os.path.join(city_path))
            for file in files:

                file_path = os.path.join(city_path, file)
                if re.search(".png", file_path):
                    os.remove(file_path)
    print("All no JSON files has been deleted")

def delete_json(base_path:str):
    base_path=os.path.join(base_path,"gtFine")
    splits = ["test/", "train/", "val/"]
    for split in tqdm(splits):
        split_path = os.path.join(base_path, split)
        cities = os.listdir(split_path)
        for city in cities:
            city_path = os.path.join(split_path, city)
            files = os.listdir(os.path.join(city_path))
            for file in files:

                file_path = os.path.join(city_path, file)
                if re.search(".json", file_path):
                    os.remove(file_path)
    print("All  JSON files has been deleted")


def main(args):
    delete_png(args.path)
    copy_json(args.path)
    createTrainIdLabelImgs.main(args.path)
    delete_json(args.path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract_JSON",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path',type=str,required=False,default="C:/projects/cnn_project/data/")
    main(parser.parse_args())




