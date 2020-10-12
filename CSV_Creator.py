import csv
import os


def build_csv(root, csv_file):
    """Creates a CSV file for a given path to a data directory which sorted to classes by folders"""

    clss = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]  # folders
    with open(csv_file, "w", newline='') as file:
        writer = csv.writer(file)
        for i, cls in enumerate(clss, start=0   ):
            imgs = [name for name in os.listdir(os.path.join(root, cls)) if name.endswith(('jpg', 'jpeg', 'png'))]
            for img in imgs:
                writer.writerow([os.path.join(root, cls, img), str(i)])
    print('\nCSV file created for the requested dataset\n')


data_root = 'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/Custom_Data'
csv_loc = 'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style_control.csv'

build_csv(root=data_root, csv_file=csv_loc)
