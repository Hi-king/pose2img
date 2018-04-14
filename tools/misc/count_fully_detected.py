import numpy
import glob

dataset = "/mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox"

for path in glob.glob("/mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox/*.npz"):
    print(path)
    data = numpy.load(path)
    print(len([v for v in data if v is not None]))
