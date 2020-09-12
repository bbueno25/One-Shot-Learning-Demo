"""
Running this demo should lead to a result of 38.8% average error rate.
    
NOTE: Models should be trained on images in 'images_background' directory
    to avoid using images and alphabets used in the one-shot evaluation.
"""
import numpy
import os
import scipy

nrun = 20
path_to_script_dir = os.path.dirname(os.path.realpath(__file__))
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'

def classification_run(folder, f_load, f_cost, ftype='cost'):
    """
    Compute error rate for one run of one-shot classification
    
    Input
        folder: contains images for a run of one-shot classification
        f_load: itemA = f_load('file.png') should read in the image file and process it
        f_cost: f_cost(itemA,itemB) should compute similarity between two
            images, using output of f_load
        ftype: 'cost' if small values from f_cost mean more similar,
            or 'score' if large values are more similar
    
    Output
        perror: percent errors (0 to 100% error)
    """
    assert ftype in {'cost', 'score'}
    with open(os.path.join(path_to_all_runs, folder, fname_label)) as f:
        pairs = [line.split() for line in f.readlines()]
    test_files, train_files = zip(*pairs)
    answers_files = list(train_files)
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    n_train = len(train_files)
    n_test = len(test_files)
    train_items = [f_load(os.path.join(path_to_all_runs, f)) for f in train_files]
    test_items = [f_load(os.path.join(path_to_all_runs, f)) for f in test_files]
    costM = numpy.zeros((n_test, n_train))
    for i, test_i in enumerate(test_items):
        for j, train_j in enumerate(train_items):
            costM[i, j] = f_cost(test_i, train_j)
    if ftype == 'cost':
        y_hats = numpy.argmin(costM, axis=1)
    elif ftype == 'score':
        y_hats = numpy.argmax(costM, axis=1)
    else:
        raise ValueError('Unexpected ftype: {}'.format(ftype))
    correct = len([1 for y_hat, answer in zip(
        y_hats, answers_files) if train_files[y_hat] == answer])
    pcorrect = correct / float(n_test)
    perror = 1.0 - pcorrect
    return perror * 100

def load_img_as_points(filename):
    """
    Load image file and return coordinates of black pixels in the binary image
    
    Input
        filename: string, absolute path to image
    
    Output:
        D: [n x 2] rows are coordinates
    """
    I = scipy.ndimage.imread(filename, flatten=True)
    I = ~numpy.array(I, dtype=numpy.bool)
    D = numpy.array(I.nonzero()).T
    return D - D.mean(axis=0)

def modified_hausdorf_distance(itemA, itemB):
    """
    Modified Hausdorff Distance
    
    Input
        itemA : [n x 2] coordinates of black pixels
        itemB : [m x 2] coordinates of black pixels
    
        M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
        International Conference on Pattern Recognition, pp. 566-568.
    """
    D = scipy.spatial.distance.cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = numpy.mean(mindist_A)
    mean_B = numpy.mean(mindist_B)
    return max(mean_A, mean_B)

def rename_images(src_dir, new_prefix):
    """
    DOCSTRING
    """
    for file_name in os.listdir(src_dir):
        os.rename(os.path.join(src_dir, file_name),
                  os.path.join(src_dir, new_prefix + file_name))
        print(file_name + ' -> ' + new_prefix + file_name)

if __name__ == '__main__':
    print('One-shot classification demo with Modified Hausdorff Distance')
    perror = numpy.zeros(nrun)
    for r in range(nrun):
        perror[r] = classification_run(
            'run{:02d}'.format(r + 1),
            load_img_as_points,
            modified_hausdorf_distance,
            'cost')
        print(' run {:02d} (error {:.1f}%)'.format(r, perror[r]))
    total = numpy.mean(perror)
    print('Average error {:.1f}%'.format(total))
