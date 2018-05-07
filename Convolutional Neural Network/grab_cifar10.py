##########################
# Fetching CIFAR 10 data #
##########################
import numpy as np
import glob, os

def unpickle(file):
    """
    python 3 version
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def make_data_into_one_batch():
    """
    Fetches training batch 1 and testing batch
    """

    data_dir = "/media/kodemannen/82684759-4f8f-4120-a5ed-a5055afc402d/CIFAR10/cifar-10-batches-py/"
    save_dir = "/media/kodemannen/82684759-4f8f-4120-a5ed-a5055afc402d/"
    test_name = "test_batch"

    
    data = []
    labels = []


    ting = os.walk(data_dir)
    for dings in list(ting)[0][2]:
        if "data_batch" in dings:
            filename = dings
            
            training_data = unpickle(data_dir + filename)

            train_imgs = np.transpose(training_data[b"data"])
            train_labels = training_data[b"labels"]

            data.append(train_imgs)
            labels.append(train_labels)

            # for inp, lbl in zip(train_imgs,train_labels):

            #     data.append(inp)
            #     labels.append(lbl) 

    data = np.array(data)
    print(data.shape)
    
    labels = np.array(labels)
    print(labels.shape)
    test_data = unpickle(data_dir + test_name) 
    test_imgs = np.transpose(test_data[b"data"])
    test_labels = test_data[b"labels"]
    
    savearray = np.array([data, labels, test_imgs, test_labels])
    
    np.save(save_dir + "cifar10_numpy_array", savearray)
    
    return {"training_images" : data, "training_labels" : labels, "test_images" : test_imgs, "test_labels" : test_labels}

if __name__ == "__main__":
    None
    #make_data_into_one_batch()