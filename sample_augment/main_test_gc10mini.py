import matplotlib.pyplot as plt
import torch
import torchvision.transforms

if __name__ == '__main__':
    # look at some images from complete set and train set
    train_set = torch.load("/home/nils/thesis/sample-augment/data/store_37f69/AugmentDataset_tensors_0.pt")
    print(train_set.dtype)
    image = torchvision.transforms.ToPILImage()(train_set[3])
    plt.imshow(image)
    plt.show()


"""
TODO liste für diese Session:
- root_directory und target endlich aus der config raus
- evtl. Input-Artefakte konsumieren lassen in steps? also dass man sie safe ändern kann
    - sonst irgendeine andere Lösung finden, kann nicht sein dass dataset geändert wird
    - einfachste Lösung fürs erste: EINFACH NICHT DIE ARTEFAKTE BEARBEITEN
- Bei mini GC10 die Bilder bei allen relevanten Schritten ansehen und validieren
"""