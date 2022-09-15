import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


class FeatureExtractor:
    def __init__(self, model, feature_vec_len=512):
        self.model = model
        self.to_tensor = transforms.ToTensor()
        # Get last layer containing embeddings
        self.layer = self.model._modules.get("net")._modules.get("avgpool")
        self.feature_vec_len = feature_vec_len

    def encode_train_images(self, train_path):
        feature_vectors = []

        folders = self._list_dir_nohidden(train_path)

        for folder in folders:
            folder_path = f"{train_path + folder}/"
            folder_files = [
                folder_path + filename
                for filename in self._list_dir_nohidden(folder_path)
            ]

            with torch.no_grad():  # no gradients calculation for the ops inside this block
                for file_path in folder_files:
                    img = Image.open(file_path)
                    feature_vec = self.extract_feature_vector(img)
                    feature_vectors.append(
                        {
                            "feature_vec": feature_vec.tolist(),
                            "img_path": file_path,
                            "brand": folder,
                        }
                    )

        return feature_vectors

    def extract_feature_vector(self, img):
        t_img = Variable(self.to_tensor(img).unsqueeze(0))
        my_embedding = torch.zeros(1, self.feature_vec_len, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.layer.register_forward_hook(copy_data)

        self.model(t_img)
        h.remove()

        return my_embedding.squeeze().numpy()

    def _list_dir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith("."):
                yield f