import os
import pandas as pd
import torch
from typing import Optional
import torchvision.transforms as transforms
import umap.umap_ as umap

from PIL import Image
from torchvision import models
from torchvision.models import VGG16_Weights

from src.datasets.constants import VAL_TEST_TRANSFORMS, VAL_TEST_TRANSFORMS_CAFFE
from src.model.test.extract_from_caffe import get_state_dict
from src.modules.finetune_dex import FinetuneDEX
from src.modules.saliency_utils import compute_gradcam, save_saliency_overlay

def test(model: FinetuneDEX, image_path: str, transform, results_dir: Optional[str] = None):
    """Performs forward passes through the model and saves embeddings

    Args:
        transform: VAL_TEST_TRANSFORMS or VAL_TEST_TRANSFORMS_CAFFE
    """
    with torch.no_grad():  # No need to calculate gradients during inference
        image = Image.open(image_path).convert("RGB")

        try:
            inp_tensor = torch.unsqueeze(transform(image), 0).to("cuda")
            output, embeddings = model.forward(
                inp_tensor,
                capture_embeddings=True,  # Captures embeddings at the last FC layer
            )

            # Move data frrom CUDA device back to CPU (required)
            output, embeddings = output.to("cpu"), embeddings.to("cpu")

            # DEX softmax expected value
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            age_range = torch.arange(101)
            estimated_age = (prob * age_range).sum()

            # optionally save saliency overlay (Grad-CAM) for this image
            if results_dir is not None:
                try:
                    # compute Grad-CAM with gradients enabled (we're running inside a
                    # torch.no_grad() block for inference; temporarily enable grads)
                    with torch.enable_grad():
                        # use last conv layer in VGG features by default
                        target_layer = model.model.features[-1]
                        cam = compute_gradcam(model.model, inp_tensor, target_layer)
                    saliency_dir = os.path.join(results_dir, 'saliency_overlays')
                    os.makedirs(saliency_dir, exist_ok=True)
                    out_path = os.path.join(saliency_dir, os.path.basename(image_path) + '_gradcam.png')
                    save_saliency_overlay(out_path, image, cam[0])
                except Exception as e:
                    print('Failed to save saliency for', image_path, e)

            return estimated_age.item(), torch.flatten(embeddings)
        except Exception as e:
            print(f"Error opening `{image_path}`:\n{e}")


def test_on_set(testing_set, model_path, results_dir: Optional[str] = None):
    """Tests a model on a testing set

    Args:
        testing_set:    APPAREAL or CHALEARN
        model_path:     Path to a .chkpt or .pth file.
    """

    # Get testing set from APPA-REAL, ChaLearn LAP 2015, or Filipino Faces
    test_images = None
    match testing_set:
        case "APPAREAL":
            test_images = pd.read_csv("data/appa-real-release/gt_avg_test.csv")
            test_images["file_name"] = (
                "data/appa-real-release/test/"
                + test_images["file_name"]
                + "_face.jpg"  # ..._face.jpg is the cropped face image
            )

            # Extract race and gender information
            race_sex = pd.read_csv("data/appa-real-release/allcategories_test.csv")
            race_sex = race_sex[["race", "gender"]]

            # Should be the same length
            test_images = pd.concat([test_images, race_sex], axis=1)
        case "CHALEARN":
            test_images = pd.read_csv("data/chalearn_2015/test.csv")
            test_images["file_name"] = (
                "data/chalearn_2015/test/" + test_images["file_name"]
            )
        case "FILIPINO_FACES":
            test_images = pd.read_csv("data/filipino_faces_dataset/labels.csv")
            test_images["file_name"] = (
                "data/filipino_faces_dataset/" + test_images["path"]
            )
            # Keep gender column as is
            test_images = test_images[["file_name", "age", "gender"]]

    assert test_images is not None, f"Testing set {testing_set} not recognized. Please choose APPAREAL, CHALEARN, or FILIPINO_FACES, or define a new testing set."

    # Load the model
    vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
    vgg16.classifier[6] = torch.nn.Linear(4096, 101)

    model = FinetuneDEX(model=vgg16, criterion=None).to("cuda")

    file_extension = os.path.splitext(model_path)[1]

    transform = VAL_TEST_TRANSFORMS
    if file_extension == ".ckpt":
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif file_extension == ".pth":
        transform = (
            VAL_TEST_TRANSFORMS_CAFFE  # Assume Caffe model, use different transforms
        )
        model.load_state_dict(get_state_dict(model_path))

    model.eval()  # Switches model to evaluation mode (needed for inference)

    # Get model outputs on testing set
    # Series of tuples containing age estimates and embeddings
    output = test_images["file_name"].progress_apply(
        lambda x: test(model, x, transform, results_dir=results_dir)
    )

    # Split into the estimate and embeddings columns
    test_images[["estimate", "embeddings"]] = pd.DataFrame(
        output.tolist(), index=test_images.index
    )

    return test_images
