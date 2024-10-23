import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image_with_annotations(image_name, image_folder):
    """
    Display an image with its annotations.

    Parameters:
    - image_name: str, the name of the image file to display.
    - image_folder: str, the folder where the images are stored.

    Returns:
    - cv2_image: The annotated image.
    """
    # Load the COCO annotations from the JSON file
    with open("cyber2a/rts/rts_coco.json", "r") as f:
        rts_coco = json.load(f)

    # Get the image ID for the image
    image_id = None
    for image in rts_coco["images"]:
        if image["file_name"] == image_name:
            image_id = image["id"]
            break

    if image_id is None:
        raise ValueError(f"Image {image_name} not found in COCO JSON file.")

    # Get the annotations for the image
    annotations = []
    for annotation in rts_coco["annotations"]:
        if annotation["image_id"] == image_id:
            annotations.append(annotation["segmentation"])

    # Read the image
    cv2_image = cv2.imread(f"{image_folder}/{image_name}")
    if cv2_image is None:
        raise FileNotFoundError(
            f"Image file {image_name} not found in folder {image_folder}."
        )

    # Overlay the polygons on top of the image
    for annotation in annotations:
        for polygon in annotation:
            # Reshape polygon to an appropriate format for cv2.polylines
            polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
            cv2.polylines(
                cv2_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2
            )

    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    return cv2_image

def display_sample_images(dataset, num_images=3):
    """
    Display sample images from the dataset.

    Args:
        dataset (Dataset): The dataset to sample images from.
        num_images (int): Number of images to display.
        save_path (str): Path to save the displayed images.
    """
    data, label = dataset[0]
    if type(data) is dict:
        num_modalities = len(data)
        fig, axs = plt.subplots(num_modalities, num_images, figsize=(20, 5))
        for i in range(num_images):
            data, label = dataset[i]
            for j, modality in enumerate(data):
                axs[j, i].imshow(data[modality])
                if j == 0:
                    axs[j, i].set_title(f"label: {label}")
                else:
                    axs[j, i].set_title(f"modality: {modality}")
                axs[j, i].axis("off")

    else:
        fig, axs = plt.subplots(1, num_images, figsize=(20, 5))
        for i in range(num_images):
            data, label = dataset[i]
            axs[i].imshow(data)
            axs[i].set_title(f"Label: {label}")
            axs[i].axis("off")

    plt.show()

