import torch
import torchvision.transforms as T

from PIL import Image

def predict_image(model, image_path):
    """
    Predict the class of a sample image.
    
    Args:
        model: The trained model.
        image_path (str): Path to the image to predict.
    
    Returns:
        int: Predicted class label.
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    # Apply the transformations and add a batch dimension
    image = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    image = image.to(device)

    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()