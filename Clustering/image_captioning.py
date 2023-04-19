import os
import torch

from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def predict_step(image_paths, gen_kwargs, device='cpu'):

      images = []
      feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
      tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
      model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
      model.to(device)

      for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
          i_image = i_image.convert(mode="RGB")

        images.append(i_image)

      pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
      pixel_values = pixel_values.to(device)

      output_ids = model.generate(pixel_values, **gen_kwargs)

      preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
      preds = [pred.strip() for pred in preds]

      return preds



def generate_image_captions(folder_name):

    dataset = []
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = os.listdir(folder_name)
    string_object = []


    for i, clas in enumerate(classes):
        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)

        class_strings = []

        for file in files:
            file_name = os.path.join(class_name, file)

            text = predict_step([file_name], gen_kwargs, device) # ['a woman in a hospital bed with a woman in a hospital bed']
            class_strings.append(text)

            dataset.append((i, text))

        string_object.append(class_strings)

    return string_object, dataset


