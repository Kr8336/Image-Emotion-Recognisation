import os
import json
import torch

from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


def predict_step(image_paths, gen_kwargs, feature_extractor, tokenizer, model):
    images = []

    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to('cpu')

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def generate_image_captions(folder_name, save_text):

    dataset = []
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    classes = os.listdir(folder_name)
    # string_object = []

    for i, clas in enumerate(classes):
        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)

        # class_strings = []
        print('Generating Captions for class', clas)

        batch_text = []
        for file in tqdm(files):
            file_name = os.path.join(class_name, file)
            batch_text.append(file_name)
            if len(batch_text) % 16 == 0:
                text = predict_step(batch_text, gen_kwargs, feature_extractor, tokenizer, model)
                # class_strings.append(text)
                for txt in text:
                    dataset.append((i, txt))
                batch_text.clear()
        save_captions_to_file = clas + '_captions.json'
        with open(os.path.join(save_text, save_captions_to_file), 'w') as file:
            print('Generated Image Captions!')
            json.dump(dataset, file)
        # string_object.append(class_strings)

    return dataset
