import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 



if __name__ == "__main__":
    video = cv2.VideoCapture("input/c4.mp4")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # img = Image.fromarray(frame)
        # raw_image = Image.open(img_path).convert('RGB')
        img = Image.fromarray(frame)
        if img.mode != 'RGB':
            img = img.convert(mode="RGB")
        # conditional image captioning
        # text = ""
        # inputs = processor(img, text, return_tensors="pt")

        # out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))

        # unconditional image captioning
        inputs = processor(img, return_tensors="pt")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))