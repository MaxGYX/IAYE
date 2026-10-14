import openai
import os
import time
import io
from picamzero import Camera
from openai import OpenAI
import base64
import dashscope
from dashscope.audio.tts_v2 import *

# Initialize the Picamzero object
cam = Camera()
# cam.start_preview()
# cam.take_photo("./image.jpg")
# cam.stop_preview()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("sk-c3dc5b2f0aed4befb3371c0756d0c1d7"),
    api_key="sk-c3dc5b2f0aed4befb3371c0756d0c1d7",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

dashscope.api_key = "sk-c3dc5b2f0aed4befb3371c0756d0c1d7"
model = "cosyvoice-v1"
voice = "longtong"

synthesizer = SpeechSynthesizer(model=model, voice=voice, format=AudioFormat.WAV_22050HZ_MONO_16BIT)

def capture_image():
    """Capture an image and return it as a byte stream."""
    print("Capturing image...")
    cam.take_photo("./image.jpg")

def get_image_description():
    """Send the image to GPT via API for description."""
    '''
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{"role": "user","content": [
                {"type": "text","text": "请你为盲人描述一下这张图片"},
                {"type": "image_url",
                 "image_url": {"url": "./image.jpg"}}
                ]}]
    )
    '''
    base64_image = encode_image("./image.jpg")
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "请你为一个盲人儿童描述一下这张照片，帮助盲人儿童理解和欣赏这个世界"},
                ],
            }
        ],
    )
    
    description = response.choices[0].message.content
    return description

def main():
    print("Starting the Vision-to-Voice System...")
    while True:
        # Capture an image
        capture_image()

        # Get description from GPT API
        print("Getting image description...")
        description = get_image_description()
        print(f"Description: {description}")

        # Convert description to speech
        print("Converting to speech...")
        audio = synthesizer.call(description)
        print('requestId: ', synthesizer.get_last_request_id())
        with open('response.wav', 'wb') as f:
            f.write(audio)
        os.system("aplay ./response.wav")

        # Wait 10 seconds before the next capture
        # time.sleep(30)
        break

if __name__ == "__main__":
    main()

