import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# ========== SETUP ==========
print("Loading models...")

# Load using safetensors to avoid torch.load vulnerability
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-vqa-base",
    use_safetensors=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
vqa_model.to(device)

# ========== FUNCTIONS ==========

def load_image(path_or_url):
    """Load image from local path or URL"""
    try:
        if path_or_url.startswith("http"):
            image = Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(path_or_url).convert("RGB")
        return image
    except Exception as e:
        print(f"[Error] Could not load image: {e}")
        return None

def generate_caption(image):
    """Generate caption for the image"""
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def visual_qa(image, question):
    """Answer a question based on the image"""
    inputs = vqa_processor(image, question, return_tensors="pt").to(device)
    out = vqa_model.generate(**inputs)
    return vqa_processor.decode(out[0], skip_special_tokens=True)

def display_header():
    print("=" * 60)
    print("🤖  BLIP Multimodal Application – Image Captioning + VQA")
    print("=" * 60)

def main():
    display_header()
    while True:
        print("\nOptions:\n1. Generate Image Caption\n2. Visual Question Answering\n3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice not in ["1", "2", "3"]:
            print("Invalid choice. Try again.")
            continue
        if choice == "3":
            print("Exiting application.")
            break

        img_path = input("Enter image path or URL: ").strip()
        image = load_image(img_path)
        if not image:
            continue

        if choice == "1":
            print("Generating caption...")
            caption = generate_caption(image)
            print(f"\n🖼️ Caption: {caption}")
        elif choice == "2":
            question = input("Enter your question: ")
            print("Generating answer...")
            answer = visual_qa(image, question)
            print(f"\n❓ Question: {question}")
            print(f"🤖 Answer: {answer}")

        print("\n--- Task Complete ---")

if __name__ == "__main__":
    main()
