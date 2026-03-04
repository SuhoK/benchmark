import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def main():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    img_path = "test.jpg"  # <-- 여기 바꿔도 됨
    img = Image.open(img_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80)

    print(processor.batch_decode(out, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()