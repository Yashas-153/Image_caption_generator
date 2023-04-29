from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import trainer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.save_pretrained("models/ED-Vit-gpt2")

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor.save_pretrained("models/IP-Vit-gpt2")

tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer.save_pretrained("models/AT-Vit-gpt2")

para_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
para_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

para_tokenizer.save_pretrained("models/AT-chaptgpt")
para_model.save_pretrained("models/AMS2S-chatgpt")

