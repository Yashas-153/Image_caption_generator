#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

para_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
para_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

max_length = 20
num_beams = 5
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
        
    inputs = processor(image,return_tensors="pt")
    
    out = model.generate(**inputs)
    
    return processor.decode(out[0], skip_special_tokens=True)


def paraphrase(
    sentence,
    num_beams=2,
    num_beam_groups=2,
    num_return_sequences=2,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = para_tokenizer(
        f'paraphrase: {sentence}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    outputs = para_model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = para_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

url = 'Image1.png'


def get_captions(image, n):
    print(f"n is {n}")
    print("Called the function")
    captions = []
    main_caption = predict_caption(image)
    if n == 1:
        captions.append(main_caption)
        print(main_caption)
    elif n == 2:
        captions.append(main_caption)
        para_captions = paraphrase(main_caption,num_beams=n, num_beam_groups=n,num_return_sequences=n)
        captions.append(para_captions[0])
    else:
        captions.append(main_caption)
        for caption in paraphrase(main_caption,num_beams=n-1, num_beam_groups=n-1,num_return_sequences=n-1):
            captions.append(caption)
    return captions 


