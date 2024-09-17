import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment

def generate_speech(text, file_name):

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text = text, return_tensors = "pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write(file_name, speech.numpy(), samplerate=16000)



def split_by_sentence(text, max_sent_length = 600):

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        
        if len(current_chunk) + len(sentence) <= max_sent_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def concat_audio_files(audio_chunks_files):
    total_audio = AudioSegment.empty()

    for audio_file in audio_chunks_files:
        audio = AudioSegment.from_wav(audio_file)
        total_audio += audio

    total_audio.export("final_speech.wav", format = "wav")
    print("combained audio saved as final_speech.wav")
    



def main():

    text = "Cloning is a process of creating genetically identical copies of an organism, cell, or DNA molecule. It occurs naturally in organisms like bacteria and plants through asexual reproduction, where offspring are exact genetic replicas of their parent. However, in the context of biotechnology, cloning typically refers to the deliberate scientific techniques used to replicate organisms, such as through reproductive cloning, where an organism is artificially produced with the same DNA as another.The most famous example of cloning is the birth of Dolly the sheep in 1996, the first mammal cloned from an adult somatic cell using a technique known as somatic cell nuclear transfer (SCNT). In this method, the nucleus of an adult cell is transferred into an egg cell that has had its nucleus removed, and the reconstructed egg is then stimulated to develop into an embryo, which is implanted into a surrogate mother. Dolly's successful cloning sparked both excitement and ethical debates about the implications of cloning in humans and animals.Cloning holds significant potential for advancing medical research, agriculture, and conservation efforts. For instance, scientists are exploring the use of cloning to produce genetically identical animals for organ transplantation, or to help preserve endangered species by creating genetically identical populations. However, it also raises ethical concerns related to identity, the welfare of cloned organisms, and the possible consequences of human cloning, which remains a highly controversial topic."

    chunks = split_by_sentence(text)

    for count, chunk in enumerate(chunks):
        file_name = f"speech_chunk_{count}.wav"
        generate_speech(chunk, file_name)
        print(f"Chunk {count} converted to speech: {file_name}")

    audio_chunks_files = [f"speech_chunk_{x}.wav" for x in range(len(chunks))]
    concat_audio_files(audio_chunks_files)

    print("Text to speech conversion complete. Please check the final_speech.wav file in your current directory.")


if __name__ == "__main__":
    main()
