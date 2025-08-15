import torch
import torchaudio
from inference import load_model, run_tts, load_audio, preprocess_wav

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt_path = "checkpoints/checkpoints.pth"
    config_path = "checkpoints/config.json"
    source_audio_path = "../filedepends/sounds/gene.wav"  # 一段 3-10 秒的干净人声 wav（16kHz）
    output_path = "../filedepends/sounds/gene_clone.wav"
    # Load the model
    print("Loading model...")
    model = torch.load(ckpt_path, map_location=device)
    model.eval()

    # Load the audio
    print("Loading audio...")
    ref_radio = load_audio(source_audio_path, target_sr=16000)
    ref_radio = preprocess_wav(ref_radio).to(device)
    speaker_embedding = torch.load("checkpoints/zh_default_se.pth", map_location=device).to(device)
    # Run the model
    print("Running model...")
    text = "森林里住着一只聪明的小猴子。"
    with torch.no_grad():
        audio = run_tts(
            model,
            text = text,
            ref_radio = ref_radio,
            src_lang = "zh",
            tgt_lang = "zh",
            speaker_embedding = speaker_embedding,
            device = device
        )

    # Save the audio
    print(f"Saving audio to {output_path}...")
    torchaudio.save(output_path, audio.unsqueeze(0), 16000)
    print("Done!")
