import numpy as np
import librosa

class VADSNRAnalyzer:
    def __init__(self, frame_length=2048, hop_length=512, energy_threshold=0.01):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold

    def analyze(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        energy = np.array([np.sum(y[i:i+self.frame_length]**2) 
                           for i in range(0, len(y)-self.frame_length, self.hop_length)])
        
        speech_frames = energy > (np.max(energy) * self.energy_threshold)
        
        # Грубое выделение сегментов (для avg_segment_snr)
        segment_snrs = []
        in_speech = False
        start = 0
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start = i
                in_speech = True
            elif not is_speech and in_speech:
                end = i
                seg_energy = energy[start:end]
                noise_energy = np.median(energy[~speech_frames]) if np.any(~speech_frames) else 1e-10
                snr = 10 * np.log10(np.mean(seg_energy) / noise_energy)
                segment_snrs.append(snr)
                in_speech = False
        if in_speech:
            end = len(speech_frames)
            seg_energy = energy[start:end]
            noise_energy = np.median(energy[~speech_frames]) if np.any(~speech_frames) else 1e-10
            snr = 10 * np.log10(np.mean(seg_energy) / noise_energy)
            segment_snrs.append(snr)

        speech_ratio = np.sum(speech_frames) / len(speech_frames) if len(speech_frames) > 0 else 0
        speech_duration_sec = (np.sum(speech_frames) * self.hop_length) / sr
        duration_sec = len(y) / sr
        noise_energy = np.median(energy[~speech_frames]) if np.any(~speech_frames) else 1e-10
        snr_db = 10 * np.log10(np.max(energy) / noise_energy) if noise_energy > 0 else 0
        avg_segment_snr = np.mean(segment_snrs) if segment_snrs else snr_db

        return {
            'speech_ratio': speech_ratio,
            'snr_db': snr_db,
            'speech_duration_sec': speech_duration_sec,
            'duration_sec': duration_sec,
            'avg_segment_snr': avg_segment_snr
        }