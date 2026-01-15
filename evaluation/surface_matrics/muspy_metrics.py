import muspy

def extract_muspy_metrics(midi_path):
    music = muspy.read_midi(midi_path)

    return {
        "pitch_entropy": muspy.pitch_entropy(music),
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "note_density": muspy.note_density(music),
        "polyphony_rate": muspy.polyphony_rate(music),
        "scale_consistency": muspy.scale_consistency(music),
        "pitch_range": muspy.pitch_range(music),
    }
