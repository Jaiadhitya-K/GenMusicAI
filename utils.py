import google.generativeai as genai
from dotenv import load_dotenv
import os
import subprocess
import librosa
import json
import re

load_dotenv()
my_api_key = os.getenv("API_KEY")
genai.configure(api_key=my_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

PARAM_LIMITS = {
    "EQ": {"frequency": (20, 20000), "gain": (-12, 12)},
    "Gain": {"gain_level": (-12, 12)},
    "Reverb": {"reverb_amount": (0, 20)},
    "Tempo": {"tempo_factor": (0.5, 2.0)},
}


def interpret_audio_with_tools(prompt):
    example_format = """
    Here is an example of the format that you must strictly output the response in:
    {
        "EQ": [
            {"frequency": 100, "gain": -12},
            {"frequency": 2500, "gain": 6}
        ],
        "Reverb": [
            {"reverb_amount": 5}
        ],
        "Gain": [
            {"gain_level": 3}
        ],
        "Tempo": [
            {"tempo_factor": 1.2}
        ],
    }
    Please only use the following tools: EQ, Reverb, Gain, Tempo.
    Ensure all parameter values are within valid ranges:
    - EQ frequency: 20 to 20000 Hz, gain: -12 to +12 dB
    - Gain: -12 to +12 dB
    - Reverb amount: 0 to 20
    - Tempo factor: 0.5 to 2.0
    Response must be in JSON format.
    """
    final_prompt = f"{prompt}\n{example_format}"

    try:
        print("Sending prompt to Gemini...")
        response = model.generate_content(final_prompt)
        print("Response received from Gemini!")
        return response
    except Exception as e:
        print(f"Error in communicating with Gemini model: {e}")
        return None


def process_audio_file(file_path):
    try:
        print(f"Loading audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)

        print(f"Audio processing complete: Tempo: {tempo}, Duration: {duration}")
        return {
            "file_name": file_path,
            "tempo": tempo,
            "duration": duration,
        }
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None


def analyze_audio_and_send_to_gemini(file_path, user_prompt):
    audio_metadata = process_audio_file(file_path)

    if audio_metadata:
        metadata_description = (
            f"The track has a tempo of {audio_metadata['tempo']} BPM and a duration of "
            f"{audio_metadata['duration']} seconds. Based on these elements, suggest audio modifications."
        )

        final_prompt = f"{user_prompt}. {metadata_description}"
        response = interpret_audio_with_tools(final_prompt)

        if response:
            return response
        else:
            return "Error: Failed to get response from Gemini."
    else:
        return "Error: Unable to process the audio file."


def validate_audio_effects(effects):
    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    for effect, params in effects.items():
        if effect in PARAM_LIMITS:
            for param in params:
                for key, value in param.items():
                    if key in PARAM_LIMITS[effect]:
                        min_val, max_val = PARAM_LIMITS[effect][key]
                        param[key] = clamp(value, min_val, max_val)
    return effects


def parse_gemini_response(response):
    try:
        modifications_text = response.candidates[0].content.parts[0].text
        print("Raw Gemini Response:\n", modifications_text)

        json_match = re.search(r"\{.*\}", modifications_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            cleaned_json = json.loads(json_text)
            print("Extracted JSON for modifications:\n", cleaned_json)
            return cleaned_json
        else:
            print("Error: No valid JSON found in the response.")
            return None

    except Exception as e:
        print(f"Error parsing Gemini response: {str(e)}")
        return None


def apply_suggested_modifications(file_path, gemini_response):
    suggestions = parse_gemini_response(gemini_response)

    if suggestions is None:
        print("No valid suggestions parsed from the response.")
        return None

    print("Applying the following modifications based on Gemini's suggestions:")
    print(suggestions)

    try:
        suggestions = validate_audio_effects(suggestions)

        ffmpeg_filters = []

        for effect, params in suggestions.items():
            if effect == "EQ":
                for param in params:
                    frequency = param.get("frequency")
                    gain = param.get("gain")
                    eq_filter = f"equalizer=f={frequency}:t=q:w=1:g={gain}"
                    ffmpeg_filters.append(eq_filter)

            elif effect == "Gain":
                for param in params:
                    gain_level = param.get("gain_level")
                    gain_filter = f"volume={gain_level}dB"
                    ffmpeg_filters.append(gain_filter)

            elif effect == "Reverb":
                for param in params:
                    reverb_amount = param.get("reverb_amount")
                    reverb_filter = f"aecho=1.0:0.5:{reverb_amount}:0.5"
                    ffmpeg_filters.append(reverb_filter)

            elif effect == "Tempo":
                for param in params:
                    tempo_factor = param.get("tempo_factor")
                    tempo_filter = f"atempo={tempo_factor}"
                    ffmpeg_filters.append(tempo_filter)

        combined_filters = ",".join(ffmpeg_filters)

        output_file = "output_track.mp3"
        ffmpeg_command = f'ffmpeg -i {file_path} -af "{combined_filters}" {output_file}'

        print(f"Executing FFmpeg command: {ffmpeg_command}")
        subprocess.run(ffmpeg_command, shell=True, check=True)

        print(f"Modified audio saved to {output_file}")
        return output_file

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {str(e)}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing FFmpeg command: {e}")
        return None


file_path = "Post_Weekend.mp3"
user_prompt = "make the tempo very very fast and make it very soft"

response = analyze_audio_and_send_to_gemini(file_path, user_prompt)

if response:
    output_file = apply_suggested_modifications(file_path, response)
    if output_file:
        print(f"Final modified audio saved at: {output_file}")
    else:
        print("Failed to apply modifications.")
else:
    print("Failed to process or retrieve modifications from Gemini.")
