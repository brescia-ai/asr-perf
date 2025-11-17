import requests
import numpy as np
import io
from scipy.io import wavfile

ASR_SERVER_ENDPOINT = "http://fulmine:8001/transcribe"

#
##
###
##
#

def inferenceFunction(audio_input: dict[str, np.ndarray | int]) -> str:
    """
    Sends a numpy array containing audio data to the ASR server and returns the transcription.

    The numpy array is converted to a WAV file in memory and sent as a file upload.
    The server is expected to return a JSON response with a 'chunks' key, which is a list of transcribed segments.

    Args:
        audio_input: A dictionary containing the audio waveform as a numpy array and the sampling rate.
                     e.g., {'array': np.array([...]), 'sampling_rate': 16000}

    Returns:
        The transcribed text from the ASR server.
    """

    waveform = audio_input["array"]
    sampling_rate = audio_input["sampling_rate"]

    try:
        # Create an in-memory bytes buffer
        buffer = io.BytesIO()

        # The server expects 16kHz mono. Let's assume the input is already mono.
        # The data needs to be in a suitable format for WAV, e.g., int16
        if waveform.dtype in [np.float32, np.float64]:
            waveform = (waveform * 32767).astype(np.int16)

        wavfile.write(buffer, sampling_rate, waveform)
        buffer.seek(0)

        # Send the POST request with the file
        files = {"file": ("audio.wav", buffer, "audio/wav")}
        response = requests.post(ASR_SERVER_ENDPOINT, files=files)

        # Raise an exception for HTTP errors (e.g., 404, 500)
        response.raise_for_status()

        # Parse the JSON response and get the transcription from chunks
        chunks = response.json().get("chunks", [])
        transcription = " ".join(chunk.get("text", "") for chunk in chunks)

        return transcription.strip()

    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        print(f"An error occurred while communicating with the ASR server: {e}")
        return ""
    except Exception as e:
        # Handle other potential errors (e.g., JSON decoding, file writing)
        print(f"An unexpected error occurred: {e}")
        return ""
