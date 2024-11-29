import cv2
from inference_sdk import InferenceHTTPClient
import threading
import pyttsx3
import openai


# Initializing clients
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=""  # Replace with your actual API key
)
tts_engine = pyttsx3.init()
openai.api_key = ""  # Replace with your actual OpenAI API key


# we won’t input our actual API keys due to privacy reasons


words = []
letter = "0"


# Try to open camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


#S Sends to roboflow’s model to predict the data
def perform_inference(frame):
    global letter
    result = CLIENT.infer(frame, model_id="american-sign-language-letters/6")
    if 'predictions' in result:
        for prediction in result['predictions']:
            label = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            print(f"Predicted Letter: {label} with confidence {confidence:.2f}")
            if label != letter:
                words.append(label)
                letter = label


# Try to predict what the user is trying to sign
def predict_phrase(letters):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ''.join(letters)}
        ],
        max_tokens=10
    )
    return response.choices[0].message['content'].strip()




# for the video frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break


        frame = cv2.resize(frame, (640, 480)) 


        thread = threading.Thread(target=perform_inference, args=(frame,))
        thread.daemon = True
        thread.start()
        cv2.imshow('Sign Language Detection', frame)


# exiting program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            predicted_phrase = predict_phrase(words)
            print(f"Predicted Phrase: {predicted_phrase}")
            tts_engine.say(predicted_phrase)
            tts_engine.runAndWait()
            break


# close program
finally:
    cap.release()
    cv2.destroyAllWindows()

