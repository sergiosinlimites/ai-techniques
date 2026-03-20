import pyttsx3
import sys

engine = pyttsx3.init()
engine.say(sys.argv[1])
engine.runAndWait()