import speech_recognition as sr

Recognition = sr.Recognizer()

with sr.Microphone() as source:
    print("Please speak anything")
    Listener = Recognition.listen(source)

    try:
        Spoken_Text = Recognition.recognize_google(Listener)
        print("The user spoke: {}".format(Spoken_Text))
    except:
        print("Sorry could not recognize the test")
