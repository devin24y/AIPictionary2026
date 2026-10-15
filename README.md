AIPictionary

Turn your hands into a controller. AIPictionary is a computer-vision Pictionary game that turns hand gestures into real-time drawing. No mouse, no keyboard: just your hands, your creativity, and an AI that tries to guess what you drew.

🏆 Winner, Most Engaging Demo, hackUMBC 2026

[Devpost](https://devpost.com/software/betterwhiteboard) | [Demo Video](https://youtu.be/34cLrNeRg5c?si=Wb40LEpWmC4cvwqa)

Overview

We wanted to make Pictionary more immersive by removing the physical drawing interface. A webcam watches your hand, and you draw, erase, pause, and clear the canvas using gestures alone. When you finish, an AI model looks at your drawing and guesses what it is.

How It Works
Webcam → Hand Tracking → Gesture Recognition → Digital Canvas → AI Guess
Hand tracking: MediaPipe detects hand landmarks from the webcam feed.
Gesture recognition: The landmark positions are used to recognize gestures and track the player's fingertip. Detection thresholds were tuned to keep drawing responsive while preventing accidental erasing or pausing.
Digital canvas: OpenCV renders the fingertip movement onto a virtual canvas in real time.
AI guess: The finished drawing is sent to an AI model, which analyzes the image and tries to guess what was drawn.
Gestures
Action	Description
Draw	Index Finger Extended
Pause	Index and Middle Finger Extended
Clear	Open palm

Tech Stack
Python
MediaPipe for hand landmark detection
OpenCV for webcam capture and canvas rendering
AI vision model (via API) for guessing the drawing
