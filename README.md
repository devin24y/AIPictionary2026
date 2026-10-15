# AIPictionary

**Turn your hands into a controller.** AIPictionary is a computer-vision Pictionary game that turns hand gestures into real-time drawing. No mouse, no keyboard: just your hands, your creativity, and an AI that tries to guess what you drew.

🏆 **Winner, Most Engaging Demo**, hackUMBC 2026

[Devpost](https://devpost.com/software/betterwhiteboard) | [Demo Video](https://www.youtube.com/watch?v=34cLrNeRg5c)

---

## Overview

We wanted to make Pictionary more immersive by removing the physical drawing interface. A webcam watches your hand, and you draw, pause, and clear the canvas using gestures alone. When you finish, an AI model looks at your drawing and guesses what it is.

## How It Works

```
Webcam → Hand Tracking → Gesture Recognition → Digital Canvas → AI Guess
```

1. **Hand tracking:** MediaPipe detects hand landmarks from the webcam feed.
2. **Gesture recognition:** The landmark positions are used to recognize gestures and track the player's fingertip. Detection thresholds were tuned to keep drawing responsive while preventing accidental erasing or pausing.
3. **Digital canvas:** OpenCV renders the fingertip movement onto a virtual canvas in real time.
4. **AI guess:** The finished drawing is sent to an AI model, which analyzes the image and tries to guess what was drawn.

## Gestures

| Action | Gesture |
|--------|---------|
| Draw   | Index finger extended |
| Pause  | Index and middle finger extended |
| Clear  | Open palm |

## Tech Stack

- **Python**
- **MediaPipe** for hand landmark detection
- **OpenCV** for webcam capture and canvas rendering
- **AI vision model (via API)** for guessing the drawing
