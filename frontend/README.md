# Frontend

| File | Content |
|---|---|
| `index.html` | Page structure and logic (JavaScript): camera, microphone, API calls |
| `style.css` | All the styles. Theme colors are in the `:root` variables at the top of the file |

They must be **in the same folder**: `index.html` loads `style.css` with a relative path.

The API serves them at `http://localhost:8000/ui/` (folder mounted in `app/main.py`).

What the interface does:
- Shows the camera and sends frames to `POST /emotion` (multipart, field `image`).
- Records audio, converts it to 16 kHz WAV and sends it to `POST /voice/command` (field `audio`).
- Sends typed commands to `POST /command` (`{"text": "..."}`).
- Lists the models from `GET /models` and runs any of them with `POST /predict/{model}`.
