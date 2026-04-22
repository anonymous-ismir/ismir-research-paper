# BackgroundMellow 🎬🎧
**A Multi-Modal Cohesive Framework for Narrative-Driven Rich Cinematic Soundscape Generation**

Generating immersive, synchronized, and cinematic audio for long-form textual narratives remains a significant challenge in multi-modal AI. While current Text-to-Audio (TTA) frameworks successfully synthesize isolated sound effects, they struggle with narrative cohesion, temporal alignment, and cinematic emotional depth. 

**BackgroundMellow** is a framework that treats story-to-audio generation as a precise orchestration and signal processing problem. This framework is enabled without ground-truth through a **master-specialist agent architecture** that decomposes text into precise and multi-layered audio cues, generates each category of sounds with a suitable specialist model, and superimposes the soundscapes to create a unified and aligned audio segment. 

Our pipeline is built over the **Tango2** latent diffusion model for environmental synthesis alongside a novel **Cinematic BGM Retriever** mined from professional soundtracks. To automate the sound mixing process, we use an NLP-based module that predicts precise audio parameters—like start time, duration, and relative loudness—based on the narrative timeline. We further empirically evaluate and show the efficacy of the proposed framework leveraging nearest-neighbor retrieval against a curated dataset of YouTube cinematic trailers to measure temporal synchronization, coverage, and spectral richness.

---

## 🎥 Demo
Check out our generated cinematic soundscapes in action:
* **[Watch the Demo Video Here](./Demo/demo_1)** *(Note: Append `.mp4` or your specific video extension if necessary)*

## 🏗️ System Architecture
Below is the high-level architecture of the BackgroundMellow master-specialist framework:

![BackgroundMellow Architecture](./architecture.png)

---

## ✨ Key Features
* **Master-Specialist Agent Architecture:** Intelligently decomposes narrative text into discrete, layered audio cues (SFX, Ambience, Music, Narrator).
* **NLP-Driven Audio Mixing:** Automatically predicts and assigns start times, durations, and decibel (loudness) weights to ensure temporal alignment without human intervention.
* **Cinematic BGM Retriever:** Enhances emotional depth by retrieving and aligning professional background music tracks.
* **Tango2 Integration:** Leverages state-of-the-art latent diffusion for high-fidelity environmental and Foley sound synthesis.
* **Automated Superimposition:** Applies DSP techniques (like audio ducking and crossfading) to merge stems into a cohesive cinematic file.

---

## 🚀 Getting Started

### Prerequisites
* **Conda** (Miniconda or Anaconda)
* **Python 3.12**
* **Node.js & npm** (for the frontend interface)

### 1. Backend Setup (FastAPI / Uvicorn)

1. **Create and activate a new Conda environment:**
   ```bash
   conda create -n bgmellow python=3.12
   conda activate bgmellow
   ```

2.  **Navigate to the backend directory and install dependencies:**

    ```bash
    cd .backgroundMellow/backend
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file inside the `.backgroundMellow/backend` directory and add your API keys:

    ```env
    HUGGINGFACEHUB_ACCESS_TOKEN="your_huggingface_token_here"
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```

4.  **Start the Backend Server:**
    You can run the server directly via Uvicorn on port 8080:

    ```bash
    uvicorn main:app --port 8080 --reload
    ```

    *Alternatively, if configured in your package.json, you can run:*

    ```bash
    npm run dev:backend
    ```

### 2\. Frontend Setup (Vite / React)

1.  **Navigate to the main project folder** (`backgroundMellow`).

2.  **Configure Environment Variables:**
    Create a `.env` file in the root of the main frontend directory to point to your local backend API:

    ```env
    VITE_BACKEND_ENDPOINT="http://localhost:8080/api"
    ```

3.  **Install Node modules:**

    ```bash
    npm i
    ```

4.  **Start the Frontend Server:**

    ```bash
    npm run dev:frontend
    ```

    *The application should now be running locally. Check your terminal for the localhost port (usually `http://localhost:5173`).*

-----

## 📊 Dataset & Results

We empirically evaluated BackgroundMellow using a nearest-neighbor retrieval approach against a curated dataset of YouTube cinematic trailers, measuring temporal synchronization (Sync Score), coverage, and spectral richness.

  * **Local Files:** All evaluation metrics, model outputs, and raw data can be found in the local directory: `./Model Evaluation.xlxs`
  * **Cloud Dataset:** View the comprehensive dataset and results breakdown on Google Sheets:
    👉 **[View Google Sheets Dataset](https://docs.google.com/spreadsheets/d/1QyqbvlgpzJ0clwwLy8J5hq-WthJ6IBAN/edit?usp=sharing&ouid=117102775615004547445&rtpof=true&sd=true)**

-----

## 📄 Citation

If you use BackgroundMellow or our dataset in your research, please cite our paper:

```bibtex
@article{backgroundmellow2026,
  title={BackgroundMellow: A Multi-Modal Cohesive Framework for Narrative-Driven Rich Cinematic Soundscape Generation},
  author={},
  journal={TBD},
  year={2026}
}
```
