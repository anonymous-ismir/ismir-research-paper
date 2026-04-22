import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import HeroSection from "../components/HeroSection";
import AudioCard from "../components/AudioCard";
import MasterMixButton from "../components/MasterMixButton";
import FloatingParticles from "../components/FloatingParticles";
import AnimatedLoader from "../components/ui/AnimatedLoader";
import FinalAudioPlayer from "../components/FinalAudioPlayer";
import EvaluationForm from "../components/EvaluationForm.jsx";
import CustomMusicUpload from "../components/CustomMusicUpload.jsx";
import { Button } from "../components/ui/button";


const Index = () => {
  const [storyText, setStoryText] = useState("");
  const [audioCues, setAudioCues] = useState([]);
  const [narratorCues, setNarratorCues] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [finalAudio, setFinalAudio] = useState(null);
  const [missingCues, setMissingCues] = useState([]);
  const [enableNarrator, setEnableNarrator] = useState(true);
  
  const handleDecompose = async (storyText) => {
    // Restore the audio cues
    setAudioCues([]);
    setNarratorCues([]);
    setFinalAudio(null);

    // Use relative path to leverage Vite proxy in development
    const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
    setIsLoading(true);

    try {
      // 1) Decide cues
      const response = await fetch(`${apiBase}/v1/decide-cues`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          story_text: storyText,
          speed_wps: 2
        })
      });

      if (!response.ok) {
        throw new Error("Failed to fetch audio cues");
      }

      const data = await response.json();
      const cues = data?.cues || [];

      // Separate AudioCue and NarratorCue
      const audioCueList = [];
      const narratorCueList = [];

      cues.forEach((cue) => {
        const isNarrator = cue.audio_type === "NARRATOR" || cue.story || cue.narrator_description;

        if (isNarrator) {
          narratorCueList.push({
            id: cue.id,
            story: cue.story || "",
            narrator_description: cue.narrator_description || "",
            audio_type: "NARRATOR",
            start_time_ms: cue.start_time_ms || 0,
            duration_ms: cue.duration_ms || 10,
            audioBase64: null,
            evaluation: {
              promptAdherence: 0,
              acousticNaturalness: 0,
              recognitionRate: 0
            }
          });
        } else {
          audioCueList.push({
            id: cue.id,
            audio_class: cue.audio_class || "",
            audio_type: cue.audio_type || "SFX",
            start_time_ms: cue.start_time_ms || 0,
            duration_ms: cue.duration_ms || 10,
            weight_db: cue.weight_db || 0,
            fade_ms: cue.fade_ms || 500,
            audioBase64: null,
            evaluation: {
              promptAdherence: 0,
              acousticNaturalness: 0,
              recognitionRate: 0
            }
          });
        }
      });

      console.log("audioCueList:", audioCueList);
      console.log("narratorCueList:", narratorCueList);

      // Optionally disable narrator cues based on toggle
      const effectiveNarratorCues = enableNarrator ? narratorCueList : [];

      // Set initial cues immediately so UI can render placeholders
      setAudioCues(audioCueList);
      setNarratorCues(effectiveNarratorCues);

      const totalDurationMs = data?.total_duration_ms ?? 1000;

      // Prepare payloads for generation
      const allCues = [...audioCueList, ...effectiveNarratorCues].map((cue) => {
        const isNarrator = cue.audio_type === "NARRATOR";
        return isNarrator
          ? {
              id: cue.id,
              audio_type: cue.audio_type,
              start_time_ms: cue.start_time_ms,
              duration_ms: cue.duration_ms,
              story: cue.story,
              narrator_description: cue.narrator_description,
            }
          : {
              id: cue.id,
              audio_class: cue.audio_class,
              audio_type: cue.audio_type,
              start_time_ms: cue.start_time_ms,
              duration_ms: cue.duration_ms,
              weight_db: cue.weight_db,
              fade_ms: cue.fade_ms,
            };
      });

      // 2) Generate audio sequentially in small batches (pairs).
      // As each batch resolves, we immediately update the corresponding cues,
      // so audio appears progressively instead of waiting for all cues.
      const BATCH_SIZE = 10;

      for (let i = 0; i < allCues.length; i += BATCH_SIZE) {
        const batch = allCues.slice(i, i + BATCH_SIZE);

        try {
          const res = await fetch(`${apiBase}/v1/generate-audio`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              cues: batch,
              total_duration_ms: totalDurationMs,
            }),
          });

          if (!res.ok) {
            throw new Error("Failed to generate audio");
          }

          const audioData = await res.json();

          if (audioData.audio_cues && audioData.audio_cues.length > 0) {
            audioData.audio_cues.forEach((audioCueData) => {
              const cueId = audioCueData.audio_cue?.id;
              const audioBase64 = audioCueData.audio_base64;
              const durationMs = audioCueData.duration_ms;
              const cueType = audioCueData.audio_cue?.audio_type;

              if (audioBase64 && audioBase64 !== null && audioBase64 !== "") {
                if (cueType === "NARRATOR") {
                  setNarratorCues((prevCues) =>
                    prevCues.map((prevCue) =>
                      prevCue.id === cueId
                        ? {
                            ...prevCue,
                            audioBase64,
                            duration_ms: durationMs || prevCue.duration_ms,
                          }
                        : prevCue
                    )
                  );
                } else {
                  setAudioCues((prevCues) =>
                    prevCues.map((prevCue) =>
                      prevCue.id === cueId
                        ? {
                            ...prevCue,
                            audioBase64,
                            duration_ms: durationMs || prevCue.duration_ms,
                          }
                        : prevCue
                    )
                  );
                }
              }
            });
          }
        } catch (batchErr) {
          console.error("Error generating audio for cues batch:", batchErr);
        }
      }
    } catch (err) {
      console.error("Error fetching audio cues:", err);
      setAudioCues([]);
      setNarratorCues([]);
    } finally {
      // Deciding cues is done; per-card loaders will reflect ongoing audio generation
      setIsLoading(false);
    }
  };

  const handleUpdate = (cueId, updates) => {
    setAudioCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId ? { ...cue, ...updates } : cue
      )
    );
  };

  const handleNarratorUpdate = (cueId, updates) => {
    setNarratorCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId ? { ...cue, ...updates } : cue
      )
    );
   
  };

  const handleEvaluationUpdate = (cueId, evaluationUpdates) => {
    setAudioCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId 
          ? { 
              ...cue, 
              evaluation: { ...cue.evaluation, ...evaluationUpdates }
            } 
          : cue
      )
    );
  };

  const handleNarratorEvaluationUpdate = (cueId, evaluationUpdates) => {
    setNarratorCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId 
          ? { 
              ...cue, 
              evaluation: { ...cue.evaluation, ...evaluationUpdates }
            } 
          : cue
      )
    );
  };

  const handleMasterMix = () => {
    setFinalAudio(null);
    setMissingCues([]);
    setIsLoading(true);
    setShowEvaluation(true);

    const readyAudioCues = audioCues.filter(cue =>
      cue.audioBase64 !== null &&
      cue.audioBase64 !== undefined &&
      cue.audioBase64 !== ""
    );
    const readyNarratorCues = narratorCues.filter(cue =>
      cue.audioBase64 !== null &&
      cue.audioBase64 !== undefined &&
      cue.audioBase64 !== ""
    );

    const audioCuePayload = readyAudioCues.map((cue) => ({
      audio_cue: {
        id: cue.id,
        audio_class: cue.audio_class,
        audio_type: cue.audio_type,
        start_time_ms: cue.start_time_ms,
        duration_ms: cue.duration_ms,
        weight_db: cue.weight_db,
        fade_ms: cue.fade_ms
      },
      audio_base64: cue.audioBase64,
      duration_ms: cue.duration_ms
    }));
    const narratorCuePayload = readyNarratorCues.map((cue) => ({
      audio_cue: {
        id: cue.id,
        story: cue.story,
        narrator_description: cue.narrator_description,
        audio_type: cue.audio_type,
        start_time_ms: cue.start_time_ms,
        duration_ms: cue.duration_ms,
        weight_db: cue.weight_db || 0
      },
      audio_base64: cue.audioBase64,
      duration_ms: cue.duration_ms
    }));
    const cues = [...audioCuePayload, ...narratorCuePayload];
    const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
    const payload = { cues, story_text: storyText, speed_wps: 2 };

    // Phase 1: check missing cues so we can show them in loading state
    fetch(`${apiBase}/v1/check-missing-audio-cues`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then((res) => (res.ok ? res.json() : Promise.resolve({ missing_cues: [] })))
      .then((data) => {
        setMissingCues(data.missing_cues || []);
      })
      .catch(() => setMissingCues([]));

    // Phase 2: generate final audio
    fetch(`${apiBase}/v1/generate-audio-cues-with-audio-base64`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(async (response) => {
        if (!response.ok) throw new Error("Failed to generate audio");
        return response.json();
      })
      .then((data) => {
        const totalDurationMs = Math.max(
          ...cues.map(cue => cue.audio_cue.start_time_ms + cue.audio_cue.duration_ms),
          0
        );
        setFinalAudio({
          audioBase64: data.audio_base64 || null,
          duration: totalDurationMs / 1000
        });
        setMissingCues([]);
      })
      .catch((err) => {
        console.error("Error generating audio:", err);
        setMissingCues([]);
      })
      .finally(() => setIsLoading(false));

    return true;
  };

  const handleCustomMusicSave = async (personName, description, file) => {
    console.log("handleCustomMusicSave");
    if (!file) {
      console.error("No audio file provided for custom music upload.");
      return;
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onloadend = async () => {
        try {
          const base64Audio = reader.result;
          const payload = {
            requestType: "CUSTOM", // Required for routing in Google Apps Script
            personName: personName || "Anonymous",
            description: description || "N/A",
            audioFile: base64Audio,
          };

          console.log("Custom music payload :", payload);

          await fetch("https://script.google.com/macros/s/AKfycbwaCqI2T56bBqOoLOrxO_zp6Yw7hiHae1BLoqRoF7HeHnVfPxPeTXR4HkzPWL5vKzXJ/exec", {
            method: "POST",
            mode: "no-cors", // Required for cross-origin GAS requests
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          resolve(true);
        } catch (error) {
          console.error("Error saving custom music:", error);
          reject(error);
        }
      };

      reader.onerror = (err) => {
        console.error("FileReader error while reading custom music:", err);
        reject(err);
      };

      reader.readAsDataURL(file);
    });
  };

  return (
    <div className="min-h-screen bg-background relative">
      <FloatingParticles count={40} />

      <div className="relative z-10">
        {/* Header */}
        <motion.header
          className="border-b border-border/30 backdrop-blur-sm bg-background/50 sticky top-0 z-50"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
        >
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <motion.div
                className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center"
                animate={{
                  boxShadow: [
                    "0 0 10px hsl(var(--primary) / 0.5)",
                    "0 0 20px hsl(var(--primary) / 0.8)",
                    "0 0 10px hsl(var(--primary) / 0.5)",
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <span className="text-primary-foreground font-display text-sm font-bold">BGM</span>
              </motion.div>
              <span className="font-display text-lg">
                <span className="text-foreground">Back</span>
                <span className="text-primary">Ground</span>
                <span className="text-foreground">Mellow</span>
              </span>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground font-mono">v1.0.0-beta</span>
              <div className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
            </div>
          </div>
        </motion.header>

        {/* Hero Section */}
        <HeroSection
          isLoading={isLoading}
          onDecompose={handleDecompose}
          storyText={storyText}
          setStoryText={setStoryText}
          enableNarrator={enableNarrator}
          setEnableNarrator={setEnableNarrator}
        />

        

        {/* Narrator Cues Section */}
        <AnimatePresence>
          {narratorCues.length > 0 && (
            <motion.section
              className="container mx-auto px-4 py-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center gap-2 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-orange-500 to-orange-400 rounded-full" />
                <h2 className="font-display text-xl tracking-wider text-foreground">
                  NARRATOR CUES
                </h2>
                <span className="ml-2 px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-mono">
                  {narratorCues.length}
                </span>
              </div>

              <div className="space-y-4">
                {narratorCues
                  .map((cue, index) => (
                    <motion.div
                      key={cue.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.15 }}
                    >
                      <AudioCard
                        id={cue.id}
                        type="NARRATOR"
                        prompt={cue.story || "Narrator audio"}
                        narrator_description={cue.narrator_description || "Narrator audio"}
                        start_time_ms={cue.start_time_ms}
                        duration_ms={cue.duration_ms}
                        weight_db={0}
                        fade_ms={500}
                        audio_base64={cue.audioBase64}
                        handleUpdate={handleNarratorUpdate}
                        evaluation={cue.evaluation || { promptAdherence: 0, acousticNaturalness: 0, recognitionRate: 0 }}
                        onEvaluationUpdate={handleNarratorEvaluationUpdate}
                      />
                    </motion.div>
                  ))}
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Audio Cues Section */}
        <AnimatePresence>
          {audioCues.length > 0 && (
            <motion.section
              className="container mx-auto px-4 py-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center gap-2 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-primary to-secondary rounded-full" />
                <h2 className="font-display text-xl tracking-wider text-foreground">
                  AUDIO CUES
                </h2>
                <span className="ml-2 px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-mono">
                  {audioCues.length}
                </span>
              </div>

              <div className="space-y-4">
                {audioCues
                  .map((cue, index) => (
                    <motion.div
                      key={cue.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.15 }}
                    >
                      <AudioCard
                        id={cue.id}
                        type={cue.audio_type}
                        prompt={cue.audio_class}
                        start_time_ms={cue.start_time_ms}
                        duration_ms={cue.duration_ms}
                        weight_db={cue.weight_db}
                        fade_ms={cue.fade_ms}
                        audio_base64={cue.audioBase64}
                        handleUpdate={handleUpdate}
                        evaluation={cue.evaluation || { promptAdherence: 0, acousticNaturalness: 0, recognitionRate: 0 }}
                        onEvaluationUpdate={handleEvaluationUpdate}
                      />
                    </motion.div>
                  ))}
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Master Mix Button */}
        <AnimatePresence>
          {(audioCues.length > 0 || narratorCues.length > 0) && (
            <motion.section
              className="container mx-auto px-4 pb-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ delay: 0.5 }}
            >
              <div className="flex justify-center">
                <MasterMixButton
                  isLoading={isLoading}
                  onMix={(storyText) => handleMasterMix(storyText)}
                  disabled={audioCues.length === 0 && narratorCues.length === 0}
                />
              </div>
            </motion.section>
          )}
        </AnimatePresence>


        {/* Missing cues loading state: show which cues are being generated */}
        <AnimatePresence>
          {isLoading && missingCues.length > 0 && (
            <motion.section
              className="container mx-auto px-4 pb-6"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="glass-panel p-6 rounded-xl border border-amber-500/30 bg-amber-950/20 backdrop-blur-md">
                <h3 className="text-sm font-display font-semibold text-amber-200 mb-3 flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                  Generating audio for missing cues
                </h3>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  {missingCues.map((cue, i) => (
                    <li key={cue.id ?? i} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                      {cue.audio_class ?? cue.story?.slice(0, 40) ?? `Cue ${cue.id ?? i + 1}`}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Final Audio & Evaluation */}
        <AnimatePresence>
          {finalAudio && finalAudio.audioBase64 && (
            <motion.section
              className="container mx-auto px-4 pb-12 space-y-6"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <FinalAudioPlayer
                audioBase64={finalAudio.audioBase64}
                duration={finalAudio.duration}
              />
              
              {/* Evaluation Instructions */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-panel p-6 rounded-xl border border-border/30 bg-slate-900/50 backdrop-blur-md"
              >
                <div className="mb-6">
                  <h3 className="text-lg font-display font-bold text-foreground mb-2">
                    Evaluation Guidelines
                  </h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Please rate the generated audio on the following 5 parameters using a Likert Scale (1-5):
                  </p>
                </div>

                <div className="space-y-4">
                  {/* Parameter 1 */}
                  <div className="p-4 rounded-lg bg-slate-800/30 border border-emerald-500/20">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-emerald-500 mt-2 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-foreground mb-1">
                          1. Temporal Synchronization (Sync Accuracy)
                        </h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          <strong>Definition:</strong> The precision of sound placement relative to the narrator's voice.
                        </p>
                        <p className="text-xs text-slate-400 italic">
                          <strong>How to Judge:</strong> "Watch the timeline closely. If the story mentions 'rain started' and the audio begins exactly at that word, give it a 5. If the sound is delayed or triggers too early, distracting from the narrative, give it a 1 or 2."
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Parameter 2 */}
                  <div className="p-4 rounded-lg bg-slate-800/30 border border-blue-500/20">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-foreground mb-1">
                          2. Semantic Alignment (Semantic Fit)
                        </h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          <strong>Definition:</strong> The accuracy of the AI's choice of sound based on the story's meaning.
                        </p>
                        <p className="text-xs text-slate-400 italic">
                          <strong>How to Judge:</strong> "Does the specialist model generate the correct vibe? If the text describes a 'creaky old door' and you hear a modern sliding door, the score should be lower. We are looking for how well the LLM-Decider understood the descriptive adjectives."
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Parameter 3 */}
                  <div className="p-4 rounded-lg bg-slate-800/30 border border-yellow-500/20">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-yellow-500 mt-2 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-foreground mb-1">
                          3. Audio Fidelity & Richness (Acoustic Quality)
                        </h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          <strong>Definition:</strong> The technical clarity of the generated clips.
                        </p>
                        <p className="text-xs text-slate-400 italic">
                          <strong>How to Judge:</strong> "Listen for 'crustiness,' static, or robotic chirps (AI artifacts). A 5 represents studio-quality sound. A 1 represents audio that sounds heavily compressed, muffled, or filled with unpleasant digital noise."
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Parameter 4 */}
                  <div className="p-4 rounded-lg bg-slate-800/30 border border-red-500/20">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-red-500 mt-2 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-foreground mb-1">
                          4. Atmospheric Cohesion (Narrative Flow)
                        </h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          <strong>Definition:</strong> How well the SFX, Ambience, and Music sit together under the Narrator.
                        </p>
                        <p className="text-xs text-slate-400 italic">
                          <strong>How to Judge:</strong> "Does the background music drown out the narrator? (Poor Ducking). Do the SFX feel like they are in the same 'room' as the ambience? If the layers feel like they are fighting each other rather than blending into a scene, score it lower."
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Parameter 5 */}
                  <div className="p-4 rounded-lg bg-slate-800/30 border border-purple-500/20">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-purple-500 mt-2 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-foreground mb-1">
                          5. Cinematic Impact
                        </h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          <strong>Definition:</strong> How well the audio enhances the story's emotional engagement and immersion.
                        </p>
                        <p className="text-xs text-slate-400 italic">
                          <strong>How to Judge:</strong> "Does the audio actually make the story more engaging/immersive? Does it add dramatic value and enhance the narrative experience? Rate based on overall impact on storytelling."
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>

              <EvaluationForm
                audioBase64={finalAudio.audioBase64}
                storyText={storyText}
              />
            </motion.section>
          )}
        </AnimatePresence>

        {/* Custom Music Upload */}
        <section className="container mx-auto px-4 pb-12">
          <CustomMusicUpload onSave={handleCustomMusicSave} />
        </section>

        {/* Footer */}
        <footer className="border-t border-border/30 py-8 mt-12">
          <div className="container mx-auto px-4 text-center">
            <p className="text-xs text-muted-foreground">
              <span className="font-display tracking-wider">BackGroundMellow ENGINE</span>
              {" · "}
              <span className="font-mono">Powered by AI</span>
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;
