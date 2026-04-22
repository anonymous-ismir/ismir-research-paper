import { motion } from "framer-motion";
import { useState, useEffect, useRef, useCallback, memo } from "react";
import { Play, Pause, RotateCcw, Volume2, Loader2, ChevronDown, ChevronUp, Star, Save, CheckCircle2 } from "lucide-react";
import Waveform from "./timeline/Waveform";
import TimelineRuler from "./timeline/TimelineRuler";
import DraggableMarker from "./timeline/DraggableMarker";
import VerticalFader from "./mixer/VerticalFader";
import SpatialPanPad from "./mixer/SpatialPanPad";
import CurveSelector from "./mixer/CurveSelector";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

const AudioCard = memo(({
  id,
  type = "SFX",
  prompt = "Thunderstorm with distant rumbling",
  narrator_description = "",
  start_time_ms = 0,
  duration_ms = 10000,
  weight_db = 0,
  fade_ms = 500,
  audioBase64 = null,
  audio_base64 = null,
  isRegenerating = false,
  handleUpdate = () => { },
  onRegenerate = () => { },
  evaluation = { promptAdherence: 0, acousticNaturalness: 0, recognitionRate: 0 },
  onEvaluationUpdate = () => { },
}) => {
  const audioRef = useRef(null);
  const intervalRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [fadeIn, setFadeIn] = useState(0.1);
  const [fadeOut, setFadeOut] = useState(0.9);
  const [volume, setVolume] = useState(weight_db || 0);
  const [pan, setPan] = useState({ x: 0.5, y: 0.5 });
  const [fadeInCurve, setFadeInCurve] = useState("logarithmic");
  const [fadeOutCurve, setFadeOutCurve] = useState("exponential");
  const [editablePrompt, setEditablePrompt] = useState(prompt);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [evaluationScores, setEvaluationScores] = useState({
    promptAdherence: evaluation?.promptAdherence || 0,
    acousticNaturalness: evaluation?.acousticNaturalness || 0,
    recognitionRate: evaluation?.recognitionRate || 0,
  });
  const [isSavingEvaluation, setIsSavingEvaluation] = useState(false);
  const [saveEvaluationStatus, setSaveEvaluationStatus] = useState(null); // null, 'saving', 'success', 'error'
  const [evaluatorName, setEvaluatorName] = useState("");
  const [cueFeedback, setCueFeedback] = useState("");
  const [isRegeneratingLocal, setIsRegeneratingLocal] = useState(false);


  // Get audio data from either prop name (treat empty string as "no audio yet")
  const audioData = audioBase64 ?? audio_base64;
  const isRegeneratingEffective = isRegenerating || isRegeneratingLocal;

  // Convert duration_ms to seconds for display
  const durationSeconds = duration_ms / 1000;

  // Sync state with props when they change
  useEffect(() => {
    setEditablePrompt(prompt);

    //call handleUpdate to update the audio cue
    handleUpdate(id, { audio_class: editablePrompt });
    setVolume(weight_db || 0);
  }, [prompt, weight_db]);

  // Sync evaluation scores with props
  useEffect(() => {
    if (evaluation) {
      setEvaluationScores({
        promptAdherence: evaluation.promptAdherence || 0,
        acousticNaturalness: evaluation.acousticNaturalness || 0,
        recognitionRate: evaluation.recognitionRate || 0,
      });
    }
  }, [evaluation]);

  // Handle evaluation score change
  const handleEvaluationScoreChange = useCallback((key, value) => {
    const newScores = { ...evaluationScores, [key]: value };
    setEvaluationScores(newScores);
    onEvaluationUpdate(id, newScores);
  }, [id, evaluationScores, onEvaluationUpdate]);

  // Handle save specialist model evaluation
  const handleSaveEvaluation = useCallback(async () => {
    // Check if all evaluation scores are filled
    if (Object.values(evaluationScores).some(score => score === 0)) {
      alert("Please evaluate all three parameters before saving.");
      return;
    }

    if (!audioData) {
      alert("No audio data available to save.");
      return;
    }

    setIsSavingEvaluation(true);
    setSaveEvaluationStatus('saving');

    try {
      // Convert base64 string to Blob
      const base64Data = audioData.includes(',') 
        ? audioData.split(',')[1] 
        : audioData;
      
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: 'audio/wav' });

      // Convert Blob to Base64 for Google Apps Script
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      
      reader.onloadend = async () => {
        try {
          const base64Audio = reader.result;
          const payload = {
            requestType: "SPECIALIST", // Required for routing in Google Apps Script
            personName: evaluatorName || "Anonymous",
            prompt: editablePrompt,
            cueType: type, // SFX, AMBIENCE, MUSIC, or NARRATOR or MOVIE_BGM
            scores: {
              adherence: evaluationScores.promptAdherence,
              naturalness: evaluationScores.acousticNaturalness,
              recognition: evaluationScores.recognitionRate,
            },
            cueFeedback: cueFeedback || "",
            audioFile: base64Audio,
          };

          console.log("Specialist Model Evaluation Payload:", payload);
    
          // Send to Google Apps Script
          await fetch("https://script.google.com/macros/s/AKfycbwaCqI2T56bBqOoLOrxO_zp6Yw7hiHae1BLoqRoF7HeHnVfPxPeTXR4HkzPWL5vKzXJ/exec", {
            method: "POST",
            mode: "no-cors", // Required for cross-origin GAS requests
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
    
          setSaveEvaluationStatus('success');
          setTimeout(() => {
            alert("Specialist model evaluation saved successfully!");
            setSaveEvaluationStatus(null);
          }, 500);
        } catch (error) {
          console.error("Save failed:", error);
          setSaveEvaluationStatus('error');
          setTimeout(() => {
            alert("Error saving specialist model evaluation.");
            setSaveEvaluationStatus(null);
          }, 500);
        } finally {
          setIsSavingEvaluation(false);
        }
      };
      
      reader.onerror = () => {
        console.error("FileReader error");
        setSaveEvaluationStatus('error');
        setIsSavingEvaluation(false);
        setTimeout(() => {
          alert("Error processing audio file.");
          setSaveEvaluationStatus(null);
        }, 500);
      };
    } catch (error) {
      console.error("Save failed:", error);
      setSaveEvaluationStatus('error');
      setIsSavingEvaluation(false);
      setTimeout(() => {
        alert("Error saving data.");
        setSaveEvaluationStatus(null);
      }, 500);
    }
  }, [audioData, evaluationScores, id, type, editablePrompt, narrator_description, start_time_ms, duration_ms, volume, fade_ms, evaluatorName, cueFeedback]);

  // Initialize audio element when audio data is available
  useEffect(() => {
    // Clean up previous audio if it exists
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    if (audioData) {
      const audioUrl = audioData.startsWith('data:')
        ? audioData
        : `data:audio/wav;base64,${audioData}`;

      audioRef.current = new Audio(audioUrl);
      audioRef.current.volume = Math.max(0, Math.min(1, (volume + 12) / 12)); // Convert dB to 0-1 range

      audioRef.current.onended = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };

      audioRef.current.ontimeupdate = () => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime);
        }
      };
    } else {
      setIsPlaying(false);
      setCurrentTime(0);
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [audioData]); // Only recreate when audioData changes

  // Update audio volume when volume state changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = Math.max(0, Math.min(1, (volume + 12) / 12));
    }
  }, [volume]);

  // Handle play/pause
  const handlePlayPause = useCallback(() => {
    if (audioRef.current && audioData) {
      if (isPlaying) {
        audioRef.current.pause();
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      } else {
        audioRef.current.play().catch(err => {
          console.error("Error playing audio:", err);
          setIsPlaying(false);
        });
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying, audioData]);

  // Handle volume change with update callback
  const handleVolumeChange = useCallback((newVolume) => {
    setVolume(newVolume);
    handleUpdate(id, { weight_db: newVolume });
  }, [id, handleUpdate]);

  // Handle fade in change with update callback
  const handleFadeInChange = useCallback((newFadeIn) => {
    setFadeIn(newFadeIn);
    const fadeInMs = Math.round(newFadeIn * duration_ms);
    handleUpdate(id, { fade_in_ms: fadeInMs });
  }, [id, duration_ms, handleUpdate]);

  // Handle fade out change with update callback
  const handleFadeOutChange = useCallback((newFadeOut) => {
    setFadeOut(newFadeOut);
    const fadeOutMs = Math.round((1 - newFadeOut) * duration_ms);
    handleUpdate(id, { fade_out_ms: fadeOutMs });
  }, [id, duration_ms, handleUpdate]);

  // Handle prompt change with update callback
  const handlePromptChange = useCallback((newPrompt) => {
    setEditablePrompt(newPrompt);
    handleUpdate(id, { audio_class: newPrompt });
  }, [id, handleUpdate]);

  // Handle pan change with update callback
  const handlePanChange = useCallback((newPan) => {
    setPan(newPan);
    handleUpdate(id, { pan_x: newPan.x, pan_y: newPan.y });
  }, [id, handleUpdate]);

  // Handle narrator description change
  const handleNarratorDescriptionChange = useCallback((newDescription) => {
    handleUpdate(id, { narrator_description: newDescription });
  }, [id, handleUpdate]);

  // Handle regenerate
  const handleRegenerate = async () => {
    if (isRegeneratingEffective) return;

    setIsRegeneratingLocal(true);

    const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
    const isNarrator = type === "NARRATOR";

    const cuePayload = isNarrator
      ? {
          id,
          audio_type: "NARRATOR",
          start_time_ms,
          duration_ms,
          story: editablePrompt,
          narrator_description: narrator_description || "",
        }
      : {
          id,
          audio_class: editablePrompt,
          audio_type: type || "SFX",
          start_time_ms,
          duration_ms,
          weight_db: volume,
          fade_ms,
        };

    try {
      const res = await fetch(`${apiBase}/v1/generate-audio`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          cues: [cuePayload],
          total_duration_ms: duration_ms,
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to regenerate audio");
      }

      const data = await res.json();
      const regenerated = data?.audio_cues?.[0];
      const newBase64 = regenerated?.audio_base64;
      const newDurationMs = regenerated?.duration_ms;

      if (newBase64) {
        // Replace previous audio with regenerated audio
        handleUpdate(id, {
          audioBase64: newBase64,
          duration_ms: newDurationMs || duration_ms,
        });
      }
    } catch (err) {
      console.error("Error regenerating audio:", err);
    } finally {
      setIsRegeneratingLocal(false);
    }
  };

  const typeColors = {
    SFX: "bg-primary/20 text-primary border-primary/50",
    AMBIENCE: "bg-secondary/20 text-secondary border-secondary/50",
    Ambience: "bg-secondary/20 text-secondary border-secondary/50",
    MUSIC: "bg-neon-purple/20 text-neon-purple border-neon-purple/50",
    Music: "bg-neon-purple/20 text-neon-purple border-neon-purple/50",
    NARRATOR: "bg-orange-500/20 text-orange-400 border-orange-500/50",
    Dialogue: "bg-orange-500/20 text-orange-400 border-orange-500/50",
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <motion.div
      className="glass-panel p-4 gradient-border"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={{ scale: 1.01 }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Badge
          variant="outline"
          className={`font-display text-[10px] tracking-wider ${typeColors[type] || typeColors.SFX}`}
        >
          {type}
        </Badge>
        <input
          type="text"
          value={editablePrompt}
          onChange={(e) => handlePromptChange(e.target.value)}
          onBlur={(e) => handlePromptChange(e.target.value)}
          className="flex-1 bg-transparent border-b border-border/50 px-2 py-1 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
          placeholder="Enter audio prompt..."
        />

        {
          type === "NARRATOR" && (
            <input
              type="text"
              value={narrator_description}
              onChange={(e) => handleNarratorDescriptionChange(e.target.value)}
              onBlur={(e) => handleNarratorDescriptionChange(e.target.value)}
              className="flex-1 bg-transparent border-b border-border/50 px-2 py-1 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
              placeholder="Enter narrator description..."
            />
          )
        }
        {audioData ? (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="ml-2 flex items-center gap-1 px-2 py-1 text-xs"
              onClick={handleRegenerate}
              disabled={isRegeneratingEffective}
            >
              {isRegeneratingEffective ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin mr-1" /> Regenerating
                </>
              ) : (
                <>
                  <RotateCcw className="w-3 h-3 mr-1" /> Regenerate
                </>
              )}
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>Generating audio...</span>
          </div>
        )}
      </div>

      <div className="flex gap-4">
        {/* Timeline Section */}
        <div className="flex-1 min-w-0">
          {audioData ? (
            <>
              <div className="flex items-center justify-between mb-2">
                <TimelineRuler duration={durationSeconds} width={800} />
                <div className="text-xs font-mono text-muted-foreground">
                  {formatTime(currentTime)} / {formatTime(durationSeconds)}
                </div>
              </div>

              <div className="relative mt-2" style={{ width: '100%', maxWidth: '800px' }}>
                <Waveform
                  width={800}
                  height={80}
                  bars={50}
                  isPlaying={isPlaying}
                  highlightStart={fadeIn}
                  highlightEnd={fadeOut}
                />

                {/* Playhead indicator */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-secondary shadow-[0_0_8px_hsl(var(--secondary))] z-20 pointer-events-none"
                  style={{
                    left: `${(currentTime / durationSeconds) * 100}%`,
                    display: isPlaying ? 'block' : 'none'
                  }}
                />

                {/* Draggable Markers */}
                <DraggableMarker
                  position={fadeIn}
                  containerWidth={800}
                  onPositionChange={handleFadeInChange}
                  label="Fade In"
                  color="primary"
                />
                <DraggableMarker
                  position={fadeOut}
                  containerWidth={800}
                  onPositionChange={handleFadeOutChange}
                  label="Fade Out"
                  color="secondary"
                />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-24 rounded-lg border border-dashed border-border/40 text-xs text-muted-foreground">
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              <span>Generating waveform...</span>
            </div>
          )}

          {/* Curve Selectors */}
          {/* <div className="flex gap-4 mt-4">
            <CurveSelector 
              value={fadeInCurve} 
              onChange={setFadeInCurve}
              type="fade-in"
            />
            <CurveSelector 
              value={fadeOutCurve} 
              onChange={setFadeOutCurve}
              type="fade-out"
            />
          </div> */}
        </div>

        {/* Mixer Section */}
        <div className="flex gap-3 flex-shrink-0">
          <VerticalFader
            value={volume}
            onChange={handleVolumeChange}
            label="VOL"
          />
          
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border/30">
        <Button
          variant="ghost"
          size="sm"
          className={`glass-panel ${isPlaying ? 'neon-glow' : ''} ${!audioData ? 'opacity-50 cursor-not-allowed' : ''}`}
          onClick={handlePlayPause}
          disabled={!audioData}
          title={audioData ? (isPlaying ? 'Pause' : 'Play') : 'No audio loaded'}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4 text-primary" />
          ) : (
            <Play className="w-4 h-4 text-primary" />
          )}
        </Button>


        <div className="flex-1" />

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Volume2 className="w-3 h-3" />
            <span className="font-mono">{volume > 0 ? '+' : ''}{volume.toFixed(1)} dB</span>
          </div>
          {audioData && (
            <div className="flex items-center gap-1 pl-2 border-l border-border/30">
              <span className="text-[10px]">Start:</span>
              <span className="font-mono">{formatTime(start_time_ms / 1000)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Evaluation Section */}
      {audioData && (
        <div className="mt-4 pt-4 border-t border-border/30">
          <button
            onClick={() => setShowEvaluation(!showEvaluation)}
            className="w-full flex items-center justify-between text-sm font-medium text-foreground hover:text-primary transition-colors"
          >
            <div className="flex items-center gap-2">
              <Star className="w-4 h-4" />
              <span>Specialist Model Evaluation</span>
              {Object.values(evaluationScores).some(score => score > 0) && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">
                  {Object.values(evaluationScores).filter(score => score > 0).length}/3
                </span>
              )}
            </div>
            {showEvaluation ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          {showEvaluation && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 space-y-4"
            >
              {/* Evaluator Name (Optional) */}
              <div className="p-3 rounded-lg bg-slate-800/30 border border-border/20">
                <label className="text-xs font-semibold text-foreground mb-2 block">
                  Your Name <span className="text-muted-foreground">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={evaluatorName}
                  onChange={(e) => setEvaluatorName(e.target.value)}
                  placeholder="Enter your name..."
                  className="w-full px-3 py-2 bg-slate-900/50 border border-border/50 rounded-md text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary transition-colors"
                />
              </div>
              {/* Prompt Adherence */}
              <div className="p-3 rounded-lg bg-slate-800/30 border border-border/20">
                <div className="mb-2">
                  <label className="text-xs font-semibold text-foreground">
                    Prompt Adherence (Semantic Alignment)
                  </label>
                  <p className="text-[10px] text-muted-foreground mt-1 italic">
                    Listen to the sound and read the prompt used to generate it. Does it sound exactly like what was requested? (e.g., If the prompt was 'heavy rain,' is it actually heavy rain and not just white noise?)
                  </p>
                </div>
                <div className="flex gap-2 mt-2">
                  {[1, 2, 3, 4, 5].map((num) => (
                    <button
                      key={num}
                      onClick={() => handleEvaluationScoreChange('promptAdherence', num)}
                      className={`flex-1 py-2 rounded-md border transition-all text-xs ${
                        evaluationScores.promptAdherence === num
                          ? "bg-primary border-primary text-primary-foreground shadow-[0_0_8px_hsl(var(--primary))]"
                          : "bg-slate-800/50 border-border/50 text-muted-foreground hover:border-primary/50"
                      }`}
                    >
                      {num}
                    </button>
                  ))}
                </div>
              </div>

              {/* Acoustic Naturalness */}
              <div className="p-3 rounded-lg bg-slate-800/30 border border-border/20">
                <div className="mb-2">
                  <label className="text-xs font-semibold text-foreground">
                    Acoustic Naturalness (Sound Realism)
                  </label>
                  <p className="text-[10px] text-muted-foreground mt-1 italic">
                    Does the sound feel organic and 'real' to the ear, or does it sound like a computer-generated error? Check for robotic metallic ringing or unnatural loops.
                  </p>
                </div>
                <div className="flex gap-2 mt-2">
                  {[1, 2, 3, 4, 5].map((num) => (
                    <button
                      key={num}
                      onClick={() => handleEvaluationScoreChange('acousticNaturalness', num)}
                      className={`flex-1 py-2 rounded-md border transition-all text-xs ${
                        evaluationScores.acousticNaturalness === num
                          ? "bg-primary border-primary text-primary-foreground shadow-[0_0_8px_hsl(var(--primary))]"
                          : "bg-slate-800/50 border-border/50 text-muted-foreground hover:border-primary/50"
                      }`}
                    >
                      {num}
                    </button>
                  ))}
                </div>
              </div>

              {/* Recognition Rate */}
              <div className="p-3 rounded-lg bg-slate-800/30 border border-border/20">
                <div className="mb-2">
                  <label className="text-xs font-semibold text-foreground">
                    Recognition Rate (Object Identification)
                  </label>
                  <p className="text-[10px] text-muted-foreground mt-1 italic">
                    Without looking at the prompt, can you tell what the sound is supposed to be? High recognition means the model successfully captured the essential 'fingerprint' of that sound.
                  </p>
                </div>
                <div className="flex gap-2 mt-2">
                  {[1, 2, 3, 4, 5].map((num) => (
                    <button
                      key={num}
                      onClick={() => handleEvaluationScoreChange('recognitionRate', num)}
                      className={`flex-1 py-2 rounded-md border transition-all text-xs ${
                        evaluationScores.recognitionRate === num
                          ? "bg-primary border-primary text-primary-foreground shadow-[0_0_8px_hsl(var(--primary))]"
                          : "bg-slate-800/50 border-border/50 text-muted-foreground hover:border-primary/50"
                      }`}
                    >
                      {num}
                    </button>
                  ))}
                </div>
              </div>

              {/* Cue Feedback (Optional) */}
              <div className="p-3 rounded-lg bg-slate-800/30 border border-border/20">
                <label className="text-xs font-semibold text-foreground mb-2 block">
                  Cue Feedback <span className="text-muted-foreground">(Optional)</span>
                </label>
                <textarea
                  value={cueFeedback}
                  onChange={(e) => setCueFeedback(e.target.value)}
                  placeholder="Any specific feedback about this audio cue..."
                  className="w-full px-3 py-2 bg-slate-900/50 border border-border/50 rounded-md text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary transition-colors min-h-[60px] resize-none"
                />
              </div>

              {/* Save Button */}
              <div className="mt-4 pt-4 border-t border-border/20">
                <Button
                  onClick={handleSaveEvaluation}
                  disabled={isSavingEvaluation || Object.values(evaluationScores).some(score => score === 0)}
                  className="w-full"
                  variant={saveEvaluationStatus === 'success' ? 'default' : 'default'}
                >
                  {isSavingEvaluation ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : saveEvaluationStatus === 'success' ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Saved!
                    </>
                  ) : saveEvaluationStatus === 'error' ? (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Retry Save
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Save Specialist Model Evaluation
                    </>
                  )}
                </Button>
                {Object.values(evaluationScores).some(score => score === 0) && (
                  <p className="text-xs text-muted-foreground mt-2 text-center">
                    Please evaluate all three parameters before saving
                  </p>
                )}
              </div>
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
});

AudioCard.displayName = "AudioCard";

export default AudioCard;