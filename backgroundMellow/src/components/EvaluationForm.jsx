import { memo, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  UserCheck, 
  ArrowRight,
  Save,
  CheckCircle2,
  Star,
  Loader2
} from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Input } from "./ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";



// Mapping for automated metrics: state key -> display metadata
const automatedMetricsConfig = {
  clapScore: { name: "CLAP Score", target: 0.25, isLower: false, format: "percent" },
  spectralRichness: { name: "Spectral Richness", target: 0.70, isLower: false, format: "float" },
  noiseFloor: { name: "Noise Floor", target: -50, isLower: true, format: "db" },
  audioOnsets: { name: "Audio Onsets", target: 10, isLower: false, format: "number" },
};


const EvaluationForm = memo(({ audioBase64, storyText }) => {
  const [step, setStep] = useState("form");
  const [personName, setPersonName] = useState("");
  const [humanScores, setHumanScores] = useState({
    syncAccuracy: 0,
    semanticFit: 0,
    acousticQuality: 0,
    narrativeFlow: 0,
    cinematicImpact: 0,
  });

  const [autoMetrics, setAutoMetrics] = useState({
    clapScore: 0,
    spectralRichness: 0,
    noiseFloor: 0,
    audioOnsets: 0,
  });
  const [feedback, setFeedback] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // null, 'saving', 'success', 'error'
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);

  const handleScoreChange = useCallback((key, value) => {
    setHumanScores(prev => ({ ...prev, [key]: value }));
  }, []);

  const isFormValid = personName.trim() && Object.values(humanScores).every(v => v > 0);

  // Compute which fields are missing to help the user enable the button
  const missingFields = [];
  if (!personName.trim()) {
    missingFields.push("Your Name");
  }

  const scoreLabelMap = {
    syncAccuracy: "Sync Accuracy",
    semanticFit: "Semantic Fit",
    acousticQuality: "Acoustic Quality",
    narrativeFlow: "Narrative Flow",
    cinematicImpact: "Cinematic Impact",
  };

  Object.entries(humanScores).forEach(([key, value]) => {
    if (value <= 0 && scoreLabelMap[key]) {
      missingFields.push(scoreLabelMap[key]);
    }
  });

  const handleSubmit = useCallback(async () => {
    if (!audioBase64) {
      alert("No audio data available for evaluation.");
      return;
    }

    setIsLoadingMetrics(true);

    // Call backend to get automated metrics for audio evaluation
    let autoMetrics = {
      clapScore: 0,
      spectralRichness: 0,
      noiseFloor: 0,
      audioOnsets: 0,
      message: "",
    };

    try {
      // Extract base64 string (remove data URL prefix if present)
      const base64Data = audioBase64.includes(',') 
        ? audioBase64.split(',')[1] 
        : audioBase64;

      // Call API (use /api/v1/evaluate-audio endpoint)
      const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
      const response = await fetch(`${apiBase}/v1/evaluate-audio`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_base64: base64Data,
          text: storyText || "N/A",
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Adapt keys to expected format for saving
        autoMetrics = {
          clapScore: data.clap_score || 0,
          spectralRichness: data.spectral_richness || 0,
          noiseFloor: data.noise_floor || 0,
          audioOnsets: data.audio_onsets || 0,
          message: data.message || "Successfully evaluated audio"
        };
        console.log("Successfully fetched automated metrics:", autoMetrics);
      } else {
        // Fallback to default values if API fails
        const errorText = await response.text().catch(() => "Unknown error");
        autoMetrics = {
          clapScore: 0,
          spectralRichness: 0,
          noiseFloor: 0,
          audioOnsets: 0,
          message: `Failed to fetch automated metrics: ${response.status} ${errorText}`
        };
        console.error("Failed to get metrics from backend:", response.status, errorText);
      }
    } catch (error) {
      // Fallback to default values on error
      autoMetrics = {
        clapScore: 0,
        spectralRichness: 0,
        noiseFloor: 0,
        audioOnsets: 0,
        message: `Error calling backend: ${error.message || error}`
      };
      console.error("Error fetching audio metrics:", error);
    } finally {
      setIsLoadingMetrics(false);
    }
    
    setAutoMetrics(autoMetrics);
    setStep("results");

  }, [audioBase64, storyText]);


  // Calculate final score from human scores (average * 2 to get 0-10 scale)
  const calculateFinalScore = useCallback(() => {
    const avg = Object.values(humanScores).reduce((sum, val) => sum + val, 0) / Object.values(humanScores).length;
    return (avg / 5) * 10; // Convert 0-5 scale to 0-10 scale
  }, [humanScores]);

  
  const handleSave = useCallback(async (updatedAutoMetrics) => {
    if (!audioBase64) {
      alert("No audio data available to save.");
      return;
    }

    setIsSaving(true);
    setSaveStatus('saving');
    
    try {
      // Convert base64 string to Blob
      const base64Data = audioBase64.includes(',') 
        ? audioBase64.split(',')[1] 
        : audioBase64;
      
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
            requestType: "MASTER", // Required for routing in Google Apps Script
            personName,
            storyPrompt: storyText || "N/A",
            scores: {
              sync: humanScores.syncAccuracy || 0,
              fit: humanScores.semanticFit || 0,
              quality: humanScores.acousticQuality || 0,
              flow: humanScores.narrativeFlow || 0,
              impact: humanScores.cinematicImpact || 0,

            },
            clapScore: updatedAutoMetrics.clapScore || 0,
            spectralRichness: updatedAutoMetrics.spectralRichness || 0,
            noiseFloor: updatedAutoMetrics.noiseFloor || 0,
            audioOnsets: updatedAutoMetrics.audioOnsets || 0,
            finalScore: calculateFinalScore().toFixed(1),
            feedback: feedback || "",
            audioFile: base64Audio,
          };

          console.log("payload :", payload);
    
          // Send to Google Apps Script which will handle:
          // 1. Uploading audio file to Google Drive folder
          // 2. Writing evaluation data to Google Sheets
          await fetch("https://script.google.com/macros/s/AKfycbwaCqI2T56bBqOoLOrxO_zp6Yw7hiHae1BLoqRoF7HeHnVfPxPeTXR4HkzPWL5vKzXJ/exec", {
            method: "POST",
            mode: "no-cors", // Required for cross-origin GAS requests
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
    
          // Since mode is 'no-cors', we won't get a readable response body, 
          // but we can assume success if no error is thrown.
          setSaveStatus('success');
          setTimeout(() => {
            alert("Results saved successfully to Drive and Sheets!");
            setSaveStatus(null);
          }, 500);
        } catch (error) {
          console.error("Save failed:", error);
          setSaveStatus('error');
          setTimeout(() => {
            alert("Error saving data.");
            setSaveStatus(null);
          }, 500);
        } finally {
          setIsSaving(false);
        }
      };
      
      reader.onerror = () => {
        console.error("FileReader error");
        setSaveStatus('error');
        setIsSaving(false);
        setTimeout(() => {
          alert("Error processing audio file.");
          setSaveStatus(null);
        }, 500);
      };
    } catch (error) {
      console.error("Save failed:", error);
      setSaveStatus('error');
      setIsSaving(false);
      setTimeout(() => {
        alert("Error saving data.");
        setSaveStatus(null);
      }, 500);
    }
  }, [personName, humanScores, feedback, audioBase64, storyText, calculateFinalScore]);





  const getStatusColor = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? "text-emerald-400" : "text-rose-400";
  };


  if (step === "form") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel p-6 rounded-xl border border-border/30"
      >
        <div className="flex items-center gap-3 mb-6">
          <UserCheck className="w-5 h-5 text-primary" />
          <div>
            <h2 className="text-lg font-display font-bold text-foreground">Perceptual Evaluation</h2>
            <p className="text-sm text-muted-foreground">Rate the generated audio</p>
          </div>
        </div>

        {/* Person Name */}
        <div className="mb-6">
          <label className="text-sm font-medium text-foreground mb-2 block">Your Name</label>
          <Input
            value={personName}
            onChange={(e) => setPersonName(e.target.value)}
            placeholder="Enter your name..."
            className="bg-muted/30"
          />
        </div>

        {/* Rating Questions */}
        <div className="space-y-6">
          {[
            { id: "syncAccuracy", label: "Sync Accuracy", sub: "Temporal Alignment - Does the sound happen exactly when the narrator says it?" },
            { id: "semanticFit", label: "Semantic Fit", sub: "Contextual Relevance - Does the sound match the meaning of the text?" },
            { id: "acousticQuality", label: "Acoustic Quality", sub: "Audio Fidelity - Is the sound clear, or is it distorted/noisy?" },
            { id: "narrativeFlow", label: "Narrative Flow", sub: "Seamlessness - Do transitions between sounds feel natural?" },
            { id: "cinematicImpact", label: "Cinematic Impact", sub: "Dramatization - Does the audio make the story more engaging?" },
          ].map((q) => (
            <div key={q.id}>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-foreground">{q.label}</label>
                <span className="text-xs text-muted-foreground">{q.sub}</span>
              </div>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((num) => (
                  <button
                    key={num}
                    onClick={() => handleScoreChange(q.id, num)}
                    className={`flex-1 py-2 rounded-md border transition-all ${
                      humanScores[q.id] === num
                        ? "bg-primary border-primary text-primary-foreground"
                        : "bg-muted/30 border-border/50 text-muted-foreground hover:border-primary/50"
                    }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Feedback */}
        <div className="mt-6">
          <label className="text-sm font-medium text-foreground mb-2 block">
            Suggestions <span className="text-muted-foreground">(Optional)</span>
          </label>
          <Textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Any improvement ideas..."
            className="bg-muted/30 min-h-[80px]"
          />
        </div>

        <Button
          disabled={!isFormValid || isLoadingMetrics}
          onClick={handleSubmit}
          className="w-full mt-6"
        >
          {isLoadingMetrics ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Evaluating Audio...
            </>
          ) : (
            <>
              View Results
              <ArrowRight className="ml-2 w-4 h-4" />
            </>
          )}
        </Button>

        {!isFormValid && !isLoadingMetrics && missingFields.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            To continue, please fill:{" "}
            <span className="font-medium text-foreground">
              {missingFields.join(", ")}
            </span>
            .
          </p>
        )}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel p-6 rounded-xl border border-border/30 relative"
    >
      {/* Saving Overlay */}
      {isSaving && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-xl z-50 flex flex-col items-center justify-center gap-4"
        >
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
          <div className="text-center">
            <p className="font-display text-lg text-foreground mb-1">Saving Results</p>
            <p className="text-sm text-muted-foreground">Uploading to Google Drive and Sheets...</p>
          </div>
        </motion.div>
      )}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display text-lg tracking-wider text-foreground">
          EVALUATION RESULTS
        </h3>
        <span className="text-xs font-mono text-muted-foreground">
          Evaluator: {personName}
        </span>
      </div>

      <Table>
        <TableHeader>
          <TableRow className="border-border/30">
            <TableHead className="text-muted-foreground text-xs">SOURCE</TableHead>
            <TableHead className="text-muted-foreground text-xs">METRIC</TableHead>
            <TableHead className="text-muted-foreground text-xs text-right">VALUE</TableHead>
            <TableHead className="text-muted-foreground text-xs text-center">STATUS</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(humanScores).map(([key, val]) => (
            <TableRow key={key} className="border-border/20">
              <TableCell>
                <span className="text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded">Human</span>
              </TableCell>
              <TableCell className="capitalize text-foreground text-sm">
                {key.replace(/([A-Z])/g, ' $1')}
              </TableCell>
              <TableCell className="text-right font-mono text-amber-400">{val}/5</TableCell>
              <TableCell className="text-center">
                {val >= 4 ? <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto" /> : <Star className="w-4 h-4 text-muted-foreground mx-auto" />}
              </TableCell>
            </TableRow>
          ))}
          {autoMetrics && Object.entries(autoMetrics)
            .filter(([key]) => key !== 'message') // Exclude message from display
            .map(([key, value]) => {
              const config = automatedMetricsConfig[key];
              if (!config) return null; // Skip if no config found
              
              const { name, target, isLower, format } = config;
              const passed = isLower ? value <= target : value >= target;
              
              // Format the value based on type
              let formattedValue;
              if (format === "percent") {
                formattedValue = (value * 100).toFixed(0) + '%';
              } else if (format === "db") {
                formattedValue = value.toFixed(1) + ' dB';
              } else if (format === "float") {
                formattedValue = value.toFixed(3);
              } else {
                formattedValue = value.toFixed(0);
              }
              
              return (
                <TableRow key={key} className="border-border/20">
                  <TableCell>
                    <span className="text-[10px] bg-primary/10 text-primary px-2 py-0.5 rounded">Algorithmic</span>
                  </TableCell>
                  <TableCell className="text-foreground text-sm">{name}</TableCell>
                  <TableCell className={`text-right font-mono ${getStatusColor(value, target, isLower)}`}>
                    {formattedValue}
                  </TableCell>
                  <TableCell className="text-center">
                    {passed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto" />
                    ) : (
                      <Star className="w-4 h-4 text-muted-foreground mx-auto" />
                    )}
                  </TableCell>
                </TableRow>
              );
            })
            .filter(Boolean) // Remove null entries
          }
        </TableBody>
      </Table>

      {/* Final Score */}
      <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-emerald-400/70 uppercase tracking-wider">Final Score</p>
          <h4 className="text-2xl font-bold text-emerald-400">
            {calculateFinalScore().toFixed(1)} / 10
          </h4>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setStep("form")}
            disabled={isSaving}
          >
            Re-evaluate
          </Button>
          <Button 
            size="sm" 
            onClick={() => handleSave(autoMetrics)} 
            disabled={isSaving}
            className={saveStatus === 'success' ? 'bg-emerald-500 hover:bg-emerald-600' : ''}
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : saveStatus === 'success' ? (
              <>
                <CheckCircle2 className="w-4 h-4 mr-2" />
                Saved!
              </>
            ) : saveStatus === 'error' ? (
              <>
                <Save className="w-4 h-4 mr-2" />
                Retry Save
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Results
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  );
});

EvaluationForm.displayName = "EvaluationForm";

export default EvaluationForm;
