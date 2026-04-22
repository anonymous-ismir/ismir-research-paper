import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  CheckCircle2, 
  AlertCircle, 
  TrendingUp, 
  UserCheck, 
  ArrowRight,
  Star
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

const automatedMetrics = [
  { name: "CLAP Score", value: 0.92, target: 0.85, description: "Audio-text alignment" },
  { name: "Spectral Richness", value: 0.78, target: 0.70, description: "Frequency distribution" },
  { name: "Dynamic Range", value: 0.85, target: 0.80, description: "Loudness variation" },
  { name: "Noise Floor", value: -60, target: -50, description: "dB below signal", isLower: true },
];

const EvaluationGrid = () => {
  const [step, setStep] = useState("questions");
  const [humanScores, setHumanScores] = useState({
    syncAccuracy: 0,
    semanticFit: 0,
    acousticQuality: 0,
    narrativeFlow: 0,
    cinematicImpact: 0,
  });
  const [feedback, setFeedback] = useState("");

  const handleScoreChange = (key, value) => {
    setHumanScores((prev) => ({ ...prev, [key]: value }));
  };

  const isFormValid = Object.values(humanScores).every((v) => v > 0);

  const getStatusColor = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? "text-emerald-400" : "text-rose-400";
  };

  const getStatusIcon = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? (
      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
    ) : (
      <AlertCircle className="w-4 h-4 text-rose-400" />
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <AnimatePresence mode="wait">
        {step === "questions" ? (
          /* --- STEP 1: HUMAN FEEDBACK QUESTIONS --- */
          <motion.div
            key="questions"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            className="glass-panel p-8 rounded-xl border border-white/10 bg-slate-900/50 backdrop-blur-md"
          >
            <div className="flex items-center gap-3 mb-8">
              <UserCheck className="w-6 h-6 text-primary" />
              <div>
                <h2 className="text-xl font-display font-bold text-white tracking-tight">Perceptual Evaluation</h2>
                <p className="text-sm text-slate-400">Rate the generated audio based on cinematic standards.</p>
              </div>
            </div>

            <div className="space-y-8">
              {[
                { id: "syncAccuracy", label: "Sync Accuracy", sub: "Temporal Alignment - Does the sound happen exactly when the narrator says it?" },
                { id: "semanticFit", label: "Semantic Fit", sub: "Contextual Relevance - Does the sound match the meaning of the text?" },
                { id: "acousticQuality", label: "Acoustic Quality", sub: "Audio Fidelity - Is the sound clear, or is it distorted/noisy?" },
                { id: "narrativeFlow", label: "Narrative Flow", sub: "Seamlessness - Do transitions between sounds feel natural?" },
                { id: "cinematicImpact", label: "Cinematic Impact", sub: "Dramatization - Does the audio make the story more engaging?" },
              ].map((q) => (
                <div key={q.id} className="space-y-3">
                  <div className="flex justify-between items-end">
                    <label className="text-sm font-medium text-slate-200">{q.label}</label>
                    <span className="text-xs text-slate-500 italic">{q.sub}</span>
                  </div>
                  <div className="flex gap-4">
                    {[1, 2, 3, 4, 5].map((num) => (
                      <button
                        key={num}
                        onClick={() => handleScoreChange(q.id, num)}
                        className={`flex-1 py-3 rounded-md border transition-all ${
                          humanScores[q.id] === num
                            ? "bg-primary border-primary text-white shadow-[0_0_15px_rgba(var(--primary),0.5)]"
                            : "bg-slate-800/50 border-white/5 text-slate-400 hover:border-white/20"
                        }`}
                      >
                        {num}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Feedback Section */}
            <div className="mt-8 space-y-3">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-200">
                  Improvement Suggestions
                </label>
                <span className="text-xs text-slate-500 italic">(Optional)</span>
              </div>
              <Textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Share your thoughts on how the audio could be improved... (e.g., 'The thunder could be more distant and subtle', 'The car engine sound should start more gradually', etc.)"
                className="w-full min-h-[100px] bg-slate-800/50 border-white/10 text-slate-200 placeholder:text-slate-500 focus-visible:border-primary/50 resize-none"
              />
            </div>

            <Button
              disabled={!isFormValid}
              onClick={() => setStep("results")}
              className="w-full mt-10 py-6 bg-primary hover:bg-primary/90 transition-all group"
            >
              Generate Final Analytics
              <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Button>
          </motion.div>
        ) : (
          /* --- STEP 2: FULL RESULTS TABLE --- */
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6 rounded-xl border border-white/10 bg-slate-950 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                <h3 className="font-display text-lg font-semibold tracking-wider text-white">
                  RESEARCH CONSOLIDATION
                </h3>
              </div>
              <div className="text-[10px] px-3 py-1 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono uppercase tracking-widest">
                ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}
              </div>
            </div>

            <Table>
              <TableHeader>
                <TableRow className="border-white/10 hover:bg-transparent">
                  <TableHead className="text-slate-400 font-display text-xs">SOURCE</TableHead>
                  <TableHead className="text-slate-400 font-display text-xs">METRIC</TableHead>
                  <TableHead className="text-slate-400 font-display text-xs text-right">VALUE</TableHead>
                  <TableHead className="text-slate-400 font-display text-xs text-right">TARGET</TableHead>
                  <TableHead className="text-slate-400 font-display text-xs text-center">STATUS</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {/* Render Human Metrics First */}
                {Object.entries(humanScores).map(([key, val], idx) => (
                  <TableRow key={key} className="border-white/5 hover:bg-white/[0.02]">
                    <TableCell className="py-4">
                      <span className="text-[10px] bg-amber-500/10 text-amber-500 border border-amber-500/20 px-2 py-0.5 rounded uppercase">Human</span>
                    </TableCell>
                    <TableCell className="capitalize text-slate-200 text-sm font-medium">{key.replace(/([A-Z])/g, ' $1')}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-amber-400">{val}/5</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-500">≥4/5</TableCell>
                    <TableCell className="text-center">{val >= 4 ? <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto" /> : <Star className="w-4 h-4 text-slate-700 mx-auto" />}</TableCell>
                  </TableRow>
                ))}

                {/* Render Automated Metrics */}
                {automatedMetrics.map((metric, index) => (
                  <TableRow key={metric.name} className="border-white/5 hover:bg-white/[0.02]">
                    <TableCell className="py-4">
                      <span className="text-[10px] bg-primary/10 text-primary border border-primary/20 px-2 py-0.5 rounded uppercase">Algorithmic</span>
                    </TableCell>
                    <TableCell>
                      <div>
                        <span className="text-slate-200 text-sm font-medium">{metric.name}</span>
                        <p className="text-[10px] text-slate-500">{metric.description}</p>
                      </div>
                    </TableCell>
                    <TableCell className={`text-right font-mono text-sm ${getStatusColor(metric.value, metric.target, metric.isLower)}`}>
                      {metric.value < 1 && metric.value > -1
                        ? (metric.value * 100).toFixed(0) + '%'
                        : metric.value + (metric.isLower ? ' dB' : '')}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-500">
                      {metric.value < 1 && metric.value > -1
                        ? '≥' + (metric.target * 100).toFixed(0) + '%'
                        : (metric.isLower ? '≤' : '≥') + metric.target + (metric.isLower ? ' dB' : '')}
                    </TableCell>
                    <TableCell className="flex justify-center py-4">
                      {getStatusIcon(metric.value, metric.target, metric.isLower)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            <div className="mt-8 p-4 bg-emerald-500/5 border border-emerald-500/10 rounded-lg flex items-center justify-between">
               <div className="space-y-1">
                 <p className="text-xs text-emerald-500/70 font-display uppercase tracking-wider">Final Cinematic Score</p>
                 <h4 className="text-2xl font-bold text-emerald-400">8.4 / 10</h4>
               </div>
               <Button variant="outline" onClick={() => setStep("questions")} className="text-xs h-8 border-white/10 hover:bg-white/5">
                 Re-evaluate
               </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default EvaluationGrid;