import { create } from "zustand";

const useAudioStore = create((set, get) => ({
  audioCues: [],
  finalAudio: null,
  isGeneratingMix: false,
  
  setAudioCues: (cues) => set({ audioCues: cues }),
  
  addAudioCue: (cue) => set((state) => ({
    audioCues: [...state.audioCues, { ...cue, id: Date.now() }]
  })),
  
  updateAudioCue: (id, updates) => set((state) => ({
    audioCues: state.audioCues.map((cue) =>
      cue.id === id ? { ...cue, ...updates } : cue
    )
  })),
  
  removeAudioCue: (id) => set((state) => ({
    audioCues: state.audioCues.filter((cue) => cue.id !== id)
  })),
  
  regenerateAudioCue: async (id, prompt) => {
    const state = get();
    // Mark as regenerating
    set({
      audioCues: state.audioCues.map((cue) =>
        cue.id === id ? { ...cue, isRegenerating: true } : cue
      )
    });
    
    try {
      // TODO: Replace with actual API call
      const response = await fetch("/api/regenerate-audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, id })
      });
      
      if (response.ok) {
        const data = await response.json();
        set({
          audioCues: get().audioCues.map((cue) =>
            cue.id === id 
              ? { ...cue, audioBase64: data.audio_base64, isRegenerating: false }
              : cue
          )
        });
      }
    } catch (error) {
      console.log("Regenerate API not implemented yet");
      // Simulate regeneration for demo
      setTimeout(() => {
        set({
          audioCues: get().audioCues.map((cue) =>
            cue.id === id ? { ...cue, isRegenerating: false } : cue
          )
        });
      }, 2000);
    }
  },
  
  setFinalAudio: (audio) => set({ finalAudio: audio }),
  
  setIsGeneratingMix: (value) => set({ isGeneratingMix: value }),
  
  // Save function placeholder
  saveResults: (evaluationData) => {
    console.log("Saving results:", {
      audioCues: get().audioCues,
      finalAudio: get().finalAudio,
      evaluation: evaluationData
    });
    // TODO: Implement actual save logic
    return Promise.resolve({ success: true });
  }
}));

export default useAudioStore;
