import { motion } from "framer-motion";
import { useState } from "react";
import { Sparkles, Wand2 } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import AnimatedLoader from "./ui/AnimatedLoader";

const HeroSection = ({ isLoading, onDecompose , storyText, setStoryText, enableNarrator, setEnableNarrator}) => {


  const handleDecompose = async (e) => {
    e?.preventDefault?.();
    e?.stopPropagation?.();
    console.log("Button clicked! Story text:", storyText);
    
    if (!storyText.trim()) {
      console.log("No text, returning early");
      return;
    }
    
    console.log("Starting decomposition...");
    await onDecompose?.(storyText);
  };

  const handleTextChange = (e) => {
    setStoryText(e.target.value);
  };

  const handleToggleNarrator = () => {
    if (setEnableNarrator) {
      setEnableNarrator((prev) => !prev);
    }
  };

  return (
    <motion.section
      className="relative overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
    >
      {/* Background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-secondary/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto text-center px-4 py-12">
        {/* Logo/Title */}
        <motion.div
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
        >
          <h1 className="font-display text-4xl md:text-6xl font-bold mb-2">
            <span className="text-foreground">Back</span>
            <span className="text-primary neon-text">Ground</span>
            <span className="text-foreground">Mellow</span>
          </h1>
          <p className="font-display text-sm tracking-[0.3em] text-muted-foreground mb-8">
            AI-POWERED CINEMATIC SOUNDSCAPES
          </p>
        </motion.div>

        {/* Story Input */}
        <motion.div
          className="glass-panel p-6 gradient-border"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="font-display text-xs tracking-wider text-muted-foreground">
              STORY INPUT
            </span>
          </div>
          
          <div className="relative">
            <Textarea
              id="story-input"
              value={storyText}
              onChange={handleTextChange}
              placeholder="Enter your narrative... e.g., 'The rain pattered against the window as thunder rolled in the distance. A car engine hummed to life, tires crunching on gravel as it pulled away into the stormy night.'"
              className="w-full min-h-[128px] bg-muted/30 text-foreground placeholder:text-muted-foreground/50 resize-none"
            />
          </div>

          <div className="mt-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-left">
            <div className="text-[11px] text-muted-foreground/90">
              <span className="font-semibold text-foreground">Narrator voice</span>
              <span className="block">
                Turn this on if you want an AI narrator reading your story along with the background soundscape.
              </span>
            </div>
            <Button
              type="button"
              variant={enableNarrator ? "default" : "outline"}
              size="sm"
              onClick={handleToggleNarrator}
              className="px-4 py-2 text-xs font-display tracking-wider"
            >
              {enableNarrator ? "Narrator: ON" : "Narrator: OFF"}
            </Button>
          </div>

          {isLoading ? (
            <div className="mt-6 space-y-2">
              <AnimatedLoader text="DECOMPOSING NARRATIVE..." />
              <p className="text-[11px] text-muted-foreground/80">
                Generating individual audio cues can be slow —  <span className="font-semibold text-foreground">each cue may take up to 4–5 minutes</span>.{" "}
                Cards will appear first, and their audio will load as it finishes.
              </p>
            </div>
          ) : (
            <div className="mt-6 space-y-2">
              <Button
                onClick={handleDecompose}
                disabled={!storyText.trim()}
                className="w-full md:w-auto px-8 py-6 bg-gradient-to-r from-primary to-secondary text-primary-foreground font-display text-sm tracking-wider neon-glow disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Wand2 className="w-4 h-4 mr-2" />
                DECOMPOSE STORY
              </Button>
              <p className="text-[15px] text-muted-foreground/80">
                Note: Generating audio for <span className="font-semibold text-foreground"> each cue</span> can take up to <span className="font-semibold text-foreground">4–5 minutes.</span>{" "}
                
                Backend will generate cues by 2 workers in parallel and You will see all the audio once every cue generation is complete.
                <br>
                </br>
                That is the generation might take upto 10-15 minutes.
              </p>
            </div>
          )}
        </motion.div>
      </div>
    </motion.section>
  );
};

export default HeroSection;
