import { memo, useState, useRef, useCallback, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Download, Volume2 } from "lucide-react";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";

const FinalAudioPlayer = memo(({ audioBase64, duration = 30 }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(80);
  const [actualDuration, setActualDuration] = useState(duration);
  const audioRef = useRef(null);
  const intervalRef = useRef(null);

  const waveformBars = useMemo(() => 
    Array.from({ length: 60 }, () => Math.random() * 0.7 + 0.3),
  []);

  // Update actualDuration when duration prop changes
  useEffect(() => {
    setActualDuration(duration);
  }, [duration]);

  const handlePlayPause = useCallback(() => {
    if (!audioRef.current && audioBase64) {
      audioRef.current = new Audio(`data:audio/wav;base64,${audioBase64}`);
      audioRef.current.volume = volume / 100;
      audioRef.current.onended = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        clearInterval(intervalRef.current);
      };
      audioRef.current.ontimeupdate = () => {
        setCurrentTime(audioRef.current.currentTime);
      };
      // Get actual duration from audio element once loaded
      audioRef.current.onloadedmetadata = () => {
        if (audioRef.current && audioRef.current.duration && !isNaN(audioRef.current.duration)) {
          setActualDuration(audioRef.current.duration);
        }
      };
    }

    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(console.error);
      }
      setIsPlaying(!isPlaying);
    } else {
      // Demo mode
      if (!isPlaying) {
        intervalRef.current = setInterval(() => {
          setCurrentTime(prev => {
            if (prev >= actualDuration) {
              clearInterval(intervalRef.current);
              setIsPlaying(false);
              return 0;
            }
            return prev + 0.1;
          });
        }, 100);
      } else {
        clearInterval(intervalRef.current);
      }
      setIsPlaying(!isPlaying);
    }
  }, [audioBase64, isPlaying, volume, actualDuration]);

  const handleVolumeChange = useCallback((value) => {
    const newVolume = value[0];
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume / 100;
    }
  }, []);

  const handleDownload = useCallback(() => {
    if (audioBase64) {
      const link = document.createElement('a');
      link.href = `data:audio/wav;base64,${audioBase64}`;
      link.download = 'cinemaudio-mix.wav';
      link.click();
    }
  }, [audioBase64]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = actualDuration > 0 ? (currentTime / actualDuration) * 100 : 0;

  return (
    <motion.div
      className="glass-panel p-6 border border-primary/30 rounded-xl"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        <h3 className="font-display text-lg tracking-wider text-foreground">
          FINAL MASTER MIX
        </h3>
      </div>

      {/* Waveform Visualization */}
      <div className="relative h-20 rounded-lg overflow-hidden bg-muted/20 mb-4">
        {/* Progress overlay */}
        <div 
          className="absolute left-0 top-0 bottom-0 bg-primary/10 transition-all"
          style={{ width: `${progress}%` }}
        />

        {/* Waveform bars */}
        <div className="absolute inset-0 flex items-center justify-around px-1">
          {waveformBars.map((height, i) => {
            const barPosition = (i / waveformBars.length) * 100;
            const isPassed = barPosition < progress;
            
            return (
              <div
                key={i}
                className="w-1 rounded-full transition-colors duration-200"
                style={{
                  height: `${height * 100}%`,
                  background: isPassed 
                    ? "linear-gradient(to top, hsl(var(--primary)), hsl(var(--secondary)))"
                    : "hsl(var(--muted-foreground) / 0.4)",
                }}
              />
            );
          })}
        </div>

        {/* Playhead */}
        <motion.div
          className="absolute top-0 bottom-0 w-0.5 bg-secondary shadow-[0_0_10px_hsl(var(--secondary))]"
          style={{ left: `${progress}%` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="icon"
          className={`h-12 w-12 rounded-full ${isPlaying ? 'bg-primary text-primary-foreground' : 'bg-muted/50'}`}
          onClick={handlePlayPause}
        >
          {isPlaying ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5 ml-0.5" />
          )}
        </Button>

        {/* Time */}
        <div className="font-mono text-sm text-muted-foreground">
          {formatTime(currentTime)} / {formatTime(actualDuration)}
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2 flex-1 max-w-[150px]">
          <Volume2 className="w-4 h-4 text-muted-foreground" />
          <Slider
            value={[volume]}
            onValueChange={handleVolumeChange}
            min={0}
            max={100}
            className="flex-1"
          />
        </div>

        {/* Download */}
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          onClick={handleDownload}
          disabled={!audioBase64}
        >
          <Download className="w-4 h-4 mr-2" />
          Download
        </Button>
      </div>
    </motion.div>
  );
});

FinalAudioPlayer.displayName = "FinalAudioPlayer";

export default FinalAudioPlayer;
