import { motion } from "framer-motion";
import { useState, useEffect } from "react";

const VerticalFader = ({ 
  value = 0, 
  min = -20, 
  max = 6, 
  onChange,
  label = "VOL" 
}) => {
  const [localValue, setLocalValue] = useState(value);
  const [peakLevel, setPeakLevel] = useState(0);
  
  const normalizedValue = (localValue - min) / (max - min);
  
  // Animate peak meter
  useEffect(() => {
    const interval = setInterval(() => {
      const randomPeak = Math.random() * normalizedValue;
      setPeakLevel(randomPeak);
    }, 100);
    return () => clearInterval(interval);
  }, [normalizedValue]);

  const handleDrag = (e, info) => {
    const newValue = Math.max(min, Math.min(max, localValue - info.delta.y * 0.2));
    setLocalValue(newValue);
    onChange?.(newValue);
  };

  const dbDisplay = localValue.toFixed(1);
  const isClipping = localValue > 0;

  return (
    <div className="flex flex-col items-center gap-2 p-3 glass-panel">
      {/* dB Display */}
      <div className={`font-mono text-xs ${isClipping ? 'text-destructive' : 'text-primary'}`}>
        {localValue > 0 ? '+' : ''}{dbDisplay} dB
      </div>
      
      <div className="flex gap-2 items-stretch h-40">
        {/* Peak Meter */}
        <div className="w-3 bg-muted rounded-full overflow-hidden relative">
          <motion.div
            className="absolute bottom-0 left-0 right-0 rounded-full"
            style={{
              background: isClipping 
                ? "linear-gradient(to top, hsl(var(--secondary)), hsl(var(--destructive)))"
                : "linear-gradient(to top, hsl(var(--secondary) / 0.5), hsl(var(--secondary)))",
            }}
            animate={{ height: `${peakLevel * 100}%` }}
            transition={{ duration: 0.1 }}
          />
          {/* Peak indicator lines */}
          <div className="absolute top-2 left-0 right-0 h-px bg-destructive/50" />
          <div className="absolute top-1/4 left-0 right-0 h-px bg-primary/30" />
          <div className="absolute top-1/2 left-0 right-0 h-px bg-primary/30" />
          <div className="absolute top-3/4 left-0 right-0 h-px bg-primary/30" />
        </div>
        
        {/* Fader Track */}
        <div className="relative w-6 bg-muted rounded-full">
          {/* Fill */}
          <motion.div
            className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-primary/50 to-primary rounded-full"
            animate={{ height: `${normalizedValue * 100}%` }}
            transition={{ duration: 0.1 }}
          />
          
          {/* Fader Knob */}
          <motion.div
            className="absolute left-1/2 -translate-x-1/2 w-8 h-5 -ml-1 bg-card border-2 border-primary rounded cursor-ns-resize neon-glow"
            style={{ bottom: `calc(${normalizedValue * 100}% - 10px)` }}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={0}
            dragMomentum={false}
            onDrag={handleDrag}
            whileDrag={{ scale: 1.1 }}
          >
            {/* Knob lines */}
            <div className="flex flex-col items-center justify-center h-full gap-0.5">
              <div className="w-4 h-px bg-primary/50" />
              <div className="w-4 h-px bg-primary/50" />
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Label */}
      <div className="font-display text-[10px] tracking-wider text-muted-foreground">
        {label}
      </div>
    </div>
  );
};

export default VerticalFader;
