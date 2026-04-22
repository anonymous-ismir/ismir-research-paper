import { motion } from "framer-motion";
import { useMemo } from "react";

const Waveform = ({ 
  width = 600, 
  height = 80, 
  bars = 60, 
  isPlaying = false,
  highlightStart = 0.2,
  highlightEnd = 0.8 
}) => {
  const barData = useMemo(() => {
    return Array.from({ length: bars }, (_, i) => ({
      height: Math.random() * 0.6 + 0.2,
      delay: i * 0.02,
    }));
  }, [bars]);

  const barWidth = (width / bars) * 0.7;
  const gap = (width / bars) * 0.3;

  return (
    <div 
      className="relative rounded-lg overflow-hidden bg-muted/30"
      style={{ width, height }}
    >
      {/* Highlight region */}
      <div
        className="absolute top-0 bottom-0 bg-primary/10 border-l-2 border-r-2 border-primary/50"
        style={{
          left: `${highlightStart * 100}%`,
          right: `${(1 - highlightEnd) * 100}%`,
        }}
      />
      
      {/* Waveform bars */}
      <div className="absolute inset-0 flex items-center justify-around px-2">
        {barData.map((bar, i) => {
          const position = i / bars;
          const isHighlighted = position >= highlightStart && position <= highlightEnd;
          
          return (
            <motion.div
              key={i}
              className="rounded-full"
              style={{
                width: barWidth,
                background: isHighlighted 
                  ? "linear-gradient(to top, hsl(var(--primary) / 0.6), hsl(var(--primary)))"
                  : "linear-gradient(to top, hsl(var(--muted-foreground) / 0.3), hsl(var(--muted-foreground) / 0.6))",
                boxShadow: isHighlighted ? "0 0 10px hsl(var(--primary) / 0.5)" : "none",
              }}
              initial={{ height: height * 0.1 }}
              animate={{
                height: isPlaying 
                  ? [height * bar.height * 0.5, height * bar.height, height * bar.height * 0.5]
                  : height * bar.height,
              }}
              transition={{
                duration: isPlaying ? 0.8 : 0.3,
                delay: isPlaying ? bar.delay : 0,
                repeat: isPlaying ? Infinity : 0,
                ease: "easeInOut",
              }}
            />
          );
        })}
      </div>
      
      {/* Playhead */}
      {isPlaying && (
        <motion.div
          className="absolute top-0 bottom-0 w-0.5 bg-secondary neon-glow-green"
          initial={{ left: `${highlightStart * 100}%` }}
          animate={{ left: `${highlightEnd * 100}%` }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      )}
    </div>
  );
};

export default Waveform;
