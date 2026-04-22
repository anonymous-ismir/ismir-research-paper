import { motion } from "framer-motion";
import { useState } from "react";
import { Music2, Loader2 } from "lucide-react";

const MasterMixButton = ({  isLoading, onMix, disabled = false }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleMix = async () => {
    if (disabled || isGenerating) return;
    
    setIsGenerating(true);
    setProgress(0);
    const result = await onMix?.();
    if (result) {
      setIsGenerating(false);
      setProgress(100);
    } else {
      setIsGenerating(false);
      setProgress(0);
    }

  };

  return (
    <div className="flex flex-col items-center gap-4">
      <motion.button
        onClick={handleMix}
        disabled={disabled || isGenerating}
        className={`relative px-12 py-5 rounded-2xl font-display text-lg tracking-wider transition-all ${
          disabled 
            ? 'bg-muted text-muted-foreground cursor-not-allowed' 
            : 'bg-gradient-to-r from-primary via-secondary to-primary bg-[length:200%_100%] text-primary-foreground'
        }`}
        animate={!disabled && !isGenerating ? {
          backgroundPosition: ["0% 0%", "100% 0%", "0% 0%"],
          boxShadow: [
            "0 0 30px hsl(var(--primary) / 0.4)",
            "0 0 50px hsl(var(--secondary) / 0.6)",
            "0 0 30px hsl(var(--primary) / 0.4)",
          ],
        } : {}}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear",
        }}
        whileHover={!disabled ? { scale: 1.05 } : {}}
        whileTap={!disabled ? { scale: 0.95 } : {}}
      >
        {/* Inner glow effect */}
        {!disabled && !isGenerating && (
          <motion.div
            className="absolute inset-0 rounded-2xl"
            animate={{
              boxShadow: [
                "inset 0 0 20px hsl(var(--primary) / 0.2)",
                "inset 0 0 40px hsl(var(--primary) / 0.4)",
                "inset 0 0 20px hsl(var(--primary) / 0.2)",
              ],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
        
        <span className="relative flex items-center gap-3">
          {isGenerating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              GENERATING MASTERPIECE...
            </>
          ) : (
            <>
              <Music2 className="w-5 h-5" />
              MASTER MIX
            </>
          )}
        </span>
      </motion.button>
      
      {/* Progress bar */}
      {isGenerating && (
        <motion.div
          className="w-80 h-2 bg-muted rounded-full overflow-hidden"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <motion.div
            className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(progress, 100)}%` }}
            transition={{ duration: 0.3 }}
          />
        </motion.div>
      )}
      
      {isGenerating && (
        <motion.p
          className="text-xs font-mono text-muted-foreground"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {Math.round(Math.min(progress, 100))}% complete
        </motion.p>
      )}
    </div>
  );
};

export default MasterMixButton;
