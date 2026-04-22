import { motion } from "framer-motion";
import { useState } from "react";

const curves = [
  { 
    id: "linear", 
    label: "Linear",
    path: "M 5 35 L 35 5"
  },
  { 
    id: "logarithmic", 
    label: "Log",
    path: "M 5 35 Q 5 5, 35 5"
  },
  { 
    id: "exponential", 
    label: "Exp",
    path: "M 5 35 Q 35 35, 35 5"
  },
  { 
    id: "sigmoid", 
    label: "S-Curve",
    path: "M 5 35 C 5 20, 35 20, 35 5"
  },
];

const CurveSelector = ({ value = "linear", onChange, type = "fade-in" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const selected = curves.find(c => c.id === value) || curves[0];

  return (
    <div className="relative">
      <div className="font-display text-[10px] tracking-wider text-muted-foreground mb-1 text-center">
        {type === "fade-in" ? "FADE IN" : "FADE OUT"} CURVE
      </div>
      
      {/* Selected curve preview */}
      <motion.button
        className="glass-panel p-2 flex items-center gap-2 w-full"
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <svg width="40" height="40" viewBox="0 0 40 40" className="flex-shrink-0">
          <motion.path
            d={selected.path}
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="2"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.5 }}
          />
        </svg>
        <span className="text-xs text-foreground">{selected.label}</span>
      </motion.button>
      
      {/* Dropdown */}
      {isOpen && (
        <motion.div
          className="absolute top-full left-0 right-0 mt-1 glass-panel p-2 z-20 grid grid-cols-2 gap-2"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {curves.map((curve) => (
            <motion.button
              key={curve.id}
              className={`p-2 rounded-lg flex flex-col items-center gap-1 transition-colors ${
                curve.id === value 
                  ? 'bg-primary/20 border border-primary/50' 
                  : 'hover:bg-muted/50'
              }`}
              onClick={() => {
                onChange?.(curve.id);
                setIsOpen(false);
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <svg width="30" height="30" viewBox="0 0 40 40">
                <path
                  d={curve.path}
                  fill="none"
                  stroke={curve.id === value ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"}
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
              <span className="text-[10px] text-muted-foreground">{curve.label}</span>
            </motion.button>
          ))}
        </motion.div>
      )}
    </div>
  );
};

export default CurveSelector;
