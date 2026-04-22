import { motion } from "framer-motion";
import { useState } from "react";

const SpatialPanPad = ({ 
  x = 0.5, 
  y = 0.5, 
  onChange,
  size = 120 
}) => {
  const [position, setPosition] = useState({ x, y });
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = (e, info) => {
    const newX = Math.max(0, Math.min(1, position.x + info.delta.x / size));
    const newY = Math.max(0, Math.min(1, position.y + info.delta.y / size));
    setPosition({ x: newX, y: newY });
    onChange?.({ x: newX, y: newY });
  };

  // Convert to -100 to +100 display values
  const panDisplay = Math.round((position.x - 0.5) * 200);
  const depthDisplay = Math.round((1 - position.y) * 100);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="font-display text-[10px] tracking-wider text-muted-foreground">
        SPATIAL PAN
      </div>
      
      <div
        className="relative glass-panel overflow-hidden"
        style={{ width: size, height: size }}
      >
        {/* Grid lines */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Horizontal center */}
          <div className="absolute top-1/2 left-0 right-0 h-px bg-border/50" />
          {/* Vertical center */}
          <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border/50" />
          {/* Circular guides */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] rounded-full border border-border/30" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90%] h-[90%] rounded-full border border-border/20" />
        </div>
        
        {/* Labels */}
        <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[8px] text-muted-foreground">L</span>
        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[8px] text-muted-foreground">R</span>
        <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[8px] text-muted-foreground">FAR</span>
        <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[8px] text-muted-foreground">NEAR</span>
        
        {/* Position dot */}
        <motion.div
          className="absolute w-5 h-5 -translate-x-1/2 -translate-y-1/2 cursor-grab active:cursor-grabbing"
          style={{
            left: `${position.x * 100}%`,
            top: `${position.y * 100}%`,
          }}
          drag
          dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
          dragElastic={0}
          dragMomentum={false}
          onDrag={handleDrag}
          onDragStart={() => setIsDragging(true)}
          onDragEnd={() => setIsDragging(false)}
          whileDrag={{ scale: 1.2 }}
        >
          {/* Glow effect */}
          <motion.div
            className="absolute inset-0 rounded-full bg-primary"
            animate={{
              boxShadow: isDragging
                ? "0 0 25px hsl(var(--primary)), 0 0 50px hsl(var(--primary) / 0.5)"
                : "0 0 15px hsl(var(--primary) / 0.5)",
            }}
          />
          {/* Inner dot */}
          <div className="absolute inset-1 rounded-full bg-primary-foreground" />
        </motion.div>
      </div>
      
      {/* Readout */}
      <div className="flex gap-3 text-[10px] font-mono">
        <span className="text-muted-foreground">
          PAN: <span className="text-primary">{panDisplay > 0 ? '+' : ''}{panDisplay}</span>
        </span>
        <span className="text-muted-foreground">
          DEPTH: <span className="text-secondary">{depthDisplay}%</span>
        </span>
      </div>
    </div>
  );
};

export default SpatialPanPad;
