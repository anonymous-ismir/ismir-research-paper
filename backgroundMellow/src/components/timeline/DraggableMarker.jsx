import { motion } from "framer-motion";
import { useState } from "react";

const DraggableMarker = ({ 
  position = 20, 
  containerWidth = 1600,
  onPositionChange,
  label = "Marker",
  color = "primary" 
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  const colorClasses = {
    primary: {
      bg: "bg-primary",
      border: "border-primary",
      glow: "neon-glow",
      text: "text-primary",
    },
    secondary: {
      bg: "bg-secondary",
      border: "border-secondary", 
      glow: "neon-glow-green",
      text: "text-secondary",
    },
  };

  const colors = colorClasses[color] || colorClasses.primary;
  const timeMs = Math.round(position * 10000); // 10 second timeline

  return (
    <motion.div
      className="absolute top-0 bottom-0 cursor-ew-resize z-10"
      style={{ left: `${position * 100}%` }}
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0}
      dragMomentum={false}
      onDrag={(e, info) => {
        const newPosition = Math.max(0, Math.min(1, position + info.delta.x / containerWidth));
        onPositionChange?.(newPosition);
      }}
      onDragStart={() => setIsDragging(true)}
      onDragEnd={() => setIsDragging(false)}
      onHoverStart={() => setShowTooltip(true)}
      onHoverEnd={() => setShowTooltip(false)}
      whileDrag={{ scale: 1.1 }}
    >
      {/* Marker line */}
      <motion.div
        className={`absolute w-1 h-full ${colors.bg} ${colors.glow} rounded-full -translate-x-1/2`}
        animate={{
          boxShadow: isDragging 
            ? `0 0 30px hsl(var(--${color}))` 
            : `0 0 15px hsl(var(--${color}) / 0.5)`,
        }}
      />
      
      {/* Handle */}
      <motion.div
        className={`absolute -top-1 w-4 h-4 -translate-x-1/2 rounded-full ${colors.bg} ${colors.border} border-2`}
        animate={{
          scale: isDragging ? 1.3 : 1,
        }}
      />
      
      {/* Tooltip */}
      {(showTooltip || isDragging) && (
        <motion.div
          className="absolute -top-10 left-1/2 -translate-x-1/2 px-2 py-1 glass-panel text-xs font-mono whitespace-nowrap"
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <span className={colors.text}>{label}:</span>
          <span className="ml-1 text-foreground">{timeMs}ms</span>
        </motion.div>
      )}
    </motion.div>
  );
};

export default DraggableMarker;
