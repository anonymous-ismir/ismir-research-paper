import { motion } from "framer-motion";

const TimelineRuler = ({ duration = 100, width = 1600 }) => {
  const markers = [];
  const majorInterval = 1; // 1 second
  const minorInterval = 0.5; // 0.5 second
  
  for (let i = 0; i <= duration; i += minorInterval) {
    const isMajor = i % majorInterval === 0;
    markers.push({
      time: i,
      isMajor,
      position: (i / duration) * 100,
    });
  }

  return (
    <div 
      className="relative h-8 bg-muted/20 border-b border-border/50"
      style={{ width }}
    >
      {markers.map((marker, index) => (
        <motion.div
          key={index}
          className="absolute bottom-0"
          style={{ left: `${marker.position}%` }}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.01 }}
        >
          <div
            className={`w-px ${marker.isMajor ? 'h-4 bg-primary' : 'h-2 bg-muted-foreground/50'}`}
          />
          {marker.isMajor && (
            <span className="absolute -translate-x-1/2 top-0 text-[10px] font-mono text-muted-foreground">
              {marker.time}s
            </span>
          )}
        </motion.div>
      ))}
    </div>
  );
};

export default TimelineRuler;
