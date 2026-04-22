import { motion } from "framer-motion";

const AnimatedLoader = ({ text = "Loading..." }) => {
  return (
    <div className="flex flex-col items-center justify-center gap-6">
      {/* Marshmello-inspired loader */}
      <div className="relative w-24 h-24">
        {/* Outer glow ring */}
        <motion.div
          className="absolute inset-0 rounded-2xl border-2 border-primary/50"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 0.2, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        {/* Inner rotating squares */}
        <motion.div
          className="absolute inset-2 rounded-xl bg-gradient-to-br from-primary/20 to-secondary/20 backdrop-blur-sm"
          animate={{ rotate: 360 }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "linear",
          }}
        />
        
        {/* Face container */}
        <div className="absolute inset-4 rounded-lg bg-card border border-border/50 flex items-center justify-center overflow-hidden">
          {/* Eyes */}
          <motion.div
            className="flex gap-3"
            animate={{ y: [0, -2, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <div className="w-2 h-2 rounded-full bg-primary neon-glow" />
            <div className="w-2 h-2 rounded-full bg-primary neon-glow" />
          </motion.div>
        </div>
        
        {/* Orbiting particles */}
        {[...Array(4)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 rounded-full bg-secondary"
            style={{
              top: "50%",
              left: "50%",
            }}
            animate={{
              x: [0, Math.cos((i * Math.PI) / 2) * 40, 0],
              y: [0, Math.sin((i * Math.PI) / 2) * 40, 0],
              opacity: [0.8, 0.3, 0.8],
            }}
            transition={{
              duration: 2,
              delay: i * 0.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
      
      {/* Loading text with shimmer */}
      <motion.p
        className="font-display text-sm tracking-widest text-primary neon-text"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        {text}
      </motion.p>
      
      {/* Progress bar */}
      <div className="w-48 h-1 bg-muted rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
          initial={{ x: "-100%" }}
          animate={{ x: "100%" }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>
    </div>
  );
};

export default AnimatedLoader;
