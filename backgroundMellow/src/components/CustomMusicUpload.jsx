import { useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

const CustomMusicUpload = ({ onSave }) => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setFile(selected || null);
    setStatusMessage("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !name.trim()) {
      setStatusMessage("Please provide your name and select an audio file.");
      return;
    }

    try {
      setIsSubmitting(true);
      setStatusMessage("");
      await onSave?.(name.trim(), description.trim(), file);
      setStatusMessage("Thank you! Your music has been submitted.");
      setDescription("");
      setFile(null);
      (document.getElementById("custom-music-file-input") || {}).value = "";
    } catch {
      setStatusMessage("Something went wrong while uploading. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <motion.div
      className="glass-panel p-6 rounded-xl border border-border/40 bg-slate-900/60 backdrop-blur-md space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-start gap-3">
        <div className="mt-1">
          <UploadCloud className="w-5 h-5 text-primary" />
        </div>
        <div className="flex-1 space-y-1">
          <h3 className="font-display text-lg text-foreground">
            Got your own music?
          </h3>
          <p className="text-xs text-muted-foreground">
            Got your own music or want to help increase the dataset?{" "}
            <span className="font-semibold text-foreground">
              Your music will help the model get better.
            </span>
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              Your name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="How should we credit you?"
              className="w-full px-3 py-2 rounded-md bg-slate-900/70 border border-border/50 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary transition-colors"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              Audio file
            </label>
            <input
              id="custom-music-file-input"
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="w-full text-xs text-muted-foreground file:mr-3 file:px-3 file:py-1.5 file:rounded-md file:border-0 file:bg-primary/20 file:text-primary file:text-xs file:font-medium file:cursor-pointer cursor-pointer"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">
            Description of your audio
          </label>
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe the mood, instruments, or use-case for this track..."
            className="min-h-[80px] bg-slate-900/70 text-sm text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="flex items-center justify-between gap-3 pt-2">
          <p className="text-[11px] text-muted-foreground max-w-md">
            By uploading, you agree that this audio may be used to{" "}
            improve our models and datasets.
          </p>
          <Button
            type="submit"
            disabled={isSubmitting || !file || !name.trim()}
            className="px-4 py-2 text-xs font-semibold tracking-wide"
          >
            {isSubmitting ? "Uploading..." : "Upload Music"}
          </Button>
        </div>

        {statusMessage && (
          <p className="text-[11px] text-muted-foreground mt-1">
            {statusMessage}
          </p>
        )}
      </form>
    </motion.div>
  );
};

export default CustomMusicUpload;

