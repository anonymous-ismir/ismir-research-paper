#!/usr/bin/env node

/**
 * Script to start the FastAPI backend server
 * This script runs the Python backend server using uvicorn
 */

import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Get the backend directory path
const backendDir = path.join(__dirname, "..", "backend");

// Set environment variables
process.env.PORT = process.env.PORT || "8001";
process.env.HOST = process.env.HOST || "0.0.0.0";
process.env.PYTHONUNBUFFERED = "1"; // Ensure Python output is unbuffered

console.log("Starting FastAPI backend server...");
console.log(`Backend directory: ${backendDir}`);
console.log(
  `Server will be available at http://${process.env.HOST}:${process.env.PORT}`,
);
console.log(
  `API Documentation: http://${process.env.HOST}:${process.env.PORT}/docs\n`,
);

// Check if conda environment is specified
const condaEnv = "musicgen";

let pythonCmd;
let args;

// Just run ./run_server (shell script)

pythonCmd = "./run_server.sh";
args = [];

// Spawn the uvicorn process
const backendProcess = spawn(pythonCmd, args, {
  cwd: backendDir,
  stdio: ["inherit", "pipe", "pipe"], // stdin: inherit, stdout: pipe, stderr: pipe
  shell: false,
  env: {
    ...process.env, // Inherit all environment variables including PYTHONUNBUFFERED
  },
});

// Pipe stdout to console
backendProcess.stdout.on("data", (data) => {
  process.stdout.write(data);
});

// Pipe stderr to console
backendProcess.stderr.on("data", (data) => {
  process.stderr.write(data);
});

// Handle process events
backendProcess.on("error", (error) => {
  console.error("Failed to start backend server:", error.message);
  console.error("\nMake sure Python and uvicorn are installed.");
  if (condaEnv) {
    console.error(
      `   Or activate your conda environment first: conda activate ${condaEnv}`,
    );
  }
  console.error(
    "   Install dependencies: pip install -r src/backend/requirements.txt",
  );
  process.exit(1);
});

backendProcess.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`\nBackend server exited with code ${code}`);
    process.exit(code);
  }
});

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.log("\nShutting down backend server...");

  // Try graceful shutdown: send SIGINT, wait for process to exit
  backendProcess.kill("SIGINT");

  // Give backend some time to shut down gracefully
  const timeout = setTimeout(() => {
    if (!backendProcess.killed) {
      console.warn(
        "Backend server did not shut down in time, force killing...",
      );
      backendProcess.kill("SIGKILL"); // Force kill
    }
    // As a last resort, exit anyway
    process.exit(0);
  }, 5000);

  backendProcess.on("exit", () => {
    clearTimeout(timeout);
    console.log("Backend server exited. Cleaning up...");
    // Any additional cleanup (close open handles, etc) can be done here
    process.exit(0);
  });
});

process.on("SIGTERM", () => {
  console.log("\nShutting down backend server...");
  backendProcess.kill("SIGTERM");
  process.exit(0);
});
