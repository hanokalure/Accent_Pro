import { useMemo, useRef, useState } from "react";
import { float32ToWavBlob } from "./audio";
import { promptBank } from "./promptBank";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export default function App() {
  const [prompt, setPrompt] = useState("This is a test sentence for accent scoring.");
  const [promptIdx, setPromptIdx] = useState(0);
  const [status, setStatus] = useState("Idle");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState("");
  const [result, setResult] = useState(null);

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const pcmChunksRef = useRef([]);

  const promptInfo = useMemo(
    () => `Prompt ${promptIdx + 1} / ${promptBank.length}`,
    [promptIdx]
  );

  function pickPrompt() {
    const i = Math.floor(Math.random() * promptBank.length);
    setPromptIdx(i);
    setPrompt(promptBank[i]);
  }

  async function startRecording() {
    const sampleRate = 16000;
    const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate
    });
    const sourceNode = audioContext.createMediaStreamSource(mediaStream);
    const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    pcmChunksRef.current = [];

    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      pcmChunksRef.current.push(new Float32Array(input));
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    audioContextRef.current = audioContext;
    mediaStreamRef.current = mediaStream;
    sourceNodeRef.current = sourceNode;
    processorNodeRef.current = processorNode;
    setIsRecording(true);
    setStatus("Recording...");
  }

  async function stopRecording() {
    if (!audioContextRef.current || !processorNodeRef.current || !mediaStreamRef.current) {
      return;
    }
    processorNodeRef.current.disconnect();
    sourceNodeRef.current.disconnect();
    mediaStreamRef.current.getTracks().forEach((t) => t.stop());
    await audioContextRef.current.close();
    setIsRecording(false);

    const blob = float32ToWavBlob(pcmChunksRef.current, 16000);
    if (recordingUrl) URL.revokeObjectURL(recordingUrl);
    const url = URL.createObjectURL(blob);
    setRecordingUrl(url);
    setStatus("Processing...");
    await submitAudio(blob);
  }

  async function submitAudio(blob) {
    try {
      const formData = new FormData();
      formData.append("prompt_text", prompt);
      formData.append("audio_file", blob, "recording.wav");
      const res = await fetch(`${API_BASE}/score-compact`, {
        method: "POST",
        body: formData
      });
      const raw = await res.text();
      let data = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        data = { detail: raw || "Non-JSON response from API" };
      }
      if (!res.ok) throw new Error(data.detail || "Failed to score audio");
      setResult(data);
      setStatus("Done");
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  }

  function playCoachAudio() {
    if (!prompt.trim()) return;
    const utter = new SpeechSynthesisUtterance(prompt.trim());
    utter.rate = 0.9;
    utter.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-copy">
          <h1>Accent Coach</h1>
          <p className="muted">
            Practice a prompt, record your voice, and get instant pronunciation feedback.
          </p>
        </div>
        <div className="pill">{status}</div>
      </header>

      <main className="layout">
        <section className="panel practice-panel">
          <h2>Practice</h2>
          <div className="row">
            <label htmlFor="prompt">Prompt text</label>
            <textarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            <div className="controls">
              <button type="button" className="secondary" onClick={pickPrompt}>
                Reload Prompt
              </button>
              <button type="button" className="secondary" onClick={playCoachAudio}>
                Play Coach Audio
              </button>
              <span className="muted">{promptInfo}</span>
            </div>
          </div>

          <div className="actions">
            <button className="primary" onClick={startRecording} disabled={isRecording}>
              Start Recording
            </button>
            <button className="danger" onClick={stopRecording} disabled={!isRecording}>
              Stop Recording
            </button>
          </div>

          {recordingUrl ? (
            <div className="row">
              <audio controls src={recordingUrl} />
            </div>
          ) : null}
        </section>

        <section className="panel result-panel">
          <h2>Result</h2>
          {result ? (
            <section className="result-card">
              <div className="score-wrap">
                <div className="score-label">Overall score</div>
                <div className="score">{result.overall_score}</div>
                <div className="level-chip">{result.level}</div>
              </div>
              <div className="transcript">Transcript: {result.transcript}</div>
              {result.content_mismatch_gate ? (
                <div className="warning">
                  Content mismatch detected: overall score forced to 0. Please say the prompted
                  sentence more closely.
                </div>
              ) : null}
              <h3>Feedback</h3>
              <ul className="feedback-list">
                {(result.feedback || []).length ? (
                  result.feedback.map((f, i) => (
                    <li key={`${f.phone}-${i}`}>
                      {f.phone} ({f.avg_score}): {f.tip}
                    </li>
                  ))
                ) : (
                  <li>Great job. No major weak phoneme patterns detected.</li>
                )}
              </ul>
            </section>
          ) : (
            <div className="placeholder muted">
              Record your voice to view score, transcript, and pronunciation feedback.
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
