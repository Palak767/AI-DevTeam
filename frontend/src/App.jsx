import React, { useState, useRef } from 'react';
import { 
  Play, Download, Terminal, Code2, ShieldCheck, CheckCircle2, 
  AlertTriangle, RefreshCw, Cpu, Layers, Sparkles, Copy, Check, FileText
} from 'lucide-react';

export default function App() {
  const [prompt, setPrompt] = useState("Build a Student Attendance Management REST API with FastAPI, PostgreSQL, and Pytest");
  const [activeTab, setActiveTab] = useState("main.py");
  const [isRunning, setIsRunning] = useState(false);
  const [copied, setCopied] = useState(false);
  const [filter, setFilter] = useState("all");
  const terminalEndRef = useRef(null);

  const [logs, setLogs] = useState([
    { type: 'info', text: '[System] AutoDev Orchestrator initialized. LangGraph State Machine listening.', time: '18:10:01' },
    { type: 'agent', text: '[Requirement Agent] Analyzing request with Gemini 3.5 Flash...', time: '18:10:02' },
    { type: 'success', text: '[Requirement Agent] RequirementSpec generated: 6 functional specs, 5 acceptance criteria.', time: '18:10:04' },
    { type: 'agent', text: '[System Design Agent] ArchitectureBlueprint created: 18 files, 12 REST endpoints, 5 DB tables.', time: '18:10:06' },
    { type: 'agent', text: '[Code Gen Agent] Scaffolding workspace at ./workspace_output/proj_attendance/', time: '18:10:08' },
    { type: 'agent', text: '[Code Gen Agent] Generated: main.py, models.py, schemas.py, Dockerfile, test_api.py', time: '18:10:10' },
    { type: 'docker', text: '[Docker Sandbox] Spawning container python:3.11-slim mounting /app', time: '18:10:12' },
    { type: 'docker', text: '[Docker Sandbox] Executing pytest --json-report tests/test_api.py', time: '18:10:14' },
    { type: 'warn', text: '[Testing Agent] ⚠️ 1 test failed (404 in test_attendance). Routing to Bug Agent...', time: '18:10:15' },
    { type: 'agent', text: '[Bug Agent] Root cause diagnosed: Missing route registration in main.py. Patch applied (Attempt 1/3).', time: '18:10:17' },
    { type: 'success', text: '[Docker Sandbox] Re-running tests... ✅ 5 passed in 1.42s!', time: '18:10:19' },
    { type: 'success', text: '[Code Review Agent] ChromaDB RAG security audit complete. Quality score: 96/100.', time: '18:10:21' },
    { type: 'success', text: '[LangGraph Engine] Workflow completed cleanly! Project ready for export.', time: '18:10:22' }
  ]);

  const fileContents = {
    "main.py": `from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app import models, schemas
from app.database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Student Attendance Management REST API",
    version="1.0.0",
    description="Multi-Agent Autonomous Generated FastAPI Backend"
)

@app.get("/api/v1/attendance", response_model=list[schemas.AttendanceOut])
def get_attendance(db: Session = Depends(get_db)):
    """Retrieve all student attendance records."""
    return db.query(models.Attendance).all()

@app.post("/api/v1/attendance", status_code=status.HTTP_201_CREATED)
def mark_attendance(record: schemas.AttendanceCreate, db: Session = Depends(get_db)):
    """Mark attendance entry for a student."""
    entry = models.Attendance(**record.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "record_id": entry.id}`,

    "models.py": `from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from datetime import datetime
from app.database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    roll_number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

class Attendance(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"))
    status = Column(String, nullable=False)  # Present, Absent, Late
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("student_id", "timestamp", name="uq_student_attendance_date"),
    )`,

    "schemas.py": `from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Literal

class AttendanceBase(BaseModel):
    student_id: int
    status: Literal["Present", "Absent", "Late"]

class AttendanceCreate(AttendanceBase):
    pass

class AttendanceOut(AttendanceBase):
    id: int
    timestamp: datetime
    
    model_config = ConfigDict(from_attributes=True)`,

    "test_api.py": `import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_attendance_empty():
    response = client.get("/api/v1/attendance")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_mark_attendance_success():
    payload = {"student_id": 1, "status": "Present"}
    response = client.post("/api/v1/attendance", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "success"`,

    "Dockerfile": `FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
  };

  const agents = [
    { id: 1, name: "Requirement Agent", role: "Parses prompt to Pydantic specs", status: "completed" },
    { id: 2, name: "System Design Agent", role: "Blueprints files, routes & DB", status: "completed" },
    { id: 3, name: "Code Gen Agent", role: "Scaffolds multi-file code to disk", status: "completed" },
    { id: 4, name: "Testing Agent", role: "Runs Pytest in Docker sandbox", status: "completed" },
    { id: 5, name: "Bug Agent (Self-Fix)", role: "Diagnoses errors & auto-patches", status: "completed", badge: "1 Patch Applied" },
    { id: 6, name: "Code Review Agent", role: "ChromaDB RAG OWASP audit", status: "completed", badge: "96/100 Score" }
  ];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(fileContents[activeTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRunSimulation = () => {
    setIsRunning(true);
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [
      ...prev,
      { type: 'info', text: `[${time}] User triggered generation run for: "${prompt}"`, time },
      { type: 'agent', text: `[LangGraph] State machine active. Invoking agent cycle...`, time }
    ]);
    setTimeout(() => setIsRunning(false), 1500);
  };

  const filteredLogs = logs.filter(log => {
    if (filter === "errors") return log.type === "warn";
    if (filter === "agents") return log.type === "agent" || log.type === "success";
    return true;
  });

  return (
    <div className="h-screen flex flex-col bg-[#05080f] text-slate-100 font-sans overflow-hidden">
      
      {/* 🌟 SHINING TOP NAVIGATION BAR */}
      <header className="h-14 border-b border-cyan-500/20 bg-[#0a0f1d]/90 backdrop-blur-md px-5 flex items-center justify-between shrink-0 z-20 shadow-lg shadow-cyan-950/40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-400 via-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-cyan-500/40 hover:scale-105 transition-transform duration-300">
            <Cpu className="w-4 h-4 text-cyan-100" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-cyan-300 via-sky-200 to-indigo-300 bg-clip-text text-transparent">
                AI DevTeam / AutoDev
              </span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-950/90 text-cyan-300 border border-cyan-500/40 font-semibold shadow-[0_0_10px_rgba(6,182,212,0.25)]">
                LangGraph v0.2
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Autonomous Multi-Agent Software Engineering IDE</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Active Model Indicator */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-cyan-500/30 text-xs shadow-[0_0_15px_rgba(6,182,212,0.15)] hover:border-cyan-400 transition-all">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            <span className="text-cyan-200 font-mono text-[11px]">Gemini 3.5 Flash Active</span>
          </div>

          {/* Glowing Export Zip Button */}
          <button 
            onClick={() => alert("🎉 Downloading full project: student_attendance_api.zip")}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 via-indigo-600 to-purple-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold text-xs transition-all duration-300 shadow-[0_0_20px_rgba(99,102,241,0.35)] hover:shadow-[0_0_25px_rgba(6,182,212,0.5)] hover:-translate-y-0.5 active:scale-95"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export (.ZIP)</span>
          </button>
        </div>
      </header>

      {/* 🚀 GLOWING COMMAND BAR WITH PRESET CHIPS */}
      <section className="px-5 py-3 border-b border-cyan-500/20 bg-[#080d19]/80 flex flex-col gap-2 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <input 
              type="text" 
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What would you like the AI DevTeam to build?"
              className="w-full bg-[#0c1424] border border-cyan-500/30 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:shadow-[0_0_20px_rgba(6,182,212,0.35)] transition-all duration-300 font-sans"
            />
          </div>

          <button 
            onClick={handleRunSimulation}
            disabled={isRunning}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-400 via-sky-500 to-blue-600 hover:from-cyan-300 hover:to-blue-400 text-slate-950 font-bold text-xs transition-all duration-300 shadow-[0_0_25px_rgba(6,182,212,0.45)] hover:shadow-[0_0_35px_rgba(6,182,212,0.7)] hover:-translate-y-0.5 active:scale-95"
          >
            {isRunning ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>{isRunning ? "Executing..." : "Run AI DevTeam"}</span>
          </button>
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span className="text-slate-400 font-medium">Quick Presets:</span>
          <button 
            onClick={() => setPrompt("Build a Student Attendance Management REST API with FastAPI, PostgreSQL, and Pytest")} 
            className="px-2.5 py-1 rounded-lg bg-[#0e172a] hover:bg-[#131f38] text-slate-300 hover:text-cyan-300 transition-all duration-200 border border-slate-700/80 hover:border-cyan-400 hover:shadow-[0_0_12px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 font-mono text-[10px]"
          >
            🎓 Attendance API
          </button>
          <button 
            onClick={() => setPrompt("Build an E-Commerce Cart & Checkout microservice with FastAPI, Redis, and Pytest")} 
            className="px-2.5 py-1 rounded-lg bg-[#0e172a] hover:bg-[#131f38] text-slate-300 hover:text-cyan-300 transition-all duration-200 border border-slate-700/80 hover:border-cyan-400 hover:shadow-[0_0_12px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 font-mono text-[10px]"
          >
            🛒 E-Commerce API
          </button>
          <button 
            onClick={() => setPrompt("Build a Real-Time Chat WebSocket server with FastAPI and Pytest")} 
            className="px-2.5 py-1 rounded-lg bg-[#0e172a] hover:bg-[#131f38] text-slate-300 hover:text-cyan-300 transition-all duration-200 border border-slate-700/80 hover:border-cyan-400 hover:shadow-[0_0_12px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 font-mono text-[10px]"
          >
            💬 Chat WebSocket
          </button>
        </div>
      </section>

      {/* 🖥️ MAIN 3-COLUMN WORKSPACE */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* 1️⃣ LEFT SIDEBAR: AGENT PIPELINE & QUALITY HUD */}
        <aside className="w-80 border-r border-cyan-500/20 bg-[#070c18] flex flex-col p-4 gap-4 overflow-y-auto shrink-0 shadow-lg">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Agent State Machine
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/40 font-semibold shadow-[0_0_10px_rgba(16,185,129,0.25)]">
              Closed-Loop Active
            </span>
          </div>

          {/* Interactive Agent Cards with Hover Shines */}
          <div className="space-y-2.5">
            {agents.map((agent) => (
              <div 
                key={agent.id}
                className="p-3 rounded-xl bg-[#0c1426]/90 border border-emerald-500/30 flex items-start gap-3 transition-all duration-300 hover:border-cyan-400 hover:shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 cursor-pointer group"
              >
                <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5 border border-emerald-400/40 group-hover:scale-110 group-hover:shadow-[0_0_10px_rgba(16,185,129,0.5)] transition-all">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-center">
                    <h3 className="text-xs font-bold text-slate-100 group-hover:text-cyan-300 transition-colors">
                      {agent.name}
                    </h3>
                    {agent.badge && (
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/40 font-semibold shadow-[0_0_8px_rgba(6,182,212,0.25)]">
                        {agent.badge}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-0.5 group-hover:text-slate-300 transition-colors">
                    {agent.role}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* AUTONOMOUS SELF-CORRECTION HUD WITH RADIANT GLOW */}
          <div className="mt-auto p-4 rounded-xl bg-gradient-to-b from-[#0f1a33] to-[#0a1224] border border-cyan-500/40 shadow-[0_0_20px_rgba(6,182,212,0.2)] hover:border-cyan-300 hover:shadow-[0_0_25px_rgba(6,182,212,0.35)] transition-all duration-300 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                Security Quality Score
              </span>
              <span className="font-mono text-emerald-400 font-bold text-sm shadow-[0_0_10px_rgba(16,185,129,0.3)]">96 / 100</span>
            </div>

            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
              <div className="bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 h-full w-[96%] rounded-full shadow-[0_0_12px_rgba(6,182,212,0.6)]"></div>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] font-mono">
              <div className="bg-[#090f1d] p-2 rounded-lg border border-cyan-500/20 hover:border-cyan-400/50 transition-all">
                <span className="text-slate-400 block text-[10px]">Self-Fix Loop:</span>
                <span className="text-cyan-300 font-bold">1 / 3 Retries</span>
              </div>
              <div className="bg-[#090f1d] p-2 rounded-lg border border-emerald-500/20 hover:border-emerald-400/50 transition-all">
                <span className="text-slate-400 block text-[10px]">OWASP RAG:</span>
                <span className="text-emerald-400 font-bold">Passed</span>
              </div>
            </div>
          </div>
        </aside>

        {/* 2️⃣ CENTER: INTERACTIVE MONACO CODE VIEWER */}
        <div className="flex-1 flex flex-col bg-[#050811] overflow-hidden border-r border-cyan-500/20">
          
          {/* File Tabs Bar with Active Glow */}
          <div className="h-10 bg-[#080d19] border-b border-cyan-500/20 flex items-center justify-between px-3 shrink-0">
            <div className="flex items-center gap-1.5 overflow-x-auto text-xs font-mono">
              {Object.keys(fileContents).map((fileName) => (
                <button
                  key={fileName}
                  onClick={() => setActiveTab(fileName)}
                  className={`px-3 py-1.5 rounded-t-lg transition-all duration-200 flex items-center gap-1.5 text-xs ${
                    activeTab === fileName
                      ? "bg-[#050811] text-cyan-300 border-t-2 border-cyan-400 font-semibold shadow-[0_-5px_15px_rgba(6,182,212,0.25)]"
                      : "text-slate-400 hover:text-slate-200 hover:bg-[#0c1424] hover:border-t-2 hover:border-slate-600"
                  }`}
                >
                  <FileText className={`w-3.5 h-3.5 ${activeTab === fileName ? "text-cyan-400" : "text-slate-500"}`} />
                  <span>{fileName}</span>
                </button>
              ))}
            </div>

            <button 
              onClick={handleCopyCode}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#0e172a] hover:bg-[#13203b] text-slate-300 hover:text-cyan-300 text-[11px] transition-all font-mono border border-slate-700/80 hover:border-cyan-400 hover:shadow-[0_0_12px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 active:scale-95"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied!" : "Copy Code"}</span>
            </button>
          </div>

          {/* High-Contrast Code Editor Body */}
          <div className="flex-1 p-5 overflow-y-auto font-mono text-[12.5px] leading-relaxed text-slate-200 bg-[#050811]">
            <pre>
              <code className="text-slate-200">{fileContents[activeTab]}</code>
            </pre>
          </div>
        </div>

        {/* 3️⃣ RIGHT: REAL-TIME WEBSOCKET TERMINAL LOGS */}
        <div className="w-96 flex flex-col bg-[#050811] overflow-hidden shrink-0">
          <div className="h-10 bg-[#080d19] border-b border-cyan-500/20 flex items-center justify-between px-4 shrink-0">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-slate-200">
              <Terminal className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span>WebSocket Log Stream</span>
            </div>

            {/* Filter buttons with Hover Highlights */}
            <div className="flex items-center gap-1 text-[10px] font-mono">
              <button 
                onClick={() => setFilter("all")} 
                className={`px-2.5 py-0.5 rounded transition-all font-semibold ${
                  filter === "all" ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.3)]" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}
              >
                All
              </button>
              <button 
                onClick={() => setFilter("errors")} 
                className={`px-2.5 py-0.5 rounded transition-all font-semibold ${
                  filter === "errors" ? "bg-rose-950 text-rose-300 border border-rose-500/40 shadow-[0_0_10px_rgba(244,63,94,0.3)]" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}
              >
                Errors
              </button>
              <button 
                onClick={() => setFilter("agents")} 
                className={`px-2.5 py-0.5 rounded transition-all font-semibold ${
                  filter === "agents" ? "bg-indigo-950 text-indigo-300 border border-indigo-500/40 shadow-[0_0_10px_rgba(99,102,241,0.3)]" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}
              >
                Agents
              </button>
            </div>
          </div>

          <div className="flex-1 p-4 font-mono text-[11px] leading-relaxed overflow-y-auto space-y-2 bg-[#04060d]">
            {filteredLogs.map((log, index) => {
              let badgeColor = "text-slate-300";
              if (log.type === 'success') badgeColor = "text-emerald-400 font-semibold";
              if (log.type === 'agent') badgeColor = "text-cyan-300";
              if (log.type === 'warn') badgeColor = "text-amber-300 font-semibold";
              if (log.type === 'docker') badgeColor = "text-indigo-300";

              return (
                <div 
                  key={index} 
                  className="flex items-start gap-2 border-b border-slate-900/80 pb-1.5 hover:bg-slate-900/40 hover:pl-1 transition-all rounded px-1 cursor-default"
                >
                  <span className="text-slate-500 shrink-0 text-[10px] font-mono">{log.time}</span>
                  <span className={`${badgeColor} flex-1 break-words`}>{log.text}</span>
                </div>
              );
            })}
            <div ref={terminalEndRef} />
          </div>
        </div>

      </div>
    </div>
  );
}