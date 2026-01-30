import React, { useState, useEffect, useRef } from 'react';
import { 
  Plus, Send, Bot, User, MessageSquare, Loader2, 
  Trash2, Shield, PanelLeftClose, PanelLeftOpen,
  Copy, Check, MoreVertical, Edit2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// --- [FIX] ฟังก์ชันแก้ภาษาไทยและตัวเลขเว้นวรรค ---
// ประกาศไว้ตรงนี้เพื่อให้ทุกส่วนเรียกใช้ได้แน่นอน
const cleanThaiSpacing = (text) => {
  if (!text) return "";
  return text
    // 1. ลบช่องว่างระหว่างตัวอักษรไทย (เช่น "ส ว ั ส ด ี" -> "สวัสดี")
    .replace(/([\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])/g, '$1')
    // 2. ลบช่องว่างระหว่างตัวเลข (เช่น "7 6" -> "76") เพื่อแก้ปัญหาปี/จำนวน
    .replace(/(\d)\s+(?=\d)/g, '$1');
};

// --- Sub-Component สำหรับปุ่ม Copy ---
const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      // Clean text ก่อน copy ด้วย เพื่อให้ได้ข้อความที่สวยงาม
      const cleanText = cleanThaiSpacing(text);
      await navigator.clipboard.writeText(cleanText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) { console.error('Failed to copy!', err); }
  };
  return (
    <button 
      onClick={handleCopy}
      className="p-1.5 hover:bg-gray-100 rounded-md transition-colors text-gray-400 hover:text-[#D97757]"
      title="คัดลอกข้อความ"
    >
      {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
    </button>
  );
};

function App() {
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(`session_${Date.now()}`);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [newTitle, setNewTitle] = useState('');
  const [activeMenuId, setActiveMenuId] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const fetchSessions = () => {
    fetch('http://localhost:8000/api/sessions')
      .then(res => res.json())
      .then(data => setSessions(data || []))
      .catch(err => console.error("Fetch sessions error:", err));
  };

  useEffect(() => { 
    fetchSessions(); 
    document.body.style.fontFamily = "'Sarabun', system-ui, -apple-system, sans-serif";
  }, []);

  const loadSession = async (id) => {
    if (editingSessionId === id) return;
    setSessionId(id);
    try {
      const res = await fetch(`http://localhost:8000/api/messages/${id}`);
      const data = await res.json();
      setMessages(data);
    } catch (err) { console.error("Load messages error:", err); }
  };

  const handleRename = async (id) => {
    if (!newTitle.trim()) { setEditingSessionId(null); return; }
    try {
      await fetch(`http://localhost:8000/api/sessions/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
      });
      setEditingSessionId(null);
      fetchSessions();
    } catch (err) { console.error("Rename error:", err); }
  };

  const confirmDelete = async () => {
    const id = deleteConfirmId;
    try {
      await fetch(`http://localhost:8000/api/sessions/${id}`, { method: 'DELETE' });
      if (sessionId === id) {
        setMessages([]);
        setSessionId(`session_${Date.now()}`);
      }
      fetchSessions();
    } catch (err) { console.error("Delete error:", err); } finally {
      setDeleteConfirmId(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    setIsLoading(true);
    const q = input;
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setInput('');
    
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, question: q }),
        headers: { 'Content-Type': 'application/json' }
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accContent = "";
      
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            const content = line.substring(6);
            if (content) {
              accContent += content;
              // เรียกใช้ฟังก์ชัน cleanThaiSpacing ตรงนี้ (ตอนนี้ฟังก์ชันมีอยู่จริงแล้ว)
              const displayContent = cleanThaiSpacing(accContent); 
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: displayContent };
                return updated;
              });
            }
          }
        });
      }
      fetchSessions(); 
    } catch (e) { console.error("Chat error:", e); } finally { setIsLoading(false); }
  };

  return (
    <div className="flex h-screen bg-[#FDFBF7] overflow-hidden font-sans">
      
      {/* Modal ยืนยันการลบ */}
      {deleteConfirmId && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[100] flex items-center justify-center p-4 animate-in fade-in">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 animate-in zoom-in-95">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 size={20} className="text-red-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800">ลบบทสนทนา</h3>
            </div>
            <p className="text-gray-600 mb-6">คุณต้องการลบบทสนทนานี้ใช่หรือไม่? การดำเนินการนี้ไม่สามารถย้อนกลับได้</p>
            <div className="flex gap-3">
              <button onClick={() => setDeleteConfirmId(null)} className="flex-1 px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium">ยกเลิก</button>
              <button onClick={confirmDelete} className="flex-1 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium">ลบ</button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar Section */}
      <aside className={`bg-[#171717] text-white flex flex-col transition-all duration-300 ease-in-out border-r border-white/5 shadow-2xl ${isSidebarOpen ? 'w-72' : 'w-0'} overflow-hidden relative`}>
        <div className="p-6 flex items-center justify-between min-w-[280px]">
          <h2 className="text-xl font-semibold tracking-tight flex items-center gap-3">
            <div className="w-8 h-8 bg-[#D97757] rounded-lg flex items-center justify-center shadow-lg shadow-[#D97757]/20">
              <Shield size={18} className="text-white" />
            </div>
            <span>IT support</span>
          </h2>
          <button onClick={() => setIsSidebarOpen(false)} className="p-1 hover:bg-white/10 rounded-md transition-colors text-gray-400">
            <PanelLeftClose size={20} />
          </button>
        </div>

        <div className="px-4 mb-4 min-w-[280px]">
          <button onClick={() => { setMessages([]); setSessionId(`session_${Date.now()}`); }}
            className="w-full flex items-center gap-3 bg-white/5 hover:bg-white/10 border border-white/10 p-3 rounded-xl transition-all">
            <div className="bg-[#D97757] rounded-full p-1"><Plus size={14} /></div>
            <span className="text-sm font-medium">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 min-w-[280px] custom-scrollbar">
          <p className="px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">Recents</p>
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s.id)} 
              className={`group relative p-3 rounded-xl cursor-pointer flex items-center justify-between transition-all ${sessionId === s.id ? 'bg-white/10' : 'hover:bg-white/5 text-gray-400 hover:text-white'}`}>
              
              <div className="flex items-center gap-3 truncate pr-2 flex-1">
                <MessageSquare size={16} className={sessionId === s.id ? 'text-[#D97757]' : 'text-gray-500'} />
                {editingSessionId === s.id ? (
                  <input autoFocus value={newTitle} onChange={e => setNewTitle(e.target.value)}
                    onBlur={() => handleRename(s.id)} onKeyDown={e => e.key === 'Enter' && handleRename(s.id)}
                    className="bg-white/10 border border-[#D97757] rounded px-2 py-0.5 text-sm outline-none w-full"
                    onClick={e => e.stopPropagation()} />
                ) : (
                  <span className="text-sm font-medium truncate">{s.title}</span>
                )}
              </div>

              {/* Action Menu */}
              <div className="relative">
                <button onClick={(e) => { e.stopPropagation(); setActiveMenuId(activeMenuId === s.id ? null : s.id); }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all">
                  <MoreVertical size={14} />
                </button>
                {activeMenuId === s.id && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setActiveMenuId(null)} />
                    <div className="absolute right-0 mt-2 w-32 bg-[#212121] border border-white/10 rounded-lg shadow-xl z-20 py-1 overflow-hidden animate-in fade-in zoom-in-95">
                      <button onClick={(e) => { e.stopPropagation(); setEditingSessionId(s.id); setNewTitle(s.title); setActiveMenuId(null); }}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-white/5 text-gray-300">
                        <Edit2 size={12} /> เปลี่ยนชื่อ
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(s.id); setActiveMenuId(null); }}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-white/5 text-red-400">
                        <Trash2 size={12} /> ลบแชท
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {!isSidebarOpen && (
          <button onClick={() => setIsSidebarOpen(true)} className="absolute top-4 left-4 z-50 p-2 bg-white border border-gray-200 rounded-lg shadow-md hover:bg-gray-50 transition-all">
            <PanelLeftOpen size={20} className="text-gray-600" />
          </button>
        )}

        <div className="flex-1 overflow-y-auto p-4 md:p-8">
          <div className={`max-w-4xl mx-auto space-y-8 ${!isSidebarOpen ? 'pt-12' : ''}`}>
            {messages.length === 0 && (
              <div className="h-[60vh] flex flex-col items-center justify-center text-gray-300">
                <Bot size={64} className="mb-4 text-[#D97757] opacity-20" />
                <p className="text-xl font-medium text-gray-400">วันนี้ให้ช่วยตรวจสอบอะไรดีคะ?</p>
              </div>
            )}
            
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                <div className={`flex gap-3 max-w-[85%] md:max-w-[75%] ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className="shrink-0 flex items-end mb-1">
                    <div className={`p-2 rounded-2xl shadow-sm ${m.role === 'user' ? 'bg-[#D97757] text-white' : 'bg-white text-gray-500 border border-gray-100'}`}>
                      {m.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                    </div>
                  </div>
                  
                  <div className="relative group">
                    <div className={`p-4 shadow-sm ${
                      m.role === 'user' ? 'bg-[#FFE8DC] text-[#723b21] rounded-t-3xl rounded-bl-3xl rounded-br-lg' 
                                       : 'bg-white text-gray-700 border border-gray-100 rounded-t-3xl rounded-br-3xl rounded-bl-lg'
                    }`}>
                      <div className="prose prose-slate max-w-none text-sm font-medium">
                        {/* --- [FIX] ใช้ฟังก์ชัน cleanThaiSpacing ครอบตรงนี้เพื่อให้ Render ใหม่สวยงามเสมอ --- */}
                        <ReactMarkdown>{cleanThaiSpacing(m.content)}</ReactMarkdown>
                      </div>
                    </div>
                    <div className={`absolute -bottom-8 ${m.role === 'user' ? 'left-0' : 'right-0'} opacity-0 group-hover:opacity-100 transition-opacity`}>
                      <CopyButton text={m.content} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white/80 backdrop-blur-md border-t border-gray-100">
          <div className="max-w-4xl mx-auto flex gap-4 items-center">
            <input 
              value={input} 
              onChange={e => setInput(e.target.value)} 
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              className="flex-1 p-4 bg-gray-50 border border-gray-200 rounded-2xl outline-none focus:border-[#D97757] focus:bg-white transition-all text-sm"
              placeholder="ถามอะไรซักอย่าง..."
              disabled={isLoading}
            />
            <button onClick={handleSend} disabled={isLoading || !input.trim()} className="bg-[#D97757] text-white p-4 rounded-2xl shadow-lg hover:bg-[#ff8a50] transition-all disabled:opacity-50">
              {isLoading ? <Loader2 className="animate-spin" size={24} /> : <Send size={24} />}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;