//3# ============================================================================
import React, { useState, useEffect, useRef } from 'react';
import { 
  Plus, Send, Bot, User, MessageSquare, Loader2, 
  Trash2, Shield, PanelLeftClose, PanelLeftOpen 
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

function App() {
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(`session_${Date.now()}`);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
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

  useEffect(() => { fetchSessions(); }, []);

  const loadSession = async (id) => {
    setSessionId(id);
    try {
      const res = await fetch(`http://localhost:8000/api/messages/${id}`);
      const data = await res.json();
      setMessages(data);
    } catch (err) {
      console.error("Load messages error:", err);
    }
  };

  const deleteSession = async (e, id) => {
    e.stopPropagation();
    setDeleteConfirmId(id);
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
    } catch (err) {
      console.error("Delete error:", err);
    } finally {
      setDeleteConfirmId(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    setIsLoading(true);
    const q = input;

    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setInput('');
    setSessions(prevSessions => {
      const currentIndex = prevSessions.findIndex(s => s.id === sessionId);
      if (currentIndex === -1) return prevSessions; 
      if (currentIndex === 0) return prevSessions;
      const updatedSessions = [...prevSessions];
      const [currentSession] = updatedSessions.splice(currentIndex, 1);
      return [currentSession, ...updatedSessions];
    });

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
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: accContent };
                return updated;
              });
            }
          }
        });
      }
      fetchSessions(); 
    } catch (e) {
      console.error("Chat error:", e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    document.body.style.fontFamily = "'Sarabun', system-ui, -apple-system, sans-serif";
  }, []);

  return (
    <div className="flex h-screen bg-[#FDFBF7] overflow-hidden font-sans">
      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 animate-in zoom-in-95 slide-in-from-bottom-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 size={20} className="text-red-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800">ลบบทสนทนา</h3>
            </div>
            <p className="text-gray-600 mb-6 leading-relaxed">
              คุณต้องการลบบทสนทนานี้ใช่หรือไม่? การดำเนินการนี้ไม่สามารถย้อนกลับได้
            </p>
            <div className="flex gap-3">
              <button 
                onClick={() => setDeleteConfirmId(null)}
                className="flex-1 px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                ยกเลิก
              </button>
              <button 
                onClick={confirmDelete}
                className="flex-1 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium transition-colors"
              >
                ลบ
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar Section */}
      <aside className={`bg-[#171717] text-white flex flex-col transition-all duration-300 ease-in-out border-r border-white/5 shadow-2xl ${
        isSidebarOpen ? 'w-72' : 'w-0'
      } overflow-hidden relative`}>
        <div className="p-6 flex items-center justify-between min-w-[280px]">
          <h2 className="text-xl font-semibold tracking-tight flex items-center gap-3">
            <div className="w-8 h-8 bg-[#D97757] rounded-lg flex items-center justify-center shadow-lg shadow-[#D97757]/20">
              <Shield size={18} className="text-white" />
            </div>
            <span>IT support</span>
          </h2>
          <button onClick={() => setIsSidebarOpen(false)} className="p-1 hover:bg-white/10 rounded-md transition-colors">
            <PanelLeftClose size={20} className="text-gray-400" />
          </button>
        </div>

        <div className="px-4 mb-4 min-w-[280px]">
          <button 
            onClick={() => { setMessages([]); setSessionId(`session_${Date.now()}`); }}
            className="w-full flex items-center gap-3 bg-white/5 hover:bg-white/10 border border-white/10 p-3 rounded-xl transition-all group"
          >
            <div className="bg-[#D97757] rounded-full p-1"><Plus size={14} /></div>
            <span className="text-sm font-medium">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 min-w-[280px] custom-scrollbar">
          <p className="px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">Recents</p>
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s.id)} 
              className={`group relative p-3 rounded-xl cursor-pointer flex items-center justify-between transition-all ${
                sessionId === s.id ? 'bg-white/10' : 'hover:bg-white/5 text-gray-400 hover:text-white'
              }`}>
              <div className="flex items-center gap-3 truncate pr-6">
                <MessageSquare size={16} className={sessionId === s.id ? 'text-[#D97757]' : 'text-gray-500'} />
                <span className="text-sm font-medium truncate">{s.title}</span>
              </div>
              <button onClick={(e) => deleteSession(e, s.id)} className="opacity-0 group-hover:opacity-100 p-1.5 hover:text-red-400 transition-all">
                <Trash2 size={13}/>
              </button>
            </div>
          ))}
        </div>
      </aside>

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
                  <div className={`p-4 shadow-sm ${
                    m.role === 'user' ? 'bg-[#FFE8DC] text-[#723b21] rounded-t-3xl rounded-bl-3xl rounded-br-lg' 
                                      : 'bg-white text-gray-700 border border-gray-100 rounded-t-3xl rounded-br-3xl rounded-bl-lg'
                  }`}>
                    <div className="prose prose-slate max-w-none text-sm font-medium">
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

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
            <button onClick={handleSend} disabled={isLoading} className="bg-[#D97757] text-white p-4 rounded-2xl shadow-lg hover:bg-[#ff8a50] transition-all disabled:opacity-50">
              {isLoading ? <Loader2 className="animate-spin" size={24} /> : <Send size={24} />}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;