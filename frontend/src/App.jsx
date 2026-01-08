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
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const fetchSessions = () => {
    fetch('http://localhost:8000/api/sessions')
      .then(res => res.json())
      .then(setSessions);
  };

  useEffect(() => { fetchSessions(); }, []);

  const loadSession = async (id) => {
    setSessionId(id);
    const res = await fetch(`http://localhost:8000/api/messages/${id}`);
    const data = await res.json();
    setMessages(data);
  };

  const deleteSession = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm("คุณต้องการลบแชทนี้ใช่หรือไม่?")) return;
    await fetch(`http://localhost:8000/api/sessions/${id}`, { method: 'DELETE' });
    if (sessionId === id) {
      setMessages([]);
      setSessionId(`session_${Date.now()}`);
    }
    fetchSessions();
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
            accContent += line.replace('data: ', '');
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: accContent };
              return updated;
            });
          }
        });
      }
      fetchSessions();
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#FDFBF7] overflow-hidden font-sans">
      
      {/* --- Sidebar Section --- */}
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

      {/* --- Main Content Section --- */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        
        {/* Floating Toggle Button */}
        {!isSidebarOpen && (
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="absolute top-4 left-4 z-50 p-2 bg-white border border-gray-200 rounded-lg shadow-md hover:bg-gray-50 transition-all"
          >
            <PanelLeftOpen size={20} className="text-gray-600" />
          </button>
        )}

        {/* Chat Content (Centered Layout) */}
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
                    m.role === 'user' 
                      ? 'bg-[#FFE8DC] text-[#723b21] rounded-t-3xl rounded-bl-3xl rounded-br-lg' 
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

        {/* Input Area (Centered) */}
        <div className="p-6 bg-white/80 backdrop-blur-md border-t border-gray-100">
          <div className="max-w-4xl mx-auto flex gap-4 items-center">
            <input 
              value={input} 
              onChange={e => setInput(e.target.value)} 
              onKeyPress={e => e.key === 'Enter' && handleSend()}
              className="flex-1 p-4 bg-gray-50 border border-gray-200 rounded-2xl outline-none focus:border-[#D97757] focus:bg-white transition-all text-sm"
              placeholder="ถามเกี่ยวกับ Serial Number อุปกรณ์..."
              disabled={isLoading}
            />
            <button 
              onClick={handleSend} 
              disabled={isLoading} 
              className="bg-[#D97757] text-white p-4 rounded-2xl shadow-lg hover:bg-[#ff8a50] transition-all disabled:opacity-50 active:scale-95"
            >
              {isLoading ? <Loader2 className="animate-spin" size={24} /> : <Send size={24} />}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;