import React, { useState, useEffect, useRef } from 'react';
import { Plus, Send, Bot, User, MessageSquare, Loader2, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

function App() {
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(`session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom(); }, [messages]);

  const fetchSessions = () => {
    fetch('http://localhost:8000/api/sessions').then(res => res.json()).then(setSessions);
  };

  useEffect(() => { fetchSessions(); }, []);

  const loadSession = async (id) => {
    setSessionId(id);
    const res = await fetch(`http://localhost:8000/api/messages/${id}`);
    const data = await res.json();
    setMessages(data);
  };

  const deleteSession = async (e, id) => {
    e.stopPropagation(); // ไม่ให้โหลดแชทตอนกดลบ
    if (!window.confirm("ต้องการลบประวัติแชทนี้ใช่หรือไม่?")) return;
    await fetch(`http://localhost:8000/api/sessions/${id}`, { method: 'DELETE' });
    if (sessionId === id) { setMessages([]); setSessionId(`session_${Date.now()}`); }
    fetchSessions();
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    setIsLoading(true);
    const currentInput = input;
    setMessages(prev => [...prev, { role: 'user', content: currentInput }]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, question: currentInput }),
        headers: { 'Content-Type': 'application/json' }
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = ""; // ตัวแปรสะสมข้อความแก้ปัญหาค้าง
      
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            accumulatedContent += line.replace('data: ', '');
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: accumulatedContent };
              return updated;
            });
          }
        });
      }
      fetchSessions();
    } catch (e) { console.error(e); }
    setIsLoading(false);
  };

  return (
    <div className="flex h-screen bg-[#F9F6F0]">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1E1E1E] text-white flex flex-col p-4 shadow-xl">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-2">🖥️ IT Support</h2>
        <button onClick={() => { setMessages([]); setSessionId(`session_${Date.now()}`); }}
          className="flex items-center justify-center gap-2 bg-[#D97757] p-3 rounded-lg hover:bg-[#ff8a50] transition-all font-semibold shadow-lg">
          <Plus size={20} /> New Chat
        </button>
        <div className="flex-1 overflow-y-auto mt-6 space-y-1">
          {sessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s.id)}
              className={`group p-3 rounded-lg cursor-pointer flex items-center justify-between transition-all ${
                sessionId === s.id ? 'bg-[#D97757] text-white' : 'text-gray-400 hover:bg-white/10'
              }`}>
              <div className="flex items-center gap-2 truncate text-sm">
                <MessageSquare size={16} /> {s.title}
              </div>
              <button onClick={(e) => deleteSession(e, s.id)} className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400">
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Chat */}
      <main className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-300">
              <Bot size={80} className="mb-4 text-[#D97757]" />
              <p className="text-xl">สวัสดีครับ มีอะไรให้ช่วยไหมจ๊ะ?</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-4 p-5 rounded-2xl shadow-sm ${
              m.role === 'user' ? 'bg-[#FFE8DC] border-l-4 border-[#FF8A50] ml-20' : 'bg-white border mr-20'
            }`}>
              {m.role === 'user' ? <User className="text-[#D97757]" /> : <Bot className="text-gray-500" />}
              <div className="prose prose-slate max-w-none text-sm"><ReactMarkdown>{m.content}</ReactMarkdown></div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-6 bg-white border-t">
          <div className="max-w-4xl mx-auto flex gap-4">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyPress={e => e.key === 'Enter' && handleSend()}
              className="flex-1 p-4 border-2 rounded-2xl outline-none focus:border-[#D97757]" placeholder="พิมพ์คำถามของคุณที่นี่..." disabled={isLoading} />
            <button onClick={handleSend} disabled={isLoading} className="bg-[#D97757] text-white p-4 rounded-2xl shadow-lg hover:scale-105 active:scale-95 disabled:opacity-50">
              {isLoading ? <Loader2 className="animate-spin" size={24} /> : <Send size={24} />}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;