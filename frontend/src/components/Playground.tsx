import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Send,
  FileText,
  Sparkles,
  ChevronDown,
  RotateCcw,
  BookOpen,
  Copy,
  Check,
  Cpu,
  Layers,
  ArrowUpRight,
  Loader2,
  Info,
  ArrowDown
} from 'lucide-react';

interface ChatMessage {
  id: string;
  role: 'Human' | 'AI';
  content: string;
  rewritten?: string;
  sources?: Array<{ content: string; metadata: Record<string, unknown> }>;
  timestamp: string;
}

interface PlaygroundProps {
  documents: string[];
  activeDoc: string | null;
  onSelectDoc: (doc: string) => void;
  onUploadSuccess: (docName: string) => void;
  selectedModel: string;
  availableModels: string[];
  onSelectModel: (model: string) => void;
  hasKey: boolean;
  injectedPrompt?: string | null;
  onClearInjectedPrompt?: () => void;
}

const formatModelLabel = (modelId: string) => {
  if (modelId.includes('120b')) return 'GPT-OSS 120B';
  if (modelId.includes('20b')) return 'GPT-OSS 20B';
  if (modelId.includes('qwen')) return 'Qwen 27B';
  return modelId.split('/').pop() || modelId;
};

// Strips out unwanted markdown asterisks and leading dash bullets
const cleanTextOutput = (text: string): string => {
  if (!text) return '';
  let cleaned = text.replace(/\*{1,3}/g, '');
  cleaned = cleaned.replace(/^\s*[-•–—]\s+/gm, '');
  return cleaned;
};

export const Playground: React.FC<PlaygroundProps> = ({
  documents,
  activeDoc,
  onSelectDoc,
  onUploadSuccess,
  selectedModel,
  availableModels,
  onSelectModel,
  hasKey,
  injectedPrompt,
  onClearInjectedPrompt,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeSourcesModal, setActiveSourcesModal] = useState<Array<{ content: string; metadata: Record<string, unknown> }> | null>(null);
  const [showScrollBottomBtn, setShowScrollBottomBtn] = useState(false);

  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isUserNearBottomRef = useRef(true);

  // Smoothly scroll ONLY the chat container, NEVER touching the window scroll
  const scrollChatToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const container = messagesContainerRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior,
    });
    isUserNearBottomRef.current = true;
    setShowScrollBottomBtn(false);
  }, []);

  // Monitor user scroll inside the chat container
  const handleContainerScroll = () => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const threshold = 120; // px from bottom
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    const isAtBottom = distanceFromBottom <= threshold;
    isUserNearBottomRef.current = isAtBottom;

    if (!isAtBottom && messages.length > 2) {
      setShowScrollBottomBtn(true);
    } else {
      setShowScrollBottomBtn(false);
    }
  };

  // When messages or loading change: only auto-scroll if user is near bottom
  useEffect(() => {
    if (isUserNearBottomRef.current) {
      scrollChatToBottom('smooth');
    }
  }, [messages, isLoading, scrollChatToBottom]);

  // Adjust textarea height dynamically up to max 140px
  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(cleanTextOutput(text));
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF document.');
      return;
    }

    setIsUploading(true);
    setUploadMessage('Extracting pages & computing Chroma embeddings...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      setUploadMessage(`Indexed ${data.page_count} pages (${data.chunk_count} chunks) into Chroma vector store.`);
      onUploadSuccess(data.document_name);
      setTimeout(() => setUploadMessage(null), 4000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      alert(`Error uploading document: ${msg}`);
      setUploadMessage(null);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSend = async (queryText?: string) => {
    const textToSend = (queryText !== undefined ? queryText : input).trim();
    if (!textToSend || isLoading) return;

    if (!hasKey) {
      alert('Please ensure GROQ_API_KEY is set in your .env file.');
      return;
    }

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'Human',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Append user message immediately
    setMessages((prev) => [...prev, userMsg]);
    if (!queryText) {
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }

    // Always scroll to show user's newly submitted question
    isUserNearBottomRef.current = true;
    setTimeout(() => scrollChatToBottom('smooth'), 50);

    setIsLoading(true);

    try {
      const historyPayload = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMsg.content,
          document_name: activeDoc,
          chat_history: historyPayload,
          model: selectedModel,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Query failed');
      }

      const data = await res.json();

      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'AI',
        content: cleanTextOutput(data.answer),
        rewritten: data.rewritten_question !== userMsg.content ? data.rewritten_question : undefined,
        sources: data.sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Query failed';
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'AI',
        content: `Error: ${msg}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If user is actively typing with an IME, don't trigger send
    if (e.nativeEvent.isComposing) return;

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (injectedPrompt) {
      handleSend(injectedPrompt);
      if (onClearInjectedPrompt) {
        onClearInjectedPrompt();
      }
    }
  }, [injectedPrompt]);

  const clearChat = () => {
    setMessages([]);
  };

  const displayModels = availableModels.length > 0
    ? availableModels
    : ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b'];

  return (
    <div className="relative w-full h-[calc(100dvh-4rem)] max-h-[calc(100vh-4rem)] bg-[#07070a] overflow-hidden flex flex-col font-sans">
      {/* Top Header Control Bar - Pinned at top */}
      <div className="border-b border-white/[0.08] bg-[#0c0c12]/90 backdrop-blur-xl px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3 shrink-0 z-10">
        {/* Left: Document Selector & Upload */}
        <div className="flex items-center space-x-2.5">
          <div className="relative inline-flex items-center">
            <FileText className="absolute left-3 h-3.5 w-3.5 text-cyan-400 pointer-events-none" />
            <select
              value={activeDoc || ''}
              onChange={(e) => onSelectDoc(e.target.value)}
              className="rounded-xl bg-white/[0.06] border border-white/[0.12] pl-8 pr-8 py-1.5 text-xs text-white appearance-none focus:outline-none focus:border-cyan-400/50 cursor-pointer font-medium max-w-[200px] sm:max-w-xs truncate"
            >
              {documents.length === 0 ? (
                <option value="" disabled className="bg-[#121216] text-white">
                  No documents indexed yet
                </option>
              ) : (
                documents.map((doc) => (
                  <option key={doc} value={doc} className="bg-[#121216] text-white">
                    {doc}
                  </option>
                ))
              )}
            </select>
            <ChevronDown className="absolute right-2.5 h-3.5 w-3.5 text-white/40 pointer-events-none" />
          </div>

          {/* Upload PDF */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="application/pdf"
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.1] text-xs font-medium text-white transition-colors disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
            ) : (
              <Upload className="h-3.5 w-3.5 text-white/70" />
            )}
            <span className="hidden sm:inline">{isUploading ? 'Indexing...' : 'Upload PDF'}</span>
            <span className="sm:hidden">{isUploading ? '...' : 'PDF'}</span>
          </button>
        </div>

        {/* Right: Model Selector & Reset */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="flex items-center space-x-1 bg-white/[0.03] border border-white/[0.08] rounded-xl p-1">
            {displayModels.slice(0, 3).map((mId) => (
              <button
                key={mId}
                type="button"
                onClick={() => onSelectModel(mId)}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  selectedModel === mId
                    ? 'bg-white text-black shadow-sm font-semibold'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                {formatModelLabel(mId)}
              </button>
            ))}
          </div>

          {/* Clear chat */}
          <button
            type="button"
            onClick={clearChat}
            title="Clear conversation"
            className="p-1.5 rounded-xl text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Upload Status Banner */}
      {uploadMessage && (
        <div className="bg-cyan-500/10 border-b border-cyan-500/20 px-6 py-2 text-xs font-medium text-cyan-300 flex items-center gap-2 shrink-0 animate-fade-in">
          <Sparkles className="h-3.5 w-3.5 shrink-0" />
          <span>{uploadMessage}</span>
        </div>
      )}

      {/* Conversation List: Independently scrollable viewport */}
      <div
        ref={messagesContainerRef}
        onScroll={handleContainerScroll}
        className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 md:px-8 py-6 space-y-6 scroll-smooth"
      >
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="min-h-[55vh] flex flex-col items-center justify-center text-center max-w-lg mx-auto py-8">
              <div className="h-14 w-14 rounded-3xl bg-white/[0.04] border border-white/10 flex items-center justify-center mb-4 text-white/80 shadow-inner">
                <BookOpen className="h-7 w-7 text-cyan-400" />
              </div>
              <h3 className="text-xl font-semibold tracking-tight text-white">Ask your knowledge base</h3>
              <p className="text-xs sm:text-sm text-white/50 mt-1.5 leading-relaxed max-w-md">
                Active document:{' '}
                <span className="font-mono text-cyan-300 font-medium">
                  {activeDoc || 'No document selected yet'}
                </span>
                . Inquire about concepts, methods, formulas, or summaries below.
              </p>

              {/* Sample Prompt Chips */}
              <div className="mt-8 flex flex-wrap gap-2 justify-center max-w-md">
                {[
                  'Summarize the core takeaways',
                  'Explain key concepts and definitions',
                  'What are the primary rules & principles?',
                  'How do these components interact?',
                ].map((promptText) => (
                  <button
                    key={promptText}
                    type="button"
                    onClick={() => handleSend(promptText)}
                    className="text-xs bg-white/[0.04] hover:bg-white/[0.1] border border-white/[0.08] hover:border-white/20 text-white/80 hover:text-white px-3.5 py-2 rounded-xl transition-all shadow-sm"
                  >
                    {promptText}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${
                  msg.role === 'Human' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[90%] md:max-w-[80%] rounded-2xl px-5 py-4 ${
                    msg.role === 'Human'
                      ? 'bg-white text-black font-medium text-sm leading-relaxed shadow-lg shadow-white/5'
                      : 'bg-white/[0.04] border border-white/[0.09] text-white/90 text-sm leading-relaxed backdrop-blur-md shadow-md'
                  }`}
                >
                  {/* Role header for AI */}
                  {msg.role === 'AI' && (
                    <div className="flex items-center justify-between pb-2 mb-2 border-b border-white/[0.06] text-[11px] text-white/40">
                      <div className="flex items-center gap-1.5">
                        <Cpu className="h-3 w-3 text-cyan-400" />
                        <span className="font-mono text-white/60">Groq Response</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="flex items-center gap-1 hover:text-white transition-colors"
                      >
                        {copiedId === msg.id ? (
                          <>
                            <Check className="h-3 w-3 text-emerald-400" />
                            <span className="text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="h-3 w-3" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  <div className="whitespace-pre-wrap leading-relaxed">
                    {cleanTextOutput(msg.content)}
                  </div>

                  {/* Rewritten Standalone Query Badge */}
                  {msg.rewritten && (
                    <div className="mt-3 pt-2.5 border-t border-white/[0.06] flex items-start gap-1.5 text-[11px] text-white/50">
                      <Info className="h-3.5 w-3.5 text-cyan-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="text-white/40 font-mono">Reformulated query: </span>
                        <span className="text-white/80 italic">&ldquo;{msg.rewritten}&rdquo;</span>
                      </div>
                    </div>
                  )}

                  {/* Sources Badge */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 flex items-center justify-between pt-2 border-t border-white/[0.04]">
                      <button
                        type="button"
                        onClick={() => setActiveSourcesModal(msg.sources || null)}
                        className="inline-flex items-center space-x-1.5 text-[11px] font-mono text-cyan-400 hover:text-cyan-300 transition-colors bg-cyan-500/10 px-2.5 py-1 rounded-lg border border-cyan-500/20"
                      >
                        <Layers className="h-3 w-3" />
                        <span>View {msg.sources.length} Context Chunks</span>
                        <ArrowUpRight className="h-3 w-3" />
                      </button>
                      <span className="text-[10px] text-white/30 font-mono">{msg.timestamp}</span>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex items-start space-x-3">
              <div className="rounded-2xl bg-white/[0.04] border border-white/[0.09] px-5 py-4 text-xs text-white/70 flex items-center space-x-3 backdrop-blur-md shadow-md">
                <Loader2 className="h-4 w-4 animate-spin text-cyan-400" />
                <span>Searching Chroma vector index & streaming Groq response...</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Floating Jump to Bottom Button if user scrolled up */}
      {showScrollBottomBtn && (
        <button
          type="button"
          onClick={() => scrollChatToBottom('smooth')}
          className="absolute bottom-24 right-8 z-20 flex items-center space-x-1.5 bg-[#1a1b26]/90 border border-white/20 text-white text-xs px-3 py-1.5 rounded-full shadow-2xl backdrop-blur-md hover:bg-white hover:text-black transition-all"
        >
          <ArrowDown className="h-3.5 w-3.5 text-cyan-400" />
          <span>Latest response</span>
        </button>
      )}

      {/* Stationary Question Composer Anchored at Viewport Bottom */}
      <div className="border-t border-white/[0.08] bg-[#0c0c12]/95 backdrop-blur-xl p-3 sm:p-4 shrink-0 z-10">
        <div className="max-w-4xl mx-auto">
          <div className="relative rounded-2xl bg-[#161722] border border-white/15 focus-within:border-cyan-400/70 focus-within:ring-2 focus-within:ring-cyan-400/20 shadow-xl transition-all p-1.5 flex items-end">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              placeholder={
                activeDoc
                  ? `Ask anything about ${activeDoc}...`
                  : 'Select or upload a document to begin asking questions...'
              }
              rows={1}
              className="w-full bg-transparent px-3.5 py-2 text-sm text-white placeholder-white/40 resize-none focus:outline-none min-h-[38px] max-h-[140px] leading-relaxed"
            />
            <button
              type="button"
              onClick={() => handleSend()}
              disabled={isLoading || !input.trim()}
              title="Send question (Enter)"
              className="h-9 w-9 rounded-xl bg-white text-black flex items-center justify-center hover:bg-white/90 disabled:opacity-20 disabled:hover:bg-white transition-all shrink-0 ml-2 mb-0.5"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          <div className="flex items-center justify-between mt-2 px-2 text-[10px] sm:text-[11px] text-white/40 font-mono">
            <span>Press <kbd className="px-1 py-0.5 bg-white/10 rounded text-white/70">Enter</kbd> to send, <kbd className="px-1 py-0.5 bg-white/10 rounded text-white/70">Shift+Enter</kbd> for new line</span>
            <span className="hidden sm:inline">100% Context Grounded</span>
          </div>
        </div>
      </div>

      {/* Sources Modal Drawer */}
      {activeSourcesModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="relative w-full max-w-2xl max-h-[80vh] overflow-hidden rounded-3xl bg-[#0e0e13] border border-white/10 p-6 shadow-2xl flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
              <div className="flex items-center space-x-2">
                <Layers className="h-4 w-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Chroma Retrieved Context Chunks</h3>
              </div>
              <button
                type="button"
                onClick={() => setActiveSourcesModal(null)}
                className="text-xs text-white/50 hover:text-white px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                Close
              </button>
            </div>

            <div className="flex-1 overflow-y-auto mt-4 space-y-4 pr-1">
              {activeSourcesModal.map((source, idx) => (
                <div
                  key={idx}
                  className="rounded-2xl bg-white/[0.03] border border-white/[0.06] p-4 text-xs text-white/80 space-y-2"
                >
                  <div className="flex items-center justify-between text-[11px] text-white/40 font-mono">
                    <span>Chunk #{idx + 1}</span>
                    {source.metadata?.page !== undefined && (
                      <span className="bg-white/10 px-2 py-0.5 rounded text-white/70">
                        Page {String(source.metadata.page)}
                      </span>
                    )}
                  </div>
                  <p className="leading-relaxed whitespace-pre-wrap font-sans text-white/70">
                    {source.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
