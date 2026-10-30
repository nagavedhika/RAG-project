import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { Features } from './components/Features';
import { Architecture } from './components/Architecture';
import { Playground } from './components/Playground';
import { KnowledgeGraph } from './components/KnowledgeGraph';
import { Footer } from './components/Footer';

export function App() {
  const [currentView, setCurrentView] = useState<'chat' | 'graph' | 'overview'>('chat');
  const [documents, setDocuments] = useState<string[]>([]);
  const [activeDoc, setActiveDoc] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('openai/gpt-oss-120b');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [hasKey, setHasKey] = useState<boolean>(true);
  const [injectedPrompt, setInjectedPrompt] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setHasKey(data.has_env_groq_key);
        if (data.available_models && data.available_models.length > 0) {
          setAvailableModels(data.available_models);
        }
        if (data.default_model) {
          setSelectedModel(data.default_model);
        }
        if (data.indexed_documents && data.indexed_documents.length > 0) {
          setDocuments(data.indexed_documents);
          if (!activeDoc) {
            const preferred = ['language_models_basics', 'Java_Notes'];
            const found = preferred.find((p) => data.indexed_documents.includes(p));
            setActiveDoc(found || data.indexed_documents[0]);
          }
        }
      }
    } catch (err) {
      console.warn('Backend status check failed:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleUploadSuccess = (docName: string) => {
    setDocuments((prev) => (prev.includes(docName) ? prev : [docName, ...prev]));
    setActiveDoc(docName);
  };

  const handleQueryInChat = (query: string) => {
    setInjectedPrompt(query);
    setCurrentView('chat');
  };

  return (
    <div className="h-screen w-screen bg-[#050507] text-[#f5f5f7] flex flex-col font-sans selection:bg-white/20 selection:text-white overflow-hidden">
      {/* Sticky Clean Minimal Navbar with View Switcher */}
      <Navbar
        hasKey={hasKey}
        currentView={currentView}
        onSelectView={(view) => setCurrentView(view)}
      />

      <main className="flex-1 min-h-0 relative overflow-hidden">
        {/* Knowledge Graph View */}
        {currentView === 'graph' && (
          <KnowledgeGraph
            documents={documents}
            activeDoc={activeDoc}
            onSelectDoc={(doc) => setActiveDoc(doc)}
            onQueryInChat={handleQueryInChat}
          />
        )}

        {/* Playground / Chat View (Preserved via display styling to maintain chat state and message history) */}
        <div
          className="h-full w-full"
          style={{ display: currentView === 'chat' ? 'block' : 'none' }}
        >
          <Playground
            documents={documents}
            activeDoc={activeDoc}
            onSelectDoc={(doc) => setActiveDoc(doc)}
            onUploadSuccess={handleUploadSuccess}
            selectedModel={selectedModel}
            availableModels={availableModels}
            onSelectModel={(model) => setSelectedModel(model)}
            hasKey={hasKey}
            injectedPrompt={injectedPrompt}
            onClearInjectedPrompt={() => setInjectedPrompt(null)}
          />
        </div>

        {/* Overview & Architecture Marketing Landing View */}
        {currentView === 'overview' && (
          <div className="h-full w-full overflow-y-auto">
            <Hero
              onScrollToPlayground={() => setCurrentView('chat')}
              documentCount={documents.length}
            />
            <Features />
            <Architecture />
            <Footer />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
